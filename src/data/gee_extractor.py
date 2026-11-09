"""
Modul Ekstraksi Tensor Prediktor & Mask Validitas dari Google Earth Engine (GEE)
menggunakan `ee.data.computePixels` langsung ke memori Python (NumPy/NPZ/Zarr).
Mematuhi Aturan Anti-Bocor Waktu (PRD Bagian 1.11) dan Konfigurasi Lite-17.
"""

import io
import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import GEE_CATALOG, LITE_17_CHANNELS, CA_PHYS_CHANNELS


def verify_anti_leakage_timestamps(
    date_t: str,
    obs_end_timestamp: str,
    gfs_creation_timestamp: str,
    static_year: int,
) -> bool:
    """
    Unit test otomatis pencegahan kebocoran waktu sesuai PRD Bagian 1.11 & 3.2:
    1. Seluruh prediktor observasi wajib memiliki timestamp <= akhir hari t.
    2. Forecast GFS wajib memiliki creation_time <= akhir hari t.
    3. Produk tahunan (MCD12Q1, Hansen, WorldPop) wajib bertahun <= tahun(t) - 1.
    """
    t_end = pd.to_datetime(date_t) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    obs_ts = pd.to_datetime(obs_end_timestamp)
    gfs_ts = pd.to_datetime(gfs_creation_timestamp)
    sample_year = pd.to_datetime(date_t).year

    if obs_ts > t_end:
        raise AssertionError(f"Leaked observation! obs_end={obs_ts} > t_end={t_end}")
    if gfs_ts > t_end:
        raise AssertionError(f"Leaked forecast! gfs_creation_time={gfs_ts} > t_end={t_end}")
    if static_year > sample_year - 1:
        raise AssertionError(f"Anachronistic annual map! static_year={static_year} > {sample_year - 1}")
    return True


def build_gee_lite17_image(
    date_t: str,
    latency_realistic_mode: bool = False,
):
    """
    Membangun objek `ee.Image` multi-band pada Google Earth Engine untuk tanggal `date_t`
    sesuai spesifikasi PRD Bagian 1.3 & 1.4.2 (Kanal Statis, Dinamis, Forecast, & FireMask).
    Catatan: Kanal riwayat api (fire_t, fire_t-1, fire_t-2, count_14d, fire_freq_prev_years)
    dihitung dari rasterisasi arsip FIRMS lokal di Python dan digabungkan setelah ekstraksi.
    """
    import ee

    dt = pd.to_datetime(date_t)
    t_start_str = dt.strftime("%Y-%m-%d")
    t_next_str = (dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    t_plus2_str = (dt + pd.Timedelta(days=2)).strftime("%Y-%m-%d")

    # Untuk mode operasional (latensi realistis PRD 3.10), CEMS & ERA5-Land di-lag 5-7 hari
    lag_days = 7 if latency_realistic_mode else 0
    dt_meteo = dt - pd.Timedelta(days=lag_days)
    meteo_start_str = dt_meteo.strftime("%Y-%m-%d")
    meteo_next_str = (dt_meteo + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    # Tahun peta tahunan <= tahun sampel - 1 (PRD Bagian 1.1 prinsip 5)
    annual_year = min(max(dt.year - 1, 2001), 2022)

    # 1. Mask Validitas Observasi VNP14A1 FireMask (PRD Bagian 1.5.1)
    # Kelas valid darat/api: {5, 6, 7, 8, 9} -> valid = 1; {0, 1, 2, 3 (air), 4 (awan)} -> valid = 0
    vnp14a1_col = (
        ee.ImageCollection(GEE_CATALOG["vnp14a1_mask"])
        .filterDate(t_start_str, t_next_str)
        .select("FireMask")
    )
    fire_mask_raw = ee.Algorithms.If(
        vnp14a1_col.size().gt(0),
        vnp14a1_col.max(),
        ee.Image.constant(5),  # fallback darat bersih jika koleksi kosong
    )
    fire_mask_img = ee.Image(fire_mask_raw)
    valid_t_img = fire_mask_img.gte(5).And(fire_mask_img.lte(9)).rename("valid_t").toFloat()
    water_mask_img = fire_mask_img.eq(3).rename("water_mask")

    # Mask validitas hari berikutnya (t+24) untuk target loss M_{t+24}
    vnp14a1_next_col = (
        ee.ImageCollection(GEE_CATALOG["vnp14a1_mask"])
        .filterDate(t_next_str, t_plus2_str)
        .select("FireMask")
    )
    fire_mask_next_img = ee.Image(
        ee.Algorithms.If(vnp14a1_next_col.size().gt(0), vnp14a1_next_col.max(), ee.Image.constant(5))
    )
    valid_next_img = fire_mask_next_img.gte(5).And(fire_mask_next_img.lte(9)).rename("valid_next").toFloat()

    # 2. Grup S (Statis / Bertahun): peat, landcover_idx, elevation, slope (+ aspect untuk fisik CA)
    peat_img = (
        ee.Image(GEE_CATALOG["peat_gpm2"])
        .gt(0)
        .unmask(0)
        .rename("peat")
        .toFloat()
    )

    lc_col = (
        ee.ImageCollection(GEE_CATALOG["landcover_modis"])
        .filterDate(f"{annual_year}-01-01", f"{annual_year}-12-31")
        .select("LC_Type1")
    )
    lc_img = ee.Image(
        ee.Algorithms.If(lc_col.size().gt(0), lc_col.first(), ee.Image.constant(2))
    ).unmask(0).rename("landcover_idx").toFloat()

    srtm = ee.Image(GEE_CATALOG["srtm"]).unmask(0)
    elevation_img = srtm.select("elevation").rename("elevation").toFloat()
    terrain = ee.Terrain.products(srtm)
    slope_img = terrain.select("slope").rename("slope").toFloat()
    aspect_deg_img = terrain.select("aspect").rename("aspect_deg").toFloat()

    # 3. Grup D (Dinamis Teramati): precip_7d, precip_30d (IMERG V07), DC (CEMS), wind_u, wind_v (ERA5-Land)
    d_minus_7 = (dt - pd.Timedelta(days=6)).strftime("%Y-%m-%d")
    d_minus_30 = (dt - pd.Timedelta(days=29)).strftime("%Y-%m-%d")

    # IMERG V07: half-hourly precipitation (mm/hr) -> kali 0.5 jam lalu jumlahkan
    imerg_7d = (
        ee.ImageCollection(GEE_CATALOG["imerg_v07"])
        .filterDate(d_minus_7, t_next_str)
        .select("precipitation")
        .map(lambda img: img.multiply(0.5))
        .sum()
        .unmask(0)
        .rename("precip_7d")
        .toFloat()
    )
    imerg_30d = (
        ee.ImageCollection(GEE_CATALOG["imerg_v07"])
        .filterDate(d_minus_30, t_next_str)
        .select("precipitation")
        .map(lambda img: img.multiply(0.5))
        .sum()
        .unmask(0)
        .rename("precip_30d")
        .toFloat()
    )

    cems_col = (
        ee.ImageCollection(GEE_CATALOG["cems_fire"])
        .filterDate(meteo_start_str, meteo_next_str)
        .select("drtcode")
    )
    dc_img = ee.Image(
        ee.Algorithms.If(cems_col.size().gt(0), cems_col.first(), ee.Image.constant(100.0))
    ).unmask(100.0).rename("DC").toFloat()

    era5_col = (
        ee.ImageCollection(GEE_CATALOG["era5_land_daily"])
        .filterDate(meteo_start_str, meteo_next_str)
        .select(["u_component_of_wind_10m", "v_component_of_wind_10m"])
    )
    era5_img = ee.Image(
        ee.Algorithms.If(
            era5_col.size().gt(0),
            era5_col.first(),
            ee.Image.constant([0.0, 0.0]).rename(["u_component_of_wind_10m", "v_component_of_wind_10m"]),
        )
    )
    wind_u_img = era5_img.select("u_component_of_wind_10m").unmask(0.0).rename("wind_u").toFloat()
    wind_v_img = era5_img.select("v_component_of_wind_10m").unmask(0.0).rename("wind_v").toFloat()

    # 4. Grup G (Forecast GFS untuk t -> t+24, dengan creation_time <= t)
    t_epoch_ms = int(pd.Timestamp(t_start_str, tz="UTC").timestamp() * 1000)
    gfs_col = (
        ee.ImageCollection(GEE_CATALOG["gfs_forecast"])
        .filter(ee.Filter.lte("creation_time", t_epoch_ms))
        .filter(ee.Filter.gt("forecast_time", t_epoch_ms))
        .filter(ee.Filter.lte("forecast_time", t_epoch_ms + 24 * 3600 * 1000))
        .select(["u_component_of_wind_10m_above_ground", "v_component_of_wind_10m_above_ground"])
    )
    gfs_mean = ee.Image(
        ee.Algorithms.If(
            gfs_col.size().gt(0),
            gfs_col.mean(),
            ee.Image.constant([0.0, 0.0]).rename([
                "u_component_of_wind_10m_above_ground",
                "v_component_of_wind_10m_above_ground",
            ]),
        )
    )
    gfs_u_img = gfs_mean.select("u_component_of_wind_10m_above_ground").unmask(0.0).rename("gfs_wind_u").toFloat()
    gfs_v_img = gfs_mean.select("v_component_of_wind_10m_above_ground").unmask(0.0).rename("gfs_wind_v").toFloat()

    # Gabungkan seluruh band GEE
    stacked_img = ee.Image.cat([
        valid_t_img,
        valid_next_img,
        peat_img,
        lc_img,
        elevation_img,
        slope_img,
        aspect_deg_img,
        imerg_7d,
        imerg_30d,
        dc_img,
        wind_u_img,
        wind_v_img,
        gfs_u_img,
        gfs_v_img,
    ])
    return stacked_img


def fetch_gee_patch_numpy(
    patch_spec: Dict[str, object],
    date_t: str,
    latency_realistic_mode: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Menarik array raster 256 x 256 pada resolusi 375 m langsung ke memori Python
    menggunakan `ee.data.computePixels` dengan grid affine transform & CRS UTM per-patch.
    """
    import ee

    img = build_gee_lite17_image(date_t=date_t, latency_realistic_mode=latency_realistic_mode)
    scale_x, shear_x, trans_x, shear_y, scale_y, trans_y = patch_spec["affine_transform"]

    request = {
        "expression": img,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {
                "width": patch_spec["width_px"],
                "height": patch_spec["height_px"],
            },
            "affineTransform": {
                "scaleX": scale_x,
                "shearX": shear_x,
                "translateX": trans_x,
                "shearY": shear_y,
                "scaleY": scale_y,
                "translateY": trans_y,
            },
            "crsCode": patch_spec["crs"],
        },
    }

    raw_bytes = ee.data.computePixels(request)
    structured_arr = np.load(io.BytesIO(raw_bytes))

    out_dict = {}
    for name in structured_arr.dtype.names:
        out_dict[name] = structured_arr[name].astype(np.float32)
    return out_dict


def assemble_lite17_sample_dict(
    fire_history_dict: Dict[str, np.ndarray],
    fire_freq_prev_years: np.ndarray,
    gee_bands_dict: Dict[str, np.ndarray],
    event_id: str,
    date_t: str,
    patch_spec: Dict[str, object],
    scalars: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Dict[str, object]:
    """
    Merakit dictionary lengkap sesuai kontrak berkas `.npz` / `.zarr` pada PRD Bagian 1.12 & 1.14:
      - x           : float32 [17, 256, 256] (Lite-17)
      - state_t     : float32 [3, 256, 256]  (P_U, P_F, P_B)
      - target_cum  : uint8   [1, 256, 256]  (C_{t+24})
      - target_new  : uint8   [1, 256, 256]  (Y_{t+24})
      - valid_next  : uint8   [1, 256, 256]  (M_{t+24})
      - ca_phys     : float32 [5, 256, 256]  (V, theta_wind, slope_norm, theta_upslope, M_fuel)
      - scalars     : float32 [3]            (ONI, DMI, sub-wilayah)
    """
    # Gabungkan kanal Lite-17 sesuai urutan LITE_17_CHANNELS
    channel_map = {
        "fire_t": fire_history_dict["fire_t"],
        "fire_t_minus_1": fire_history_dict["fire_t_minus_1"],
        "fire_t_minus_2": fire_history_dict["fire_t_minus_2"],
        "count_14d": fire_history_dict["count_14d"],
        "valid_t": gee_bands_dict.get("valid_t", fire_history_dict["valid_t"]),
        "fire_freq_prev_years": fire_freq_prev_years.astype(np.float32),
        "peat": gee_bands_dict["peat"],
        "landcover_idx": gee_bands_dict["landcover_idx"],
        "elevation": gee_bands_dict["elevation"],
        "slope": gee_bands_dict["slope"],
        "precip_7d": gee_bands_dict["precip_7d"],
        "precip_30d": gee_bands_dict["precip_30d"],
        "DC": gee_bands_dict["DC"],
        "wind_u": gee_bands_dict["wind_u"],
        "wind_v": gee_bands_dict["wind_v"],
        "gfs_wind_u": gee_bands_dict["gfs_wind_u"],
        "gfs_wind_v": gee_bands_dict["gfs_wind_v"],
    }

    x_tensor = np.stack([channel_map[ch] for ch in LITE_17_CHANNELS], axis=0).astype(np.float32)

    # Susun 5 kanal fisik CA sesuai PRD Bagian 1.12 & 2.7:
    # Gunakan forecast angin GFS jika tersedia (atau observasi ERA5-Land) untuk pendorong t -> t+24
    u_phys = gee_bands_dict["gfs_wind_u"]
    v_phys = gee_bands_dict["gfs_wind_v"]
    # Fallback ke ERA5 wind jika GFS bernilai 0 di seluruh patch
    if np.allclose(u_phys, 0.0) and np.allclose(v_phys, 0.0):
        u_phys = gee_bands_dict["wind_u"]
        v_phys = gee_bands_dict["wind_v"]

    wind_speed = np.clip(np.sqrt(u_phys ** 2 + v_phys ** 2), 0.0, 50.0)
    wind_dir_rad = np.arctan2(v_phys, u_phys)  # Konvensi matematika [-pi, pi]

    slope_norm = np.clip(gee_bands_dict["slope"] / 90.0, 0.0, 1.0)
    # Upslope direction: tambahkan 180 derajat dari arah aspect kompas, lalu konversi ke radian matematika
    aspect_compass_deg = gee_bands_dict.get("aspect_deg", np.zeros_like(slope_norm))
    upslope_compass_deg = (aspect_compass_deg + 180.0) % 360.0
    upslope_math_rad = np.deg2rad(90.0 - upslope_compass_deg)

    # Burnable mask M_fuel (PRD Bagian 2.6): hanya menutup air (IGBP class 0 atau 17) dan bangunan padat (class 13)
    lc = gee_bands_dict["landcover_idx"]
    burnable_mask = ((lc != 0) & (lc != 17) & (lc != 13)).astype(np.float32)

    ca_phys = np.stack([
        wind_speed,
        wind_dir_rad,
        slope_norm,
        upslope_math_rad,
        burnable_mask,
    ], axis=0).astype(np.float32)

    valid_next = gee_bands_dict.get("valid_next", fire_history_dict["valid_next"][0])[np.newaxis, ...].astype(np.uint8)

    return {
        "x": x_tensor,
        "state_t": fire_history_dict["state_t"].astype(np.float32),
        "target_cum": fire_history_dict["target_cum"].astype(np.uint8),
        "target_new": fire_history_dict["target_new"].astype(np.uint8),
        "valid_next": valid_next,
        "ca_phys": ca_phys,
        "scalars": np.array(scalars, dtype=np.float32),
        "event_id": str(event_id),
        "date": str(date_t),
        "feature_names": np.array(LITE_17_CHANNELS),
        "crs": str(patch_spec["crs"]),
        "transform": np.array(patch_spec["affine_transform"], dtype=np.float64),
        "bbox": np.array(patch_spec["bounds_utm"], dtype=np.float64),
    }

