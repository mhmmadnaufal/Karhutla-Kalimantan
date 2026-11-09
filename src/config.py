"""
Konfigurasi Terkunci (Locked Design Decisions) untuk Pipeline Data & Model
Neural-Parameterized Probabilistic Cellular Automata (DeepLabV3+ + CA) Kalimantan.
Berdasarkan PRD v2 dan hasil EDA Spasial-Temporal 2017–2025.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class GridConfig:
    """Spesifikasi grid spasial patch 256x256 pada resolusi 375 m."""
    pixel_size_m: float = 375.0
    patch_size: int = 256
    patch_extent_m: float = 96000.0  # 256 * 375 m = 96.0 km
    tile_overlap_ratio: float = 0.25
    utm_meridian_split_lon: float = 114.0  # Batas UTM 49 (<114E) dan UTM 50 (>=114E)

    @staticmethod
    def get_utm_epsg(center_lon: float, center_lat: float) -> int:
        """
        Menentukan EPSG UTM metrik per-patch berdasarkan lokasi centroid event
        agar ukuran piksel 375 m x 375 m dan sudut arah angin/slope bebas distorsi.
        - Lon < 114.0E : Zona 49 (Utara: 32649, Selatan: 32749)
        - Lon >= 114.0E: Zona 50 (Utara: 32650, Selatan: 32750)
        """
        zone = 49 if center_lon < 114.0 else 50
        if center_lat >= 0.0:
            return 32600 + zone
        return 32700 + zone


@dataclass(frozen=True)
class ClusteringConfig:
    """Spesifikasi penyaringan VIIRS 375 m dan klasterisasi spatiotemporal event."""
    start_year: int = 2017
    end_year: int = 2025
    # Bounding box Kalimantan [min_lon, min_lat, max_lon, max_lat]
    kalimantan_bbox: Tuple[float, float, float, float] = (108.5, -5.2, 119.1, 4.5)
    valid_confidences: Tuple[str, ...] = ("nominal", "high", "n", "h")
    # Filter sumber panas persisten (industri/tambang batu bara/flare)
    persistent_grid_deg: float = 0.01  # ~1.1 km grid untuk deteksi sumber panas statis
    persistent_min_days_per_year: int = 30  # Piksel aktif >= N hari/tahun atau >= 7 dari 9 tahun
    persistent_min_years: int = 7
    # Parameter klasterisasi spatiotemporal terkunci
    d_max_km: float = 5.0  # Baseline utama terkunci (dengan diagnostik 10.0 km)
    d_max_diagnostic_km: Tuple[float, ...] = (3.0, 5.0, 10.0, 15.0, 20.0)
    g_max_days: int = 2    # Toleransi celah hari tanpa deteksi akibat haze/awan


# Daftar 17 Kanal Konfigurasi Inti Tahap 1 ("Lite-17") sesuai PRD Bagian 1.4.2
LITE_17_CHANNELS: List[str] = [
    # Grup F: Riwayat Api (6 kanal)
    "fire_t",
    "fire_t_minus_1",
    "fire_t_minus_2",
    "count_14d",
    "valid_t",
    "fire_freq_prev_years",
    # Grup S: Statis / Bertahun (4 kanal)
    "peat",
    "landcover_idx",
    "elevation",
    "slope",
    # Grup D: Dinamis Teramati (5 kanal)
    "precip_7d",
    "precip_30d",
    "DC",
    "wind_u",
    "wind_v",
    # Grup G: Forecast GFS t -> t+24 (2 kanal)
    "gfs_wind_u",
    "gfs_wind_v",
]

# Daftar 32 Kanal Konfigurasi Penuh ("Core-32") sesuai PRD Bagian 1.4.1
CORE_32_CHANNELS: List[str] = [
    # Grup F: Riwayat Api (6 kanal)
    "fire_t", "fire_t_minus_1", "fire_t_minus_2", "count_14d", "valid_t", "fire_freq_prev_years",
    # Grup S: Statis / Bertahun (12 kanal)
    "peat", "landcover_idx", "oil_palm_class", "treecover_2000", "forest_loss_cum",
    "dist_road", "dist_waterway", "dist_settlement", "elevation", "slope", "aspect_sin", "aspect_cos",
    # Grup D: Dinamis Teramati (10 kanal)
    "precip_1d", "precip_7d", "precip_30d", "precip_90d", "DC", "FFMC", "RH",
    "soil_water_L1", "wind_u", "wind_v",
    # Grup G: Forecast GFS (4 kanal)
    "gfs_wind_u", "gfs_wind_v", "gfs_precip", "gfs_RH",
]

# Kanal Fisik CA (5 kanal): [V, theta_wind, slope_norm, theta_upslope, M_fuel]
CA_PHYS_CHANNELS: List[str] = [
    "wind_speed_mps",
    "wind_dir_rad",
    "slope_norm",
    "upslope_dir_rad",
    "burnable_mask",
]

# Katalog GEE Collection IDs sesuai PRD Bagian 1.3
GEE_CATALOG: Dict[str, str] = {
    "viirs_nrt": "NASA/LANCE/SNPP_VIIRS/C2",
    "vnp14a1_mask": "NASA/VIIRS/002/VNP14A1",
    "peat_gpm2": "projects/sat-io/open-datasets/GLOBAL-PEATLAND-DATABASE",
    "peat_ml": "projects/sat-io/open-datasets/ML-GLOBAL-PEATLAND-EXTENT",
    "landcover_modis": "MODIS/061/MCD12Q1",
    "oil_palm_2019": "projects/sat-io/open-datasets/landcover/oil_palm_industrial_smallholder_2019",
    "hansen_gfc": "UMD/hansen/global_forest_change_2023_v1_11",
    "srtm": "USGS/SRTMGL1_003",
    "worldpop": "WorldPop/GP/100m/pop",
    "imerg_v07": "NASA/GPM_L3/IMERG_V07",
    "cems_fire": "projects/climate-engine-pro/assets/ce-cems-fire-daily-4-1",
    "era5_land_daily": "ECMWF/ERA5_LAND/DAILY_AGGR",
    "gfs_forecast": "NOAA/GFS0P25",
}


@dataclass(frozen=True)
class ModelCAConfig:
    """Spesifikasi hiperparameter DeepLabV3+ Parameter Generator & Probabilistic CA."""
    in_channels_lite17: int = 17
    in_channels_core32: int = 32
    num_landcover_classes: int = 18  # IGBP classes 0..17
    landcover_embed_dim: int = 4
    num_ca_params: int = 6           # [p_base, alpha_w1, alpha_w2, alpha_s, gamma, fuel_factor]
    n_micro_steps: int = 15          # Kalibrasi dari {5, 10, 15, 30, 50}
    p_continue_base: float = 0.65    # Mode M0 skalar tetap
    use_adaptive_p_continue: bool = True  # Mode M1: p_continue sebagai fungsi DC, rain, & peat
    max_wind_speed_mps: float = 50.0
    # Bobot fungsi loss rujukan PRD Bagian 2.12
    w_frontier_bce: float = 0.5
    w_mse: float = 0.1
    w_area: float = 0.5
    w_iou: float = 4.0
    w_pool: float = 0.1
    w_tv_smooth: float = 0.01        # Regularisasi kelancaran spasial Total Variation
    frontier_weight: float = 10.0
