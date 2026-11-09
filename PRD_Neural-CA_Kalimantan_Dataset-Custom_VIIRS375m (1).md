# Product Research Design

## Model Prediksi Propagasi Kebakaran Hutan dan Lahan 24–72 Jam di Kalimantan dengan Neural-Parameterized Probabilistic Cellular Automata (DeepLabV3+) dan Dataset Custom Berbasis VIIRS 375 m

*Versi 2 (dataset custom). Disusun 31 Oktober 2026. Pipeline utama: Google Earth Engine (GEE) dan Python.*

---

## 0. Cara membaca dokumen ini

### 0.1 Tiga status keputusan

Setiap keputusan desain diberi satu dari tiga status. Tujuannya agar jelas mana yang didukung literatur dan mana yang masih hipotesis yang harus diuji.

| Status | Arti | Konsekuensi |
|---|---|---|
| **[R]** Rujukan | Didukung langsung oleh publikasi atau dokumentasi produk yang disebut | Dipakai sebagai baseline |
| **[A]** Adaptasi | Didukung sebagian oleh literatur, dengan penyesuaian untuk Kalimantan | Dipakai, parameternya divalidasi |
| **[H]** Hipotesis desain | Tidak ada rujukan langsung; berasal dari penalaran teknis | Wajib diuji lewat ablasi atau uji sensitivitas, tidak boleh diklaim sebagai fakta |

Rujukan ditulis dengan kunci `[Penulis Tahun]` dan daftar lengkapnya ada di bagian Referensi.

### 0.2 Status verifikasi sumber

Dalam daftar referensi dan tabel dataset:

- **✔** berarti halaman katalog, dokumentasi, atau artikelnya diakses dan dibaca saat dokumen ini disusun.
- **○** berarti rujukan standar dari pengetahuan penyusun yang belum dibuka ulang. **DOI, ID koleksi, dan rentang tahun harus dicek sebelum naskah dikirim.**

### 0.3 Koreksi terhadap jawaban sebelumnya dalam percakapan ini

Verifikasi sumber menunjukkan beberapa pernyataan sebelumnya perlu dikoreksi. Dokumen ini sudah memakai versi yang benar.

1. **NDWS memakai MODIS, bukan VIIRS.** Label Next Day Wildfire Spread berasal dari MOD14A1 (MODIS Terra, 1 km) [Huot2022]. Yang memakai VIIRS 375 m adalah WildfireSpreadTS dan TS-SatFire [Gerard2023].
2. **Atribut `type == 0` (vegetation fire).** Dokumentasi NASA FIRMS menandai atribut `Type` sebagai tersedia hanya untuk produk MCD14ML (MODIS standard quality). Saya sebelumnya menyarankan filter ini untuk VIIRS. **Cek kolom `type` pada berkas VIIRS arsip yang Anda unduh.** Jika tidak ada, gunakan mask sumber panas statis (Bagian 1.7).
3. **Ketersediaan VIIRS 375 m di GEE.** Koleksi `NASA/LANCE/SNPP_VIIRS/C2` hanya tersedia mulai 2023-09-03 [LANCE-GEE]. Arsip historis 2012–2023 harus diambil dari arsip FIRMS lewat Python. WildfireSpreadTS juga mencatat bahwa VIIRS AF adalah satu-satunya sumber yang tidak ada di GEE [Gerard2023].
4. **Band VIIRS spektral.** Saya sebelumnya menyebut band I1–I5 membawa "kebocoran label". Itu terlalu kuat. Observasi pada waktu ≤ t sah dipakai sebagai prediktor. Masalah sebenarnya adalah beban pemrosesan L1B dan gangguan awan. WildfireSpreadTS memakai VNP09GA (tersedia di GEE) dan menunjukkan bahwa I1, I2, M11 sedikit meningkatkan kinerja pada input 1–2 hari [Gerard2023]. Maka band tersebut dijadikan **grup opsional via VNP09GA**, bukan L1B.
5. **Mask awan sudah ada di GEE dalam bentuk kasar.** `NASA/VIIRS/002/VNP14A1` (1 km, band 750 m) memuat band `FireMask` dengan kelas awan, air, dan darat [VNP14A1-GEE]. Ini dipakai sebagai mask validitas utama (Bagian 1.5).

---

## Ringkasan Penelitian

Penelitian ini membangun model prediksi **perluasan kebakaran 24–72 jam ke depan di Kalimantan**. Mekanisme prediksi adalah **Probabilistic Cellular Automata (CA)** tiga keadaan dengan parameter spasial yang dihasilkan jaringan saraf, mengikuti [Zhenirovskyy2026]. Pada penelitian ini jaringan parameter diganti **DeepLabV3+** [Chen2018].

Dataset dibuat custom dengan prinsip berikut.

- **Target** adalah deteksi api aktif VIIRS 375 m (S-NPP) [Schroeder2014; Schroeder2025-UG], yang juga dipakai WildfireSpreadTS pada resolusi 375 m [Gerard2023].
- **Prediktor** dipilih berdasarkan literatur pendorong kebakaran Indonesia (kekeringan terkait ENSO/IOD, gambut, perubahan tutupan lahan, ignisi antropogenik) [Field2016; Nikonovas2022; Horton2022; Cattau2016].
- **Struktur dataset** mengikuti praktik NDWS dan WildfireSpreadTS: kelas tak teramati pada label [Huot2022], tampilan multi-hari dan proksi riwayat api [Gerard2023], serta pembagian berbasis tahun [Gerard2023; Nikonovas2022].

Formulasi utama:

$$
X_t \;\rightarrow\; \text{DeepLabV3+} \;\rightarrow\; \Theta_t \;\rightarrow\; \text{Probabilistic CA} \;\rightarrow\; P_{\mathrm{fire},t+24}
$$

dengan

$$
\Theta_t=[p_{base},\,\alpha_{w1},\,\alpha_{w2},\,\alpha_s,\,\gamma,\,fuel_{factor}]
$$

Simulasi dilanjutkan secara autoregresif sampai $t+72$ jam. Parameter, persamaan CA, dan fungsi loss rujukan diambil dari [Zhenirovskyy2026].

### Perbedaan utama dari PRD versi 1 (TS-SatFire-style)

| Aspek | PRD v1 | PRD v2 (dokumen ini) | Dasar |
|---|---|---|---|
| Prediktor VIIRS L1B (I1–I5, M11) | Inti | Dikeluarkan dari inti; opsional via VNP09GA | [Gerard2023]: fitur tambahan tidak selalu menaikkan kinerja |
| Mask label tak teramati | Belum ada | Wajib, dari `FireMask` VNP14A1 | [Huot2022; Gerard2023] |
| Prediktor kekeringan | KBDI, PDSI→soil moisture | Drought Code, FFMC, hujan IMERG bertingkat | [Field2016; Nikonovas2022; Field2015-GFWED] |
| Gambut dan akses manusia | Tidak ada | Ada | [Nikonovas2022; Horton2022; Cattau2016] |
| Unit sampel | Event 6+ hari | Event-day dengan ROI tetap; tanpa filter durasi keras | [Gerard2023; Huot2022] |
| Split data | Per tahun tetap | Leave-one-year-out + blok spasial | [Gerard2023; Nikonovas2022; Roberts2017] |
| Klaim komparabilitas | "Adaptasi TS-SatFire" | Dataset custom; komparabilitas dengan paper AS hanya pada protokol | Lihat Bagian 3.11 |

---

# 1. Dataset

## 1.1 Prinsip desain dataset

1. **Target dan prediktor dipisahkan secara waktu.** Semua prediktor memakai observasi dengan timestamp ≤ t. Forecast hanya dipakai bila `creation_time ≤ t`. **[R]** Prinsip ini sama dengan aturan anti-leakage pada PRD v1 dan sejalan dengan prinsip NDWS yang memakai data tersedia secara reguler agar dapat diterjemahkan ke prediksi dekat-waktu-nyata [Huot2022].
2. **Pilih prediktor yang bebas awan dan dapat diakses terbuka.** Alasannya: label VIIRS sendiri terganggu awan dan asap [Gerard2023], dan pada kondisi haze Kalimantan fitur optik harian cenderung ikut terganggu. **[A]** (alasan label: [Gerard2023]; ekstensi ke prediktor optik: penalaran penyusun, lihat Bagian 1.2 Tier D).
3. **Prioritaskan pendorong kebakaran yang terdokumentasi untuk Indonesia.** Pendorong yang dirujuk: kekeringan akibat El Niño dan IOD [Field2016; Pan2018], gambut [Nikonovas2022; Page2002], hilangnya hutan primer dan konversi ke perkebunan [Nikonovas2020; Nikonovas2022], serta ignisi antropogenik [Cattau2016; Horton2022]. **[R]**
4. **Jaga jumlah fitur tetap terkendali.** WildfireSpreadTS melaporkan bahwa menambah fitur atau hari input sering tidak menaikkan, bahkan menurunkan, kinerja karena kesulitan optimasi pada ruang input berdimensi tinggi [Gerard2023]. Karena itu fitur disusun dalam **grup modular** dan diuji bertahap (Bagian 3.7). **[R]**
5. **Konsistensi temporal peta statis.** Peta statis bertahun tunggal (misalnya peta tutupan lahan 2020) tidak boleh dipakai begitu saja untuk event 2014 tanpa dicatat sebagai anakronisme. ProbFire secara eksplisit membatasi dataset non-iklim pada produk beresolusi waktu tahunan agar sistem dapat dioperasikan [Nikonovas2022]. Aturan di sini: pakai produk bertahun (MCD12Q1, Hansen `lossyear`, WorldPop tahunan) dengan tahun ≤ tahun sampel − 1, dan catat setiap peta statis sebagai batasan. **[A]**
6. **Semua prediktor diharmonisasi ke satu raster 375 m**, seperti WildfireSpreadTS yang me-resample seluruh fitur ke 375 m memakai resampling bilinear internal GEE [Gerard2023]. Catatan: variabel cuaca bersumber 11–28 km sehingga tampak blok pada 375 m [Gerard2023]. Desain penanganannya ada di Bagian 1.4. **[R]**
7. **Seluruh pipeline dapat dijalankan dengan GEE + Python.** Data tambahan di luar GEE dibatasi pada: arsip FIRMS, OpenStreetMap, peta gambut Kementerian Pertanian, dan indeks ONI/DMI (Lampiran A).

---

## 1.2 Grid dan ukuran patch

| Parameter | Nilai | Dasar | Status |
|---|---|---|---|
| Resolusi grid | 375 m | Sama dengan target VIIRS I-band [Schroeder2014]; WildfireSpreadTS [Gerard2023] | **[R]** |
| Ukuran patch | 256 × 256 (96 × 96 km) | WildfireSpreadTS memakai jendela 1°×1° (±111 km; gambar 207–356 piksel) [Gerard2023]; NDWS memakai 64 km untuk menangkap "ukuran api aktif tipikal" [Huot2022] | **[A]** |
| Proyeksi | Satu CRS metrik untuk seluruh Kalimantan (mis. UTM 49N/50N atau equal-area) | Keputusan teknis; wilayah melintasi beberapa zona UTM | **[H]** |

Catatan teknis: ukuran piksel VIIRS yang sebenarnya membesar ke tepi swath. FIRMS menyediakan atribut `scan` dan `track` sebagai ukuran piksel aktual [FIRMS-attr]. Rasterisasi deteksi dengan footprint tersebut diuji sebagai analisis sensitivitas terhadap rasterisasi titik pusat.

---

## 1.3 Sumber dan fitur prediktor

### 1.3.1 Target dan mask validitas

| Peran | Sumber (ID) | Resolusi | Waktu | Akses | Dasar |
|---|---|---|---|---|---|
| **Target + state api** | VNP14IMG standard processing, S-NPP (arsip FIRMS) | 375 m | 2012-01-20 → kini [FIRMS-archive] | **Python** (unduh arsip FIRMS) | [Schroeder2014; Gerard2023] |
| Target operasional (NRT) | `NASA/LANCE/SNPP_VIIRS/C2` | 375 m | 2023-09-03 → kini; harian [LANCE-GEE] | **GEE** | [LANCE-GEE] |
| **Mask validitas observasi** | `NASA/VIIRS/002/VNP14A1` (band `FireMask`) | 1 km (band 750 m) | 2012-01-19 → kini; harian [VNP14A1-GEE] | **GEE** | [VNP14A1-GEE; Schroeder2025-UG] |
| Validasi mask (subset) | VNP14IMG swath (HDF/NetCDF), kelas awan 375 m | 375 m | harian | Python (LAADS/LANCE) | [Schroeder2025-UG] |

Catatan produk:

- Data NRT di GEE tidak dianggap berkualitas ilmiah menurut katalog, sehingga model dilatih pada arsip standard processing dan **diuji terpisah pada NRT** (Bagian 3.10) [LANCE-GEE].
- Koleksi LANCE di GEE hanya berisi piksel api (band `confidence`, `frp`, `Bright_ti4`, `Bright_ti5`, dst.) dan **tidak memuat informasi awan atau tak teramati** [LANCE-GEE]. Karena itu mask validitas harus diambil dari produk lain (VNP14A1 atau swath VNP14IMG).
- VIIRS 375 m memiliki dua sampai empat deteksi per hari menurut lintang [Gerard2023]. Semua deteksi dalam satu jendela 24 jam digabung menjadi satu mask harian, mengikuti WildfireSpreadTS [Gerard2023].

### 1.3.2 Prediktor

Tier A–C adalah **inti**. Tier D adalah **opsional** yang masuk lewat ablasi.

**Tier A: Riwayat api (memori)**

| Fitur | Sumber | Alasan | Status |
|---|---|---|---|
| `fire_t`, `fire_t-1`, `fire_t-2` | Target (arsip FIRMS) | Fire mask sebelumnya adalah fitur terpenting pada NDWS; menghapusnya menurunkan AUC-PR dari 28.4 ke 6.8 [Huot2022]. WildfireSpreadTS memakai 1–5 hari input dan melaporkan manfaat input multi-hari [Gerard2023] | **[R]** |
| `count_14d` (jumlah hari deteksi dalam 14 hari) | Target | Merangkum riwayat; panjang jendela ditetapkan empiris | **[H]** |
| `valid_t` (mask validitas observasi hari t) | VNP14A1 `FireMask` | Piksel tak teramati diperlakukan khusus pada NDWS (kelas −1) [Huot2022] | **[R]** |
| `fire_freq_prev_years` (maks. hitungan bulanan deteksi pada tahun-tahun sebelumnya, atau frekuensi) | Target (arsip) | ProbFire memakai maksimum hitungan deteksi bulanan pada tahun sebelumnya karena area yang pernah terbakar cenderung terbakar lagi (umpan balik api–vegetasi) [Nikonovas2022]; frekuensi kebakaran lahan gambut Indonesia dikaji oleh [Vetrita2020] | **[R]** |

**Tier B: Statis atau bertahun (tanpa gangguan awan)**

| Fitur | Sumber (ID) | Resolusi asli | Alasan | Status |
|---|---|---|---|---|
| Sebaran gambut | `projects/sat-io/open-datasets/GLOBAL-PEATLAND-DATABASE` (Global Peatland Map 2.0; 1 = dominan gambut, 2 = gambut dalam mosaik tanah) [GPM2.0] ✔; alternatif fraksi kontinu `projects/sat-io/open-datasets/ML-GLOBAL-PEATLAND-EXTENT` (Peat-ML) [Melton2022] ✔; opsional peta gambut Indonesia [Haryono2011] (data eksternal) | 1 km | Wilayah gambut mengalami laju kebakaran beberapa kali lebih tinggi daripada non-gambut [Nikonovas2022]; pada 2015 api menjalar ke bawah tanah di gambut yang terdrainase dan terdegradasi [Field2016] | **[R]** |
| Ketebalan gambut (opsional) | PEATGRIDS (katalog komunitas GEE; cek halaman katalog untuk path dan sitasi) ✔ | — | Hipotesis: kedalaman gambut mempengaruhi lama membara; tidak ada rujukan prediktif di dokumen ini | **[H]** |
| Tutupan lahan | `MODIS/061/MCD12Q1` (IGBP, tahunan) [Friedl2022] ○, sebagai **indeks kelas yang diembedding** | 500 m | Dipakai pada WildfireSpreadTS [Gerard2023]; embedding kelas untuk fuel factor mengikuti [Zhenirovskyy2026]. Untuk Kalimantan dibandingkan dengan peta lebih halus lewat ablasi (mis. ESA WorldCover [Zanaga2022] ○, tetapi hanya 2020–2021 sehingga anakronistik untuk tahun lain) | **[A]** |
| Perkebunan sawit | Peta sawit Descals dkk. [Descals2021] (10 m, 2019; `projects/sat-io/open-datasets/landcover/oil_palm_industrial_smallholder_2019` ✔) dan pembaruan 2021 + lapisan tahun tanam 30 m 1990–2021 [Descals2024] ○ | 10 m / 30 m | Perkebunan dan lahan pertanian memiliki aktivitas api lebih tinggi daripada hutan primer [Nikonovas2022]. **Pakai lapisan tahun tanam agar mask sawit konsisten dengan tahun sampel** | **[A]** |
| Tutupan pohon 2000 dan kehilangan tutupan | `UMD/hansen/global_forest_change_<rilis>` (cek rilis terbaru) [Hansen2013] | 30 m | ProbFire memakai tutupan hutan primer, kehilangan hutan primer dan sekunder pada tahun sebelumnya [Nikonovas2022]. Hutan primer lembap tahan api, sedangkan perkebunan dan lahan pertanian lebih rentan [Nikonovas2022; Nikonovas2020] | **[R]** |
| Jarak ke jalan | GRIP (`projects/sat-io/...` pada katalog komunitas) [Meijer2018] ○ atau OSM [Boeing2017] ○ | — | Jarak ke jalan, kanal/sungai, dan permukiman dipakai sebagai prediktor kebakaran gambut di Kalimantan Tengah [Horton2022]; sumber ignisi antropogenik diuraikan di [Cattau2016] | **[R]** |
| Jarak ke kanal/sungai | OSM `waterway=canal/river` via OSMnx/Overpass (**Python, data eksternal**) | — | [Horton2022]. **Keterbatasan:** kelengkapan kanal di OSM Kalimantan belum diketahui; harus diaudit | **[A]** |
| Jarak ke permukiman, kepadatan penduduk | `WorldPop/GP/100m/pop` [Tatem2017] ○ (tahunan) | 100 m | NDWS memakai kepadatan penduduk sebagai proksi aktivitas antropogenik [Huot2022]. WildfireSpreadTS tidak memakainya karena fokus pada api yang sudah menyala [Gerard2023]; kontribusinya untuk **spread** diuji lewat ablasi | **[A]** |
| Elevasi, slope, aspect | `USGS/SRTMGL1_003` [Farr2007] ✔ | 30 m | Dipakai pada NDWS dan WildfireSpreadTS; grup topografi adalah grup non-vegetasi terbaik pada ablasi WildfireSpreadTS [Gerard2023]. Slope dan aspect juga diperlukan input fisik CA [Zhenirovskyy2026]. Catatan: dataran gambut hampir datar sehingga informasi slope rendah (penalaran penyusun; diuji lewat ablasi CA tanpa suku slope) | **[A]** |

**Tier C: Dinamis**

| Fitur | Sumber (ID) | Resolusi | Latensi | Alasan | Status |
|---|---|---|---|---|---|
| Hujan harian dan akumulasi (1, 7, 30, 90 hari; jendela final ditetapkan empiris) | `NASA/GPM_L3/IMERG_V07` [Huffman-IMERG] ✔ (30 menit; produk provisional digantikan versi final; tidak ada produk "permanent" setelah 2025-09-30 karena transisi V08 ✔) | 0,1° | Provisional: jam–hari | Ambang hujan musim kering sekitar 4 mm/hari menentukan intensitas kebakaran Indonesia, dengan sensitivitas nonlinier yang meningkat di Kalimantan [Field2016]. ProbFire memakai presipitasi 5 bulan sebelumnya untuk menangkap penumpukan kekeringan; panjang jendela ditentukan empiris [Nikonovas2022]. Presipitasi MERRA bias di Asia Tenggara, sehingga sumber presipitasi alternatif perlu dipertimbangkan [Field2015-GFWED] | **[R]** untuk fitur; **[H]** untuk lebar jendela |
| Drought Code (DC), Fine Fuel Moisture Code (FFMC), Fire Weather Index (FWI); opsional KBDI | CEMS Fire Danger Indices, `projects/climate-engine-pro/assets/ce-cems-fire-daily-4-1` [Vitolo2020] ✔ (0,25°, harian, 1940–kini, keterlambatan ±1 minggu di GEE) | ~25 km | ~7 hari | DC melacak potensi api pertanian menjalar ke bawah tanah di gambut dan dipakai dalam sistem peringatan Indonesia [GFWED-SVS]; GFWED ditujukan untuk kalibrasi model prediksi berbasis FWI [Field2015-GFWED]; Shawki dkk. memprediksi DC untuk musim 2015 [Shawki2017]. 78% hotspot pada materi LAPAN terjadi saat FFMC > 81, padahal hanya 20% hari [LAPAN2011] | **[R]** |
| Angin 10 m (u, v), kelembapan relatif, suhu 2 m | `ECMWF/ERA5_LAND/DAILY_AGGR` [MunozSabater2019] ✔ (11 km; tersedia dari 1950 ✔) | 11 km | ±5 hari (CDS ERA5-Land-T) hingga 3 bulan (versi final) ✔ | Angin dibutuhkan CA [Zhenirovskyy2026] (yang juga memakai angin ERA5-Land). RH dihitung dari suhu 2 m dan titik embun dengan pendekatan August–Roche–Magnus, seperti ProbFire [Nikonovas2022] | **[R]** |
| Kelembapan tanah lapis 1 (opsional lapis 2–3) | `ECMWF/ERA5_LAND/DAILY_AGGR` | 11 km | idem | Kajian ENSO–hotspot Kalimantan memasukkan kelembapan tanah sebagai faktor kebasahan gambut [Quaest2023]; analisis Sumatra memakai ERA5 dan SMAP [ForGeo2025] | **[A]** |
| Indeks ENSO (ONI) dan IOD (DMI) | NOAA PSL (tabel; **eksternal, skalar**) [Saji1999] ○ | skalar bulanan | beberapa minggu | Kondisi lebih kering terkait El Niño dan fase positif IOD; wilayah Kalimantan Selatan didominasi El Niño, sedangkan Kalimantan Barat didominasi IOD [Nikonovas2022; Pan2018] | **[A]** |
| Identitas sub-wilayah (kategori) | Provinsi/pembagian pulau | — | — | ProbFire memakai penanda sub-wilayah karena sensitivitas kekeringan berbeda antar wilayah [Nikonovas2022] | **[R]** |

**Tier C′: Forecast (untuk macro-step 24, 48, 72 jam)**

| Fitur | Sumber (ID) | Resolusi | Dasar | Status |
|---|---|---|---|---|
| Angin 10 m (u, v), presipitasi, kelembapan (RH) hasil forecast | `NOAA/GFS0P25` (forecast per jam; siklus 6 jam) | 27,83 km | WildfireSpreadTS memakai 24 prediksi per jam dari GFS pada tengah malam hari yang baru, diagregasi menjadi satu prediksi [Gerard2023] | **[R]** (prosedur); **[H]** (konfigurasi untuk tropis) |

**Peringatan bukti.** Pada WildfireSpreadTS, kombinasi terbaik adalah vegetasi + tutupan lahan + topografi + cuaca dengan 5 hari input; **menambah forecast, ERC, dan drought justru menurunkan kinerja U-Net** [Gerard2023]. Karena itu fitur forecast dan indeks kekeringan harus lolos ablasi (Bagian 3.7). Pada model ini forecast angin dan hujan masuk ke mekanisme fisik CA, sehingga perannya berbeda dari sekadar kanal tambahan pada jaringan segmentasi.

**Tier D: Opsional (diuji sebagai ablasi, bukan fondasi)**

| Fitur | Sumber | Alasan | Status |
|---|---|---|---|
| Reflektansi VIIRS I1, I2, M11 | `VNP09GA` pada GEE (cek ID koleksi pada katalog; WildfireSpreadTS menyebut seluruh datanya di GEE kecuali VIIRS AF [Gerard2023]) | Pada input 1–2 hari, reflektansi VIIRS mengungguli NDVI (AP 0,314 vs 0,298 pada 1 hari) [Gerard2023]. Perlu mask kualitas awan | **[R]** (manfaat di AS); **[H]** (di Kalimantan) |
| NDVI/EVI | `NASA/VIIRS/002/VNP13A1` | NDVI/EVI2 kinerjanya mirip VIIRS pada ≥3 hari [Gerard2023]; NDVI adalah fitur vegetasi terbaik pada ablasi NDWS [Huot2022]. Komposit 16 hari: gunakan komposit yang **jendelanya berakhir ≤ t** (cek metadata) | **[A]** |
| Fire Radiative Power (FRP) | Arsip FIRMS / band `frp` LANCE | Tersedia pada produk AF [LANCE-GEE]; belum ada bukti prediktif pada rujukan yang dikaji | **[H]** |
| `CHIRPS` sebagai hujan alternatif | `UCSB-CHG/CHIRPS/DAILY` [Funk2015] ○ | Pembanding IMERG | **[H]** |

### 1.3.3 Posisi terhadap NDWS, WildfireSpreadTS, dan TS-SatFire

| Aspek | NDWS [Huot2022] | WildfireSpreadTS [Gerard2023] | TS-SatFire [Zhao-TSSatFire] | Dataset ini |
|---|---|---|---|---|
| Sumber fire mask | MOD14A1 (MODIS, 1 km) | VIIRS AF 375 m (S-NPP) | VIIRS | VIIRS AF 375 m (S-NPP) |
| Cakupan | AS, 2012–2020 | AS, 2018–2021, 607 event | AS (regional) | Kalimantan |
| Struktur | Pasangan hari (t, t+1) | Deret waktu per event | Deret waktu per event | Deret waktu per event-day, ROI tetap |
| Kelas tak teramati | Ya (−1) | Label berisik diakui, tanpa kelas khusus | — | Ya, dari VNP14A1 |
| Cuaca | GRIDMET | GRIDMET + GFS | GRIDMET + GFS | ERA5-Land + CEMS + IMERG + GFS |
| Fitur khas tropis | Tidak ada | Tidak ada | Tidak ada | Gambut, sawit, akses manusia, ENSO/IOD |

Pernyataan jujur tentang komparabilitas: karena wilayah, sumber cuaca, dan label berbeda, **angka metrik tidak dapat dibandingkan langsung** dengan paper AS. Yang dapat direproduksi adalah protokol (baseline persistence, regresi logistik, U-Net, validasi per tahun) [Huot2022; Gerard2023].

---

## 1.4 Susunan channel akhir

### 1.4.1 Konfigurasi inti ("Core-32")

| Grup | Channel | Jumlah |
|---|---|---:|
| **F** Riwayat api | `fire_t`, `fire_t-1`, `fire_t-2`, `count_14d`, `valid_t`, `fire_freq_prev_years` | 6 |
| **S** Statis/bertahun | `peat`, `landcover_idx`, `oil_palm_class`, `treecover_2000`, `forest_loss_cum`, `dist_road`, `dist_waterway`, `dist_settlement`, `elevation`, `slope`, `aspect_sin`, `aspect_cos` | 12 |
| **D** Dinamis teramati | `precip_1d`, `precip_7d`, `precip_30d`, `precip_90d`, `DC`, `FFMC`, `RH`, `soil_water_L1`, `wind_u`, `wind_v` | 10 |
| **G** Forecast GFS | `gfs_wind_u`, `gfs_wind_v`, `gfs_precip`, `gfs_RH` | 4 |
| **Total raster** | | **32** |
| Skalar per sampel | ONI, DMI, sub-wilayah | 3 |

`landcover_idx` dan `oil_palm_class` adalah indeks kategorik yang diproses lewat embedding (seperti embedding FBFM40 pada [Zhenirovskyy2026]), sehingga dimensi efektif masukan jaringan lebih besar dari 32.

### 1.4.2 Konfigurasi bertahap

Karena penambahan fitur tidak selalu menolong [Gerard2023], pengembangan dimulai dari konfigurasi kecil:

| Tahap | Isi | Channel |
|---|---|---:|
| **Lite-17** | F (6) + `peat`, `landcover_idx`, `elevation`, `slope` (4) + `precip_7d`, `precip_30d`, `DC`, `wind_u`, `wind_v` (5) + `gfs_wind_u`, `gfs_wind_v` (2) | 17 |
| **Core-32** | Tabel 1.4.1 | 32 |
| **Core-32 + D** | + `VNP09GA` I1, I2, M11; NDVI/EVI; FRP | ≈ 40 |

Pembandingan Lite-17 → Core-32 → +D menjadi ablasi grup fitur (Bagian 3.7). **[A]** (strategi bertahap berdasar [Gerard2023]).

---

## 1.5 Mask validitas dan data hilang

### 1.5.1 Mask validitas label

Fire mask NDWS mencantumkan kelas khusus untuk label tak pasti (awan atau data tak terproses) dan mengabaikannya dalam loss dan metrik [Huot2022]. WildfireSpreadTS mengakui label berisik akibat asap, awan, dan ketidaktepatan deteksi [Gerard2023]. Pada Kalimantan, tutupan awan dan haze pada musim kebakaran membuat hal ini lebih penting, dan **tanpa mask, "tidak ada deteksi" tidak dapat dibedakan dari "tidak teramati"**.

Kelas `FireMask` VNP14A1 dan VNP14IMG [VNP14A1-GEE; Schroeder2025-UG; LAADS-spec]:

| Kelas | Arti | Perlakuan |
|---:|---|---|
| 0 | Tidak diproses (data hilang/kualitas buruk) | **Tak teramati** |
| 1 | Tidak diproses (bowtie) | **Tak teramati** |
| 2 | Glint (hanya 375 m)/tidak dipakai | **Tak teramati** |
| 3 | Air | **Dikeluarkan** (tidak terbakar) |
| 4 | Awan | **Tak teramati** |
| 5 | Darat bersih | **Teramati, tanpa api** |
| 6 | Piksel api, kepercayaan tak terklasifikasi (menurut spesifikasi LAADS) | Teramati; tidak dihitung positif kecuali ada konfirmasi (analisis sensitivitas) |
| 7 | Api, kepercayaan rendah | Teramati; **dibuang dari positif** [Gerard2023 menghapus deteksi low-confidence] |
| 8, 9 | Api, kepercayaan nominal dan tinggi | **Positif** |

**Prosedur:** `valid(x,t)=1` bila kelas ∈ {5, 6, 7, 8, 9}; selain itu `valid=0`. Sel yang tak valid **dikeluarkan dari loss dan metrik** (pada kedua label, t dan t+1), seperti NDWS [Huot2022]. **[A]**

**Keterbatasan yang harus dicatat:**

1. `VNP14A1` adalah komposit 24 jam beresolusi 1 km dari band 750 m, bukan 375 m [VNP14A1-GEE]. Mask 1 km dipakai untuk 16 sel 375 m yang berada di dalamnya, sehingga ini adalah aproksimasi. Aturan komposit (mis. bagaimana overpass yang berawan dan yang cerah digabung) harus dibaca pada dokumentasi produk sebelum finalisasi. **[H]**
2. **Validasi mask:** bandingkan mask VNP14A1 dengan kelas awan pada swath VNP14IMG 375 m untuk subset (satu musim kebakaran), lalu laporkan akurasi kesepakatannya. **[H]**
3. VNP14A1 di GEE tidak NRT (ketersediaannya tertinggal), sehingga mask operasional 2026 harus berasal dari VNP14IMG NRT melalui Python [VNP14IMG-NRT] (Bagian 3.10).

### 1.5.2 Data hilang pada prediktor

| Jenis | Perlakuan | Dasar |
|---|---|---|
| Prediktor kontinu hilang | Isi nol **setelah standardisasi** (≈ rerata latih) | Mengikuti praktik TS-SatFire seperti dirangkum pada PRD v1; **[A]** |
| Piksel tak valid | Tambahkan mask validitas per grup sebagai kanal atau gunakan *masked loss* | [Huot2022] |
| NaN pada jarak (misalnya tidak ada jalan dalam ROI) | Isi dengan jarak maksimum (dibatasi) | **[H]** |
| Komposit vegetasi/reflektansi (opsional) | Sertakan flag "kosong", jangan mengisi nol tanpa flag | Alasan: nol fisik ≠ data hilang; **[H]** |

---

## 1.6 Pembentukan label

### 1.6.1 Definisi hari dan agregasi

Seluruh deteksi dalam satu jendela 24 jam digabung menjadi satu mask harian (WildfireSpreadTS menggunakan jendela 24 jam yang dimulai pada tengah malam) [Gerard2023]. Untuk Kalimantan, satu definisi hari dipakai seragam; **UTC disarankan** karena stempel waktu koleksi GEE berbasis UTC. Overpass S-NPP terjadi sekitar siang dan dini hari waktu lokal, sehingga definisi hari memengaruhi pasangan lintasan siang-malam yang tergabung. Dampaknya diuji sebagai sensitivitas. **[H]**

### 1.6.2 Penyaringan deteksi

| Filter | Aturan | Dasar |
|---|---|---|
| Kepercayaan | Buang `low`; simpan `nominal` dan `high` | [Gerard2023]; dokumentasi NASA menyatakan deteksi siang berkepercayaan rendah terkait sun glint dan anomali suhu kecil (15 K) [FIRMS-attr] |
| Sumber statis/non-vegetasi | Jika kolom `type` ada pada berkas VIIRS: simpan `type=0`. Jika tidak ada: buang piksel yang terdeteksi pada ≥ N hari dalam setahun pada lokasi yang sama (sumber panas persisten) | Atribut `type` didokumentasikan untuk MCD14ML [FIRMS-attr]; ketersediaan pada VIIRS harus dicek (Bagian 0.3). Ambang N: **[H]** |
| Satelit | S-NPP saja untuk dataset utama | Konsistensi sensor 2012–kini; NOAA-20 hanya untuk uji operasional [FIRMS-attr] |

### 1.6.3 Label dan state

Definisikan:

$$
A_t = \text{deteksi api VIIRS valid pada hari } t,\qquad
C_t=\bigcup_{\tau=t_0}^{t}A_\tau
$$

Target pertumbuhan hari berikut:

$$
Y_{t+24}=C_{t+24}\setminus C_t
$$

dengan mask validitas $M_{t+24}$:

$$
Y_{t+24}(x,y)=
\begin{cases}
1,&\text{api baru, teramati}\\
0,&\text{tidak ada api baru, teramati}\\
\text{(diabaikan)},&M_{t+24}(x,y)=0
\end{cases}
$$

Dua bentuk target disimpan: (i) **kumulatif** $C_{t+24}$, yang dipakai pada loss rujukan paper CA (loss paper dihitung pada peta api kumulatif) [Zhenirovskyy2026], dan (ii) **baru** $Y_{t+24}$, yang dipakai untuk metrik pertumbuhan. Persistence pada metrik kumulatif akan tinggi secara otomatis, sehingga **baseline persistence wajib** [Huot2022; Gerard2023]. NDWS melaporkan AUC-PR persistence 11,5% [Huot2022]; WildfireSpreadTS melaporkan AP persistence rerata 0,193 [Gerard2023].

Catatan: pada WildfireSpreadTS hanya sekitar 0,1% piksel yang berupa api [Gerard2023]. Ketidakseimbangan kelas ekstrem ini dipertimbangkan dalam loss (Bagian 2.13).

### 1.6.4 State awal CA

Peta deteksi dipetakan ke tiga probabilitas ($P_U$, $P_F$, $P_B$): sel terdeteksi pada hari $t$ → $P_F=1$; sel terdeteksi pada hari sebelumnya dalam event dan tidak pada $t$ → $P_B=1$; selainnya → $P_U=1$. **[H]**

**Masalah khusus gambut:** pada [Zhenirovskyy2026], $p_{continue}$ adalah skalar tetap [Zhenirovskyy2026]. Api gambut dapat membara berminggu-minggu sampai hujan musim penghujan datang [Field2016; Nikonovas2022]. Varian eksperimen: $p_{continue}$ sebagai fungsi DC/hujan/kelembapan tanah (Bagian 2.5). Model CA gambut yang memasukkan kelembapan gambut harian (KAPAS II) menunjukkan bahwa kelembapan memengaruhi perilaku kebakaran gambut [Purnomo2023-KAPAS2].

---

## 1.7 Quality control label

Pemeriksaan QC pada **event validasi dan uji** (bukan seluruh training):

| Pemeriksaan | Aturan | Dasar |
|---|---|---|
| Sumber panas persisten | Buang piksel/event yang didominasi lokasi dengan deteksi berulang sepanjang tahun | Deteksi non-vegetasi (industri, gas flare) perlu dibuang; atribut `type` bila tersedia [FIRMS-attr] |
| Kesesuaian dengan footprint terbakar akhir | Bandingkan hasil kumulatif dengan produk burned area (MCD64A1 pada GEE: `MODIS/061/MCD64A1`) | WildfireSpreadTS mencatat ketidakcocokan antara burned area MCD64A1 dan AF VNP14IMG (jenis produk, ambang, dan jarak waktu observasi) sehingga banyak event GlobFire tak memiliki deteksi AF [Gerard2023]. MCD64A1 dipakai **hanya** sebagai QA, bukan target harian. Catatan: file MCD64A1 bulanan, tetapi band `BurnDate` memuat tanggal terbakar per piksel (cek dokumentasi produk) |
| Kepadatan observasi | Hitung fraksi hari valid per event; event dengan fraksi valid rendah ditandai, **tidak dibuang otomatis** | Membuang event berdasarkan durasi atau kelengkapan dapat menimbulkan bias seleksi. WildfireSpreadTS tetap mempertahankan event dengan sedikit hari deteksi dan mengakuinya sebagai tantangan [Gerard2023] |
| Verifikasi visual | Sampel acak (≥ 30 event-hari) dibandingkan dengan citra optik/SAR yang tersedia | Validasi independen; Sentinel-1 menembus awan [Ban2020] ○ |

---

## 1.8 Identifikasi event kebakaran

NDWS mengambil sampel pada semua lokasi dan waktu dengan api aktif, memperlakukan api yang terpisah lebih dari 10 km sebagai api berbeda, dan hanya menyimpan sampel dengan setidaknya satu sel api pada mask sebelumnya [Huot2022]. WildfireSpreadTS menyimpan data per **event** dengan ROI tetap sepanjang event dan deret waktu harian [Gerard2023]. Karena Kalimantan tidak memiliki katalog perimeter harian setara NIFC atau GlobFire yang konsisten dengan label AF, event dibentuk dari deteksi AF itu sendiri. **[A]**

### Tahap 1: Kandidat deteksi

1. Unduh arsip FIRMS VIIRS S-NPP 375 m untuk kotak batas Kalimantan, 2012-01-20 hingga kini [FIRMS-archive].
2. Terapkan filter Bagian 1.6.2.
3. Simpan atribut `acq_date`, `acq_time`, `confidence`, `frp`, `scan`, `track`, `daynight`.

### Tahap 2: Operasi dilakukan di Python, bukan di GEE

Klasterisasi spatiotemporal dilakukan di Python (DBSCAN/graph connected components pada titik; `geopandas`, `scikit-learn`). GEE dipakai untuk mengekspor prediktor ke ROI yang sudah ditentukan. Alasan: arsip titik VIIRS historis tidak berada di GEE (Bagian 0.3), dan asosiasi spatiotemporal lebih mudah dikendalikan di Python. **[H]** (keputusan alat).

---

## 1.9 Klasterisasi spatiotemporal event

### Spasial (per hari)

Titik deteksi dalam satu hari dikelompokkan dengan jarak ambang. Nilai awal $d_{max}=10$ km, **mengikuti aturan NDWS untuk memisahkan api yang berbeda** [Huot2022]. **[R]** untuk nilai awal; sensitivitas $d_{max}\in\{3,5,10,20\}$ km diuji karena kebakaran Kalimantan cenderung tersebar (pembukaan lahan) [Cattau2016; Nikonovas2022], dan tidak ada rujukan tentang nilai optimal untuk wilayah ini.

### Temporal

Klaster hari $t$ dan $t+1$ digabung bila jaraknya ≤ $d_{max}$. Celah waktu maksimum $g_{max}$ (hari tanpa deteksi yang ditoleransi) diuji pada $\{1,2,3\}$ hari karena awan dan haze dapat menutup overpass [Gerard2023 menunjukkan banyak hari tanpa deteksi dalam event]. Tidak ada rujukan nilai optimal. **[H]**

### Diagnostik wajib sebelum nilai parameter dikunci

Hitung dan laporkan: distribusi ukuran event, durasi, jumlah event per tahun, fraksi event satu hari, fraksi hari tak valid saat overpass. Alasan: WildfireSpreadTS menunjukkan jumlah event sangat bervariasi antar tahun (74 event pada 2019 vs 201 pada 2020) dan kinerja persistence berubah mengikuti tahun [Gerard2023]. Pada tahun El Niño berat, nilai $d_{max}$ dan $g_{max}$ longgar dapat menggabung banyak kebakaran menjadi satu "event raksasa"; ini diperiksa pada diagnostik (penalaran penyusun). **[H]**

### Penyaringan event

**Tidak ada filter durasi minimum yang keras.** Sampel dibentuk per event-hari bila memenuhi: (i) ada deteksi valid pada hari $t$ dan (ii) ada setidaknya sebagian piksel valid pada hari target. Piksel tak valid diabaikan per piksel (Bagian 1.5). Alasannya: membuang event pendek menyingkirkan sebagian besar kebakaran lahan jangka pendek yang justru khas Kalimantan, dan WildfireSpreadTS sendiri mempertahankan event dengan sedikit hari deteksi [Gerard2023]. Statistik dilaporkan **dengan dan tanpa** event pendek. **[A]**

---

## 1.10 Penentuan ROI dan patch

1. **Mode dataset (retrospektif):** ROI fixed per event, berpusat pada centroid envelope event (union deteksi seluruh umur event). Ini setara dengan WildfireSpreadTS yang menentukan kotak dari pusat event retrospektif [Gerard2023]. **[R]**
2. **Mode operasional (prospektif):** ROI berpusat pada klaster aktif pada hari $t$, seperti NDWS yang berpusat pada api saat itu [Huot2022]. Evaluasi final dilaporkan juga pada mode ini, karena ROI retrospektif memakai informasi masa depan untuk penempatan. **[A]**
3. Syarat lama "api $t+72$ tidak boleh menyentuh batas patch" **dihapus**, karena memakai informasi masa depan untuk memilih sampel dan menimbulkan bias seleksi. Jika api melewati batas patch, dilakukan padding tepi dan sel di luar domain dikeluarkan dari metrik. Perlakuan batas mengikuti [Zhenirovskyy2026]: medan api di-zero-pad, medan lingkungan di-edge-pad. **[A]**
4. Event lebih besar dari satu patch dibagi ke beberapa tile ($256\times256$, overlap 25%). **Seluruh tile dari satu event berada pada split yang sama.** **[H]** (overlap); aturan satu-split adalah praktik pencegahan kebocoran [Roberts2017].

---

## 1.11 Organisasi temporal

Setiap event disimpan sebagai deret hari $D_1,\dots,D_n$. Satu hari berisi:

$$
X_t=[\,F_t,\;S,\;D_t,\;G_{t:t+24}\,]
$$

dengan $F_t$ riwayat api, $S$ statis/bertahun, $D_t$ dinamis teramati, $G$ forecast. State CA $S_t^{CA}$ dibentuk dari akumulasi deteksi (Bagian 1.6.4).

Prinsip pemrosesan: DeepLabV3+ memproses satu citra multikanal 2D per macro-step; dinamika temporal ditangani oleh CA. Ini sesuai [Zhenirovskyy2026] yang menjalankan jaringan parameter setiap macro-step. Eksperimen opsional: input $k$ hari (k = 1, 3, 5) sebagai kanal tambahan, mengikuti WildfireSpreadTS yang mendapati manfaat input multi-hari [Gerard2023].

**Aturan anti-bocor waktu (wajib, diuji dengan unit test):**

| Aturan | Dasar |
|---|---|
| Observasi dipakai jika timestamp ≤ $t$ | Prinsip dasar; [Huot2022] mengambil nilai terakhir yang tersedia pada $t$ untuk variabel lambat |
| Forecast dipakai jika `creation_time` ≤ $t$ dan `forecast_time` > $t$ | WildfireSpreadTS memakai forecast yang dibuat pada awal hari baru [Gerard2023]; latensi publikasi aktual harus diverifikasi pada metadata GFS |
| Komposit VNP13A1 dipakai jika **jendela komposit berakhir** ≤ $t$ | `system:time_start` pada GEE adalah awal jendela; periksa metadata |
| CEMS, ERA5-Land: gunakan nilai harian hingga $t-1$ atau sesuai latensi operasional | Latensi ±1 minggu untuk CEMS di GEE [Vitolo2020]; ±5 hari untuk ERA5-Land-T, sampai tiga bulan untuk versi final [ERA5Land-docs]. **Training dengan data final dan inference dengan data awal menimbulkan distribution shift**, sehingga varian "latensi realistis" dilatih/diuji (Bagian 3.10) |
| IMERG: gunakan kelas produk yang sama pada training dan inference (provisional atau final) | Koleksi GEE berisi produk provisional yang kemudian digantikan versi final [Huffman-IMERG] |

---

## 1.12 Stacking

Semua layer memiliki identitas grid yang sama: CRS, ukuran piksel 375 m, origin, affine transform, extent, tinggi dan lebar 256. Resampling kontinu: bilinear (mengikuti [Gerard2023]); kategorik (tutupan lahan, sawit): nearest neighbour; jarak: dihitung pada grid 375 m dari vektor/raster sumber.

Penanganan khusus:

| Variabel | Perlakuan | Dasar |
|---|---|---|
| Aspect | Pisahkan: sin/cos untuk jaringan; radian arah upslope untuk CA (tambahkan 180° dari arah menghadap, lalu konversi ke konvensi matematika) | [Zhenirovskyy2026] |
| Angin | u/v dinormalkan untuk jaringan; kecepatan $V=\sqrt{u^2+v^2}$ dan arah $\theta=\text{atan2}(v,u)$ untuk CA | [Zhenirovskyy2026] |
| Cuaca kasar (ERA5-Land, CEMS, GFS) | Raster bilinear 375 m **dan** ringkasan skalar per patch (rerata) disimpan | WildfireSpreadTS menampilkan perbedaan resolusi asli, termasuk forecast 27 km [Gerard2023]; ia dan [Eddin2022] menyarankan pemisahan perlakuan fitur statis dan dinamis. Varian FiLM/skalar diuji. **[H]** |
| Augmentasi | Flip dan rotasi (seperti NDWS) harus ikut mentransformasi kanal vektor (angin, aspect) | NDWS memakai flip, rotasi, dan crop acak [Huot2022]; transformasi vektor adalah koreksi logis **[H]** |

Hasil: $X_t\in\mathbb{R}^{32\times256\times256}$ (Core-32), batch $X\in\mathbb{R}^{B\times32\times256\times256}$.

---

## 1.13 Normalisasi

Statistik normalisasi dihitung **hanya dari data latih** (per fold). Prosedur mengikuti NDWS: nilai dipotong pada persentil 0,1 dan 99,9 (atau batas fisik), lalu dibakukan dengan rerata dan simpangan baku training yang dihitung setelah pemotongan [Huot2022]. **[R]**

- Hujan: transformasi $\log(1+x)$ sebelum pembakuan. **[H]**
- Angin: untuk kanal jaringan, bagi dengan $V_{max}$ seperti [Zhenirovskyy2026] (10 m/s) atau bakukan; CA menerima nilai fisik.
- Sudut tidak dibakukan sebagai derajat; gunakan sin/cos atau komponen vektor.
- Indeks kategorik tidak dibakukan; masuk embedding.

---

## 1.14 Format dataset

Satu berkas per event-hari. `.npz` dipertahankan untuk kesederhanaan; untuk skala ribuan sampel disarankan **Zarr per event** (kompresi dan akses acak). WildfireSpreadTS merekomendasikan konversi GeoTIFF ke HDF5 untuk training dan ukuran penuh dataset-nya sekitar 50 GB (607 event, 23 kanal) [Gerard2023] sebagai acuan ukuran.

```text
event_XXXX_YYYYMMDD.npz  (atau .zarr)

x                 float32 [32,256,256]    # Core-32
state_t           float32 [3,256,256]     # P_U, P_F, P_B
target_cum        uint8   [1,256,256]     # C_{t+24}
target_new        uint8   [1,256,256]     # Y_{t+24}
valid_next        uint8   [1,256,256]     # M_{t+24}
valid_x           uint8   [G,256,256]     # mask per grup fitur
ca_phys           float32 [5,256,256]     # V, theta_wind, slope[0..1], theta_upslope, M_fuel
scalars           float32 [3]             # ONI, DMI, sub-wilayah
event_id, date, feature_names, crs, transform, bbox
forecast_creation_time, product_versions (IMERG status, ERA5-Land tipe)
```

Metadata versi produk (misalnya status IMERG provisional/final dan tipe ERA5-Land) wajib disimpan agar distribution shift antara training dan inference dapat dilacak.

---

# 2. Modelling

## 2.1 Formulasi model

Arsitektur yang diusulkan: **DeepLabV3+ Parameter Generator + Probabilistic Cellular Automata**. [Zhenirovskyy2026] menggunakan MS-CNN (jalur paralel kernel 3×3, 5×5, 7×7) untuk menghasilkan enam peta parameter CA, dan CA tiga keadaan dengan agregasi ignisi Poisson. Penelitian ini mempertahankan CA dan mengganti jaringan parameter dengan DeepLabV3+ [Chen2018]. **[R]** (CA); **[H]** (manfaat penggantian jaringan).

## 2.2 Input

- Prediktor $X_t\in\mathbb{R}^{B\times32\times256\times256}$.
- State awal $S_t\in\mathbb{R}^{B\times3\times256\times256}$.
- Input fisik CA: $V$, $\theta_{wind}$, slope ($\text{derajat}/90$), $\theta_{aspect}$, $M_{fuel}$ (Bagian 1.12), seperti [Zhenirovskyy2026].
- Target $Y_{t+24}$ tidak masuk tensor prediktor.

## 2.3 Output DeepLabV3+

$$
\Theta_t=\text{DeepLabV3+}(X_t)\in\mathbb{R}^{B\times6\times256\times256}
$$

dengan enam parameter: $p_{base}$, $\alpha_{w1}$, $\alpha_{w2}$, $\alpha_s$, $\gamma$, $fuel_{factor}$ [Zhenirovskyy2026].

**Spesifikasi head (belum dirinci di paper):** setiap parameter dilewatkan aktivasi berbatas (misalnya sigmoid yang diskalakan atau softplus dengan batas atas) agar $p_{base}\in(0,1)$ dan koefisien tidak ekstrem. **[H]** Regularisasi kelancaran spasial (total variation) pada peta parameter diuji sebagai ablasi. Alasan: [Zhenirovskyy2026] sendiri menyatakan bahwa parameter yang dipelajari, terutama $fuel_{factor}$, berfungsi sebagai kuantitas efektif yang mencampurkan potensi bahan bakar dengan efek proses yang tak dimodelkan (ember spotting, suppression), sehingga interpretasi fisik perlu dijaga.

## 2.4 Alasan memakai DeepLabV3+

- DeepLabV3+ adalah arsitektur encoder-decoder dengan atrous spatial pyramid pooling untuk konteks multi-skala [Chen2018].
- DeepLabv3+ telah dipakai pada citra Sentinel-1/2 untuk memetakan sawit global dengan akurasi keseluruhan 98,52% [Descals2021], sehingga arsitekturnya pernah berhasil pada lanskap Asia Tenggara untuk tugas segmentasi.
- U-Net berbasis ResNet18 adalah baseline kuat pada WildfireSpreadTS (AP 0,341) dengan 14,4 juta parameter, sedangkan model dengan parameter jauh lebih sedikit (UTAE, 1,1 juta) bisa lebih baik pada kondisi tertentu [Gerard2023]. Jadi ukuran model bukan jaminan, dan kapasitas harus dikontrol (Bagian 3.8).

**Pemeriksaan teknis yang wajib:** pada input 256×256 dengan output stride 16 peta fitur ASPP hanya 16×16. Laju atrous bawaan (6, 12, 18) dirancang untuk masukan yang lebih besar [Chen2018], sehingga laju yang melebihi ukuran peta menjadi tidak efektif. Gunakan output stride 8 atau kurangi laju atrous, lalu verifikasi dengan uji kecil. Backbone pralatih ImageNet berasumsi tiga kanal, sehingga lapisan konvolusi pertama dilatih ulang atau backbone dilatih dari awal. Pada jumlah event yang terbatas, mulai dari backbone kecil (ResNet18 atau MobileNet). **[H]**

Novelty yang diklaim bukan "menggunakan DeepLabV3+", tetapi **menggunakan DeepLabV3+ untuk mempelajari parameter CA yang bervariasi spasial pada rezim kebakaran tropis**, lalu menguji apakah ini mengungguli generator yang lebih sederhana.

## 2.5 State dan transisi CA

Tiga probabilitas, dengan invarian

$$
P_U+P_F+P_B=1,\qquad P_{fire}=P_F+P_B
$$

[Zhenirovskyy2026].

**Potensi propagasi** dari sel sumber $u$ ke tetangga Moore $x$:

$$
\phi(u\rightarrow x)=p_{base}(x)\cdot fuel_{factor}(u)\cdot\kappa_{wind}(u)\cdot\kappa_{slope}(u)
$$

$$
\kappa_{wind}(u)=\exp\!\big(\alpha_{w1}V\big)\exp\!\big(\alpha_{w2}V[\cos(\theta_{spread}-\theta_{wind})-1]\big),\quad
\kappa_{slope}(u)=\exp\!\big(\alpha_s\,\text{slope}\cos(\theta_{spread}-\theta_{aspect})\big)
$$

$$
\lambda(x)=\sum_{u\in N(x)}\phi(u\rightarrow x),\qquad
P_{ignite}(x)=1-\exp[-\gamma(x)\lambda(x)]
$$

Pembaruan per micro-step [Zhenirovskyy2026]:

$$
\text{newly\_burning}=p_U\cdot P_{ignite},\qquad \text{burned\_now}=p_F(1-p_{continue})
$$

$$
p_U^{t+1}=p_U^t-\text{newly\_burning},\quad
p_F^{t+1}=p_F^t+\text{newly\_burning}-\text{burned\_now},\quad
p_B^{t+1}=p_B^t+\text{burned\_now}
$$

**Catatan verifikasi pada kode referensi** (`mzhen77/neural-ca-wildfire`): paper menulis $\phi$ dan $\lambda$ tanpa memuat probabilitas $p_{burning}(u)$ pada persamaan yang tertulis, padahal agregasi Poisson mensyaratkan sumber yang membara. Periksa pada kode apakah $\lambda$ dibobot dengan $p_F(u)$, lalu tuliskan versi yang diimplementasikan. Paper juga tidak merinci aktivasi tiap parameter. **[H]**

**Varian $p_{continue}$ untuk gambut (eksperimen):**
- M0: skalar tetap, seperti [Zhenirovskyy2026].
- M1: $p_{continue}(x)=\sigma(a_0+a_1\,\widehat{DC}+a_2\,\widehat{\text{rain}})$, atau peta yang dikeluarkan DeepLabV3+ sebagai kanal ketujuh. Dasar: api gambut dapat bertahan hingga hujan musim penghujan [Field2016], dan model CA gambut mengaitkan perilaku kebakaran dengan kelembapan gambut [Purnomo2023-KAPAS2; Purnomo2021-KAPAS]. **[H]**

## 2.6 Representasi fuel

Tutupan lahan **tidak dijadikan mask burnable/non-burnable keras** selain kelas yang jelas tidak terbakar (air, bangunan padat). Paper rujukan menunjukkan bahwa sebagian besar area terbakar dapat berada di luar peta bahan bakar yang dipetakan (mis. 65,3% pada Brattain 2020), sehingga mask keras menyebabkan prediksi kurang [Zhenirovskyy2026]. Untuk Kalimantan, indeks tutupan lahan (dan indeks sawit) diembedding untuk menghasilkan $fuel_{factor}$, seperti embedding FBFM40 203×4 pada paper. $M_{fuel}$ hanya menutup air dan permukiman. **[R]** untuk prinsip; **[H]** untuk transfer ke tropis, yang oleh paper tidak didemonstrasikan.

## 2.7 Angin dan terrain pada CA

DeepLabV3+ menerima angin dan terrain ternormalisasi, sedangkan CA menerima besaran fisik: $V$ (dibatasi 0–50 m/s), $\theta_{wind}$, slope/90, dan $\theta_{aspect}$ [Zhenirovskyy2026]. Pada gambut datar $\kappa_{slope}\approx1$; ablasi CA tanpa suku slope akan menunjukkan apakah suku ini bermanfaat di Kalimantan.

## 2.8 Macro-step dan micro-step

**Macro-step: 24 jam**, konsisten dengan resolusi harian label dan dengan [Zhenirovskyy2026] yang membagi satu macro-step menjadi beberapa micro-step.

**Micro-step.** Paper melaporkan 50 micro-step menghasilkan kinerja terbaik dibanding 15 yang dipakai pendahulunya [Zhenirovskyy2026]. Nilai ini **tidak ditransfer otomatis**. Pada tetangga Moore, front maju paling banyak satu sel per micro-step, sehingga **batas atas jarak tempuh per macro-step adalah $n\times 375$ m** (untuk $n=50$: ±18,75 km pada arah sumbu). Model CA umum dilaporkan menghasilkan laju penyebaran yang jauh lebih cepat daripada laju sebenarnya bila waktu dan jarak tidak dikalibrasi [Purnomo2023-KAPAS2]. Prosedur:

1. Dari data event (Bagian 1.9) hitung distribusi jarak pertumbuhan harian front (misalnya persentil 50, 90, 99 jarak sel baru ke $C_t$).
2. Uji $n\in\{5,10,15,30,50\}$ pada validasi.
3. Pilih $n$ yang sesuai dengan distribusi empiris dan metrik. **[A]**

## 2.9 Horizon prediksi

Tugas utama: $t\rightarrow t+24$ jam. Rollout autoregresif:

$$
t\rightarrow t+24\rightarrow t+48\rightarrow t+72
$$

State prediksi $S^{pred}_{t+24}$ menjadi awal macro-step berikutnya. Forcing forecast diganti ($G_{0-24}\rightarrow G_{24-48}\rightarrow G_{48-72}$). **Tidak ada state api teramati dari masa depan yang dimasukkan kembali** pada evaluasi. Pelatihan memakai unroll hingga 3 macro-step dan loss dirata-rata antar hari seperti [Zhenirovskyy2026]; ini menghindari ketidaksesuaian antara pelatihan satu langkah dan rollout. Memori untuk backpropagation melalui banyak micro-step dikelola dengan gradient checkpointing. **[A]**

Perlu dicatat bahwa paper rujukan memakai *data assimilation* 10 hari pertama untuk menyesuaikan jaringan pada tiap event sebelum prakiraan 10 hari berikutnya, dan menyebut hasilnya sebagai proyeksi bersyarat, bukan simulasi bebas [Zhenirovskyy2026]. Karena itu penelitian ini menyajikan dua mode (Bagian 3.6): **zero-shot** (tanpa penyesuaian per event) dan **teradaptasi** (fine-tuning pada $k$ hari terakhir event uji).

## 2.10 Keterbatasan struktural CA yang diakui

CA hanya menyebarkan api dari sel yang sedang membara dan tidak dapat menghasilkan ignisi independen; paper melaporkan kegagalan ini pada kebakaran Ferguson 2018 (recall awal < 0,1) [Zhenirovskyy2026]. Kebakaran Kalimantan banyak berasal dari pembukaan lahan antropogenik [Cattau2016; Nikonovas2022], sehingga:

- **Evaluasi utama dipisah:** (i) *dapat dijangkau CA* — sel baru dalam jarak Chebyshev ≤ $n$ sel dari $C_t$ (jangkauan geometri model), dan (ii) *tak terjangkau* — ignisi baru atau lompatan.
- **Varian M3 (opsional):** tambahkan kepala "hazard ignisi latar" $P_{bg}(x)=\sigma(h(X))$ yang menyumbang sel baru di luar kontagion. Jalur fitur: peat, jarak ke akses manusia, DC, riwayat api multi-tahun. **[H]**

## 2.11 Keluaran model akhir

1. $P_U, P_F, P_B$.
2. $P_{fire}=P_F+P_B$ (kumulatif).
3. $P_{ignite}$ (potensi ignisi sel yang belum terbakar).
4. $\Delta P_{fire}$ = pertumbuhan terhadap jejak api saat ini, yang menjadi produk operasional utama.

## 2.12 Fungsi loss

Konfigurasi rujukan (nilai paper):

$$
\mathcal{L}=0.5\,\mathcal{L}_{\text{frontierBCE}}+0.1\,\mathcal{L}_{\text{MSE}}+0.5\,\mathcal{L}_{\text{area}}+4.0\,\mathcal{L}_{\text{IoU}}+0.1\,\mathcal{L}_{\text{pool}}
$$

dengan bobot frontier $w_f=10$ dan dilasi 3×3 untuk menentukan frontier [Zhenirovskyy2026]. Bobot ini adalah **konfigurasi rujukan, bukan optimum yang diklaim**.

**Modifikasi untuk dataset ini:**

1. **Masked loss.** Semua suku dikalikan mask validitas $M_{t+24}$ dan dinormalkan oleh jumlah sel valid; sel tak teramati tidak berkontribusi. Alasan: NDWS mengabaikan label tak pasti pada loss dan semua metrik [Huot2022]. **[R]**
2. **Frontier BCE pada label berbintik.** Operasi dilasi 3×3 dirancang pada perimeter yang padat dan kontigu [Zhenirovskyy2026]; deteksi AF berbintik, sehingga frontier perlu dihitung pada masker yang dihaluskan (closing morfologis kecil) atau pada klaster. Diuji pada validasi. **[H]**
3. **Ketidakseimbangan kelas.** Pada WildfireSpreadTS ±0,1% piksel adalah api [Gerard2023]. Pembobotan kelas atau Dice/Jaccard loss dibandingkan; WildfireSpreadTS memakai Dice untuk U-Net, Jaccard untuk ConvLSTM, dan cross-entropy berbobot untuk UTAE [Gerard2023]. **[A]**
4. **Definisi target.** Loss rujukan dihitung pada peta kumulatif [Zhenirovskyy2026]; loss tambahan pada $\Delta P_{fire}$ terhadap $Y_{t+24}$ diuji. Karena $C_t$ sudah mendominasi metrik kumulatif, metrik pada target baru adalah metrik utama untuk klaim kemampuan prediksi.
5. Penyetelan terbatas pada $w_{BCE}, w_{area}, w_{IoU}$ dilakukan pada validasi.

---

# 3. Validation

## 3.1 Pembagian data

Pembagian dilakukan berdasarkan **tahun dan blok spasial**, bukan piksel atau patch acak. Dasar:

- WildfireSpreadTS memakai validasi silang menurut tahun (dua tahun latih, satu validasi, satu uji; 12 kombinasi) karena distribusi data sangat berbeda antar tahun, dan melaporkan rerata ± simpangan baku antar fold [Gerard2023]. **[R]**
- ProbFire memakai *leave-one-year-out* agar mencerminkan kemampuan memprediksi musim api mendatang [Nikonovas2022]. **[R]**
- NDWS membagi acak per minggu 8:1:1 dengan jeda satu hari [Huot2022]. Skema ini lebih lemah terhadap autokorelasi spasial-temporal, dan tidak dipakai sebagai skema utama. Validasi silang yang mengabaikan struktur temporal/spasial cenderung menggelembungkan kinerja [Roberts2017].

**Protokol:**

1. **Hitung jumlah event per tahun dan kondisi ENSO/IOD sebelum membagi.** Tahun api Indonesia sangat bergantung pada kekeringan [Field2016; Pan2018], sehingga tahun uji harus mencakup setidaknya satu tahun kering dan satu tahun non-kering. ProbFire menunjukkan reliabilitas prediksi berbeda antar tahun (dua tahun api tinggi kurang andal) [Nikonovas2022].
2. **Skema utama:** leave-one-year-out (LOYO), dengan satu tahun terpisah sebagai validasi untuk tuning.
3. **Skema sekunder:** blok spasial (mis. kotak 1°) di dalam tahun uji untuk menguji transfer antar wilayah (sub-wilayah dengan rezim berbeda: [Nikonovas2022]).
4. **Tidak ada tile dari event yang sama pada split berbeda.**
5. Tahun 2026 menjadi **studi kasus operasional** (Bagian 3.10), bukan bagian dari penyetelan.
6. Karena $S$-NPP mengalami drift orbit (perubahan jam overpass) pada tahun-tahun belakangan — penalaran penyusun, perlu diverifikasi — lakukan pengecekan waktu overpass tahunan pada data. **[H]**

## 3.2 Pencegahan kebocoran

Aturan Bagian 1.11 diterapkan dan diuji otomatis:

- Tidak ada piksel prediktor dengan timestamp > $t$.
- Tidak ada forecast dengan `creation_time` > $t$.
- Komposit VNP13A1 (jika dipakai) berakhir ≤ $t$.
- Normalisasi memakai statistik latih saja.
- Semua tile satu event satu split.
- Ukuran ROI dan klaster dari mode retrospektif tidak boleh dipakai pada evaluasi operasional (Bagian 1.10).

## 3.3 Perbandingan model

| Kode | Model | Tujuan | Dasar |
|---|---|---|---|
| B0 | **Persistence** | Batas bawah wajib | [Huot2022; Gerard2023] |
| B1 | Dilasi morfologis (opsional dengan bias arah angin) | Baseline fisik sederhana | **[H]** |
| B2 | Regresi logistik jendela 3×3 | Baseline statistik | [Huot2022; Gerard2023] |
| B3 | Gradient boosting/random forest per piksel (jendela 3×3) | Baseline tabular | Random forest dipakai pada [Huot2022]; boosting **[H]** |
| B4 | U-Net (ResNet18) segmentasi langsung | Baseline segmentasi | [Gerard2023; Ronneberger2015] |
| B5 | DeepLabV3+ segmentasi langsung | Pemisahan efek CA | Rancangan PRD v1 |
| B6 | CA dengan parameter konstan (global) | Efek parameterisasi spasial | [Zhenirovskyy2026] (kontras dengan CA global) |
| B7 | MS-CNN + CA | Replikasi prinsip paper rujukan | [Zhenirovskyy2026] |
| **P** | **DeepLabV3+ + CA** | Model usulan | — |
| P+ | P + $p_{continue}$ adaptif (M1) dan/atau hazard latar (M3) | Ekstensi tropis | **[H]** |

Dua pertanyaan diuji terpisah: pengaruh CA (B5 vs P) dan pengaruh generator parameter (B7 vs P). Agar adil, B5 dan B7 menerima **input yang sama** (termasuk $S_t$ dan riwayat api), dan **kapasitas dikontrol** (jumlah parameter dan receptive field disetarakan; Bagian 3.8).

**Reproduksi sebagai langkah awal.** Data dan kode paper CA tersedia secara publik (dataset Xia & Cheng di Mendeley; kode paper di GitHub) [Xia2024; Zhenirovskyy2026]. Sebelum menyentuh Kalimantan, replikasi hasil paper pada data AS memakai implementasi PyTorch Anda untuk memvalidasi kode CA. **[A]**

## 3.4 Metrik prediksi

Akurasi tidak dipakai karena kelas api sangat jarang. Metrik utama:

| Metrik | Alasan |
|---|---|
| **PR-AUC / Average Precision** | Metrik utama NDWS (AUC-PR) dan WildfireSpreadTS (AP); AP dinilai lebih sesuai daripada AUROC untuk data tak seimbang [Huot2022; Gerard2023] |
| Precision, Recall, F1/Dice | Memisahkan overprediksi dan underprediksi, seperti IoU/Precision/Recall pada [Zhenirovskyy2026] |
| IoU (Jaccard) | Kontinuitas dengan paper rujukan; dilaporkan untuk peta kumulatif dan peta **baru** secara terpisah |

Seluruh metrik dihitung **hanya pada sel valid** ($M=1$). Dilaporkan pada: (i) peta kumulatif dan (ii) sel baru, dan terpisah untuk strata "dapat dijangkau CA" dan "tak terjangkau" (Bagian 2.10).

## 3.5 Metrik spasial dan arah

| Metrik | Fungsi | Dasar |
|---|---|---|
| Error relatif luas | $\lvert A_{pred}-A_{obs}\rvert/A_{obs}$ | Dipakai sebagai suku area loss [Zhenirovskyy2026] |
| **Fractions Skill Score (FSS)** pada beberapa radius | Toleransi terhadap ketidakpastian ±1 piksel dari geolokasi dan footprint VIIRS | [Roberts2008] ○ |
| Boundary-F1 dengan buffer, HD95 | Mengganti Hausdorff mentah yang sensitif pencilan | **[H]** |
| Pergeseran centroid, sudut dan laju front | **Metrik arah dan laju spread** yang dibutuhkan oleh pertanyaan riset | **[H]** |

Alasan memakai toleransi spasial: Manhattan distance pada [Zhenirovskyy2026] dan label berbintik membuat metrik piksel-ke-piksel sangat keras terhadap pergeseran kecil.

## 3.6 Validasi horizon dan mode evaluasi

- Metrik dilaporkan terpisah untuk **+24, +48, +72 jam**.
- +48 dan +72 berasal dari **simulasi autoregresif bebas** tanpa memasukkan state teramati.
- **Dua mode:** (a) *zero-shot* — parameter dibekukan setelah pelatihan; (b) *teradaptasi* — fine-tuning singkat pada $k$ hari terakhir event uji (analog dengan jendela asimilasi 10 hari pada [Zhenirovskyy2026], yang tidak menyertakan satu pun event sebagai hold-out). Hanya mode (a) yang dapat diklaim sebagai generalisasi ke event baru; mode (b) dibandingkan dengan paper rujukan.
- Angka IoU > 0,6 pada paper rujukan **bukan target** untuk mode zero-shot.

## 3.7 Ablasi grup fitur

Grup: F (riwayat api), S (statis), D (dinamis), G (forecast), dan sub-grup (peat, tutupan lahan, perkebunan, forest loss, akses manusia, topografi, hujan, indeks kekeringan, angin/RH/tanah, forecast).

$$
Importance_g=Metric_{all}-Metric_{without\ g}
$$

Desain eksperimen mengikuti WildfireSpreadTS: (i) satu grup + fire mask, (ii) penambahan kumulatif grup terurut, dan (iii) semua fitur — karena penambahan fitur dapat tidak menaikkan kinerja [Gerard2023]. Fire mask sebelumnya adalah fitur dominan [Huot2022], sehingga ablasi juga dilaporkan **tanpa** fire mask untuk melihat sumbangan fitur lain.

Pertanyaan khusus:

1. Forecast vs tanpa forecast: pada WildfireSpreadTS forecast tidak membantu U-Net [Gerard2023]; di sini diuji apakah CA yang memakai angin dan hujan forecast secara fisik membuat forecast lebih informatif.
2. Indeks kekeringan (DC/FFMC) vs hujan IMERG saja.
3. Akses manusia (jalan, kanal, permukiman): diuji pada strata dapat dijangkau dan tak terjangkau.
4. Tier D (VNP09GA, NDVI/EVI).

## 3.8 Ablasi CA dan kontrol kapasitas

- Micro-step $n\in\{5,10,15,30,50\}$.
- CA tanpa suku angin; tanpa suku slope; $fuel_{factor}$ konstan vs dipelajari.
- $p_{continue}$: tetap (M0) vs adaptif (M1).
- Regularisasi kelancaran parameter: ada vs tidak.
- **Kontrol kapasitas:** MS-CNN dengan jumlah parameter dan receptive field disetarakan dengan DeepLabV3+ ringan, agar peningkatan tidak sekadar berasal dari jaringan lebih besar. Catatan dari WildfireSpreadTS: model dengan parameter jauh lebih sedikit dapat melampaui model besar [Gerard2023].
- **Sanity check fisik:** tanda $\alpha_s$, peta $p_{base}$ dan $fuel_{factor}$ terhadap gambut/tutupan lahan, dan kestabilan antar seed.

## 3.9 Kalibrasi probabilitas

Output CA bersifat *mean-field* (bukan simulasi Monte Carlo), sehingga probabilitasnya tidak otomatis terkalibrasi. Evaluasi:

- Skor Brier dan diagram reliabilitas, seperti pada ProbFire [Nikonovas2022; Murphy1973; Gneiting2007].
- Kalibrasi pasca-hoc (isotonik atau temperature scaling), karena jaringan modern sering tidak terkalibrasi [Guo2017]. **[A]**
- Sharpness (histogram probabilitas), sebab prakiraan yang andal tetapi tidak tajam kurang berguna bagi pengambil keputusan [Nikonovas2022].
- Opsional: analisis nilai relatif cost–loss untuk kegunaan operasional [Nikonovas2022].

Brier sensitif terhadap prevalensi kelas dan hanya dapat dibandingkan pada sampel yang sama [Nikonovas2022].

## 3.10 Validasi operasional 2026

1. **Api saat ini:** `NASA/LANCE/SNPP_VIIRS/C2` di GEE [LANCE-GEE]. Karena NRT tidak berkualitas ilmiah, bandingkan secara kuantitatif dengan arsip standard processing untuk periode tumpang tindih (≥ 2023-09-03) sebelum inference (*shift check*).
2. **Mask validitas:** VNP14IMG NRT via Python [VNP14IMG-NRT], sebab VNP14A1 di GEE tidak NRT.
3. **Cuaca:** hujan IMERG (kelas provisional, sesuai yang dilatih), CEMS dan ERA5-Land dengan latensi realistis, forecast GFS 0–24, 24–48, 48–72 jam dengan `creation_time ≤ t`.
4. **Statis dan tahunan:** versi terakhir yang tersedia sebelum $t$.
5. **Model dilatih ulang dengan varian "latensi realistis"** (fitur teramati ditunda sesuai latensi nyata) agar tidak ada ketidakcocokan antara training dan inference.
6. Opsional: NOAA-20 (`NASA/LANCE/NOAA20_VIIRS/C2`) sebagai korroborasi independen [LANCE-GEE-N20].

## 3.11 Kebaruan penelitian

Kontribusi yang dapat dipertahankan, dengan batas klaim:

1. **Dataset custom spread Kalimantan dengan label VIIRS 375 m dan mask tak teramati**, dengan prediktor yang dipilih dari literatur kebakaran Indonesia. Tidak diklaim sebagai adaptasi langsung dari TS-SatFire.
2. **Uji transfer neural-CA ke rezim tropis**, hal yang oleh [Zhenirovskyy2026] disebut belum didemonstrasikan.
3. **Uji apakah DeepLabV3+ menghasilkan peta parameter CA yang lebih baik** dibanding MS-CNN dengan kapasitas setara.
4. **Pemisahan evaluasi** antara pertumbuhan dari api yang ada dan ignisi baru, yang relevan untuk kebakaran antropogenik di Kalimantan.

Pertanyaan riset: apakah probabilistic CA berparameter DeepLabV3+ meningkatkan akurasi geometri, arah, dan laju progresi kebakaran 24–72 jam di Kalimantan dibanding segmentasi langsung dan CA berparameter MS-CNN, pada protokol zero-shot per tahun?

---

# Lampiran A. Alur kerja GEE + Python dan data eksternal

| Langkah | Alat | Keterangan |
|---|---|---|
| 1. Unduh arsip VIIRS AF S-NPP 2012–kini | Python (`requests`, FIRMS) | Tidak ada di GEE untuk 2012–2023 [LANCE-GEE; Gerard2023] |
| 2. Filter, rasterisasi 375 m | `geopandas`, `rasterio`, `numpy` | Footprint memakai `scan`, `track` |
| 3. Klaster event | `scikit-learn` (DBSCAN) / `networkx` | Bagian 1.9 |
| 4. Diagnostik event dan penentuan split | `pandas` | Bagian 3.1 |
| 5. Ekspor prediktor per ROI | **GEE Python API** (`ee`, `geemap`, `xee`) | Gunakan ekspor batch/`computePixels`, bukan ribuan task; GEE memudahkan seleksi dan resampling [Gorelick2017]. Kode ekspor GEE NDWS dan loader WildfireSpreadTS (MIT) dapat dipakai sebagai titik awal [Huot2022; Gerard2023] |
| 6. Mask validitas | **GEE** (`VNP14A1`); validasi swath: Python | Bagian 1.5 |
| 7. Fitur jarak | Python (`scipy.ndimage.distance_transform_edt`) pada raster jalan/kanal/permukiman | OSM via `osmnx`/Overpass; GRIP |
| 8. Indeks ONI/DMI | Python (`pandas.read_csv` dari NOAA PSL) | Skalar |
| 9. Penyimpanan | Zarr/HDF5 | Bagian 1.14 |
| 10. Model | PyTorch; `segmentation_models_pytorch` (dipakai WildfireSpreadTS) | [Gerard2023] |

**Data eksternal di luar GEE (semuanya terbuka):** arsip FIRMS; OpenStreetMap; peta gambut Indonesia [Haryono2011] (opsional); tabel ONI/DMI.

**Ukuran:** dataset WildfireSpreadTS (607 event, 23 kanal, 375 m) berukuran sekitar 50 GB [Gerard2023]; estimasi Anda perlu dihitung dari jumlah event-hari hasil diagnostik.

---

# Lampiran B. Risiko dan hal yang harus diverifikasi sebelum pelaksanaan

| # | Risiko / verifikasi | Dampak | Cara memeriksa |
|---|---|---|---|
| 1 | Kolom `type` pada arsip VIIRS 375 m | Penyaringan sumber non-vegetasi | Buka berkas arsip; jika tidak ada, pakai mask hotspot persisten |
| 2 | Aturan komposit `FireMask` di VNP14A1 dan kecocokan 1 km vs 375 m | Validitas mask | Baca dokumentasi LP DAAC; bandingkan dengan kelas awan VNP14IMG |
| 3 | Latensi GFS arsip dan `creation_time` pada GEE | Kebocoran/ketidaksesuaian inference | Inspeksi metadata koleksi |
| 4 | ID koleksi `VNP09GA`, `NASA/VIIRS/002/VNP13A1`, rilis Hansen, `WorldPop/GP/100m/pop` | Skrip gagal | Cek di katalog GEE |
| 5 | Kelengkapan kanal di OSM Kalimantan | Kualitas fitur jarak | Bandingkan dengan peta kanal KHG/peta lokal |
| 6 | Jumlah event per tahun dan per fase ENSO/IOD | Desain split | Diagnostik Bagian 1.9 |
| 7 | Drift orbit S-NPP dan perubahan jam overpass | Konsistensi label antar tahun | Plot distribusi `acq_time` per tahun |
| 8 | Perbedaan produk IMERG provisional vs final; ERA5-Land-T vs final | Distribution shift | Simpan versi produk di metadata; uji dengan latensi realistis |
| 9 | Kekeringan dan gambut: tidak ada produk kedalaman muka air gambut di GEE | Keterbatasan fisik | Dinyatakan sebagai batasan; proksi lewat hujan antecedent dan kelembapan tanah |
| 10 | Kebakaran sub-piksel dan membara rendah: omisi deteksi | Label berisik | MODIS melaporkan omisi kebakaran kecil lebih tinggi dan deteksi gambut membara lebih sulit, tetapi probabilitas deteksi naik pada tiap overpass [Nikonovas2022; Giglio2016]; dampak pada VIIRS diukur lewat validasi |
| 11 | Repositori paper CA: pembobotan $\lambda$ oleh $p_F(u)$, aktivasi parameter | Kebenaran implementasi | Baca kode `mzhen77/neural-ca-wildfire` |
| 12 | Perubahan kebijakan dan penindakan setelah 2015 menggeser sensitivitas api | Pergeseran distribusi antar tahun | ProbFire mencatat kemungkinan ini sebagai alasan prediksi berlebih pada 2019 [Nikonovas2022]; evaluasi per tahun |

---

# Lampiran C. Register keputusan inti

| Keputusan | Status | Rujukan |
|---|---|---|
| Target VIIRS 375 m S-NPP, gabungan 24 jam, buang low-confidence | R | [Gerard2023; Schroeder2014] |
| Kelas tak teramati dan masked loss | R | [Huot2022] |
| Mask dari `FireMask` VNP14A1 (1 km) untuk sel 375 m | H | [VNP14A1-GEE] |
| Riwayat api sebagai prediktor | R | [Huot2022; Gerard2023; Nikonovas2022] |
| Gambut, sawit/perkebunan, forest loss | R | [Nikonovas2022; Field2016; Nikonovas2020] |
| Jarak ke jalan, kanal, permukiman | R/A | [Horton2022; Cattau2016; Huot2022] |
| Hujan IMERG dan akumulasi bertingkat | R | [Field2016; Nikonovas2022; Field2015-GFWED] |
| DC dan FFMC dari CEMS | R | [GFWED-SVS; Shawki2017; Vitolo2020; LAPAN2011] |
| Fitur dikelompokkan dan diuji bertahap | R | [Gerard2023] |
| Split LOYO + blok spasial | R/A | [Gerard2023; Nikonovas2022; Roberts2017] |
| $d_{max}=10$ km sebagai nilai awal | R | [Huot2022] |
| $g_{max}$, $n$ micro-step, lebar jendela hujan | H | — |
| DeepLabV3+ menggantikan MS-CNN | H | [Chen2018; Zhenirovskyy2026] |
| Evaluasi dipisah dapat-dijangkau vs tak-terjangkau | H | [Zhenirovskyy2026] |
| $p_{continue}$ adaptif untuk gambut | H | [Field2016; Purnomo2023-KAPAS2] |

---

# Sitasi dataset (untuk bagian Data pada naskah)

Gunakan format sitasi yang diminta penyedia data. Untuk produk LANCE/FIRMS, ikuti panduan LANCE Citation, Acknowledgements, and Disclaimer [LANCE-GEE].

| Dataset | Sitasi yang disarankan |
|---|---|
| VIIRS 375 m AF (S-NPP), arsip dan NRT | Schroeder, W., Oliva, P., Giglio, L., Csiszar, I. A. (2014). The new VIIRS 375 m active fire detection data product. *Remote Sensing of Environment*, 143, 85–96. doi:10.1016/j.rse.2013.12.008 ✔. NRT: NRT VIIRS 375 m Active Fire product VNP14IMGT distributed from NASA FIRMS. doi:10.5067/FIRMS/VIIRS/VNP14IMGT_NRT.002 ✔ |
| VNP14A1 (mask) | NASA LP DAAC, VIIRS/NPP Thermal Anomalies/Fire Daily L3 Global 1 km SIN Grid, versi 002; ikuti halaman "Citing Our Data" LP DAAC [VNP14A1-GEE] ✔ |
| IMERG V07 | Huffman, G.J., Stocker, E.F., Bolvin, D.T., Nelkin, E.J., Tan, J. (2019), GPM IMERG L3 Half Hourly 0.1° × 0.1° V07, GES DISC. doi:10.5067/GPM/IMERG/3B-HH/07 ✔ |
| CEMS Fire Danger Indices | Vitolo, C. et al. (2020). ERA5-based global meteorological wildfire danger maps. *Scientific Data*, 7. ; Copernicus Climate Change Service (2019), Fire danger indices historical data from the Copernicus Emergency Management Service, CDS. doi:10.24381/cds.0e89c522 ✔. Acknowledge: "Contains modified Copernicus Climate Change Service information" ✔ |
| ERA5-Land | Muñoz-Sabater, J. (2019). ERA5-Land hourly data from 1950 to present. C3S CDS. doi:10.24381/cds.e2161bac ○; Muñoz-Sabater et al. (2021). *Earth System Science Data*, 13, 4349–4383 ○ |
| GFS | NOAA/NCEP Global Forecast System 0.25°, melalui GEE `NOAA/GFS0P25` (cek halaman katalog untuk sitasi) |
| Global Peatland Map 2.0 | Greifswald Mire Centre (2022). Global Peatland Map 2.0. UNEP, Nairobi; dikurasi di GEE oleh Ka Hei dan S. Roy ✔ |
| Peat-ML | Melton, J.R. et al. (2022). A map of global peatland extent created using machine learning (Peat-ML). Zenodo. doi:10.5281/zenodo.7352284 ✔ |
| Oil palm | Descals, A. et al. (2021). High-resolution global map of smallholder and industrial closed-canopy oil palm plantations. *ESSD*, 13, 1211–1231. doi:10.5194/essd-13-1211-2021 ✔; Zenodo doi:10.5281/zenodo.4473715 ✔; Descals et al. (2024). *ESSD*, 16, 5111–5129. doi:10.5194/essd-16-5111-2024 ✔ |
| Hansen GFC | Hansen, M.C. et al. (2013). High-resolution global maps of 21st-century forest cover change. *Science*, 342, 850–853. doi:10.1126/science.1244693 ✔ |
| MCD12Q1 | Friedl, M., Sulla-Menashe, D. (2022). MODIS/Terra+Aqua Land Cover Type Yearly L3 Global 500m SIN Grid V061. NASA LP DAAC. doi:10.5067/MODIS/MCD12Q1.061 ✔ |
| SRTM | Farr, T.G. et al. (2007). The Shuttle Radar Topography Mission. *Reviews of Geophysics*, 45, RG2004. doi:10.1029/2005RG000183 ✔; NASA JPL (2013), SRTMGL1 v003. doi:10.5067/MEaSUREs/SRTM/SRTMGL1.003 ✔ |
| GRIP | Meijer, J.R. et al. (2018). Global patterns of current and future road infrastructure. *Environmental Research Letters*, 13, 064006 ○ |
| OpenStreetMap | © OpenStreetMap contributors (ODbL). Boeing, G. (2017). OSMnx. *Computers, Environment and Urban Systems*, 65, 126–139 ○ |
| WorldPop | Tatem, A.J. (2017). WorldPop, open data for spatial demography. *Scientific Data*, 4, 170004 ○ |
| ONI / DMI | NOAA Physical Sciences Laboratory / CPC (tabel ONI); Saji, N.H. et al. (1999). *Nature*, 401, 360–363 ○ |
| Peta gambut Indonesia | Haryono, S.M. et al. (2011). Peatland Map of Indonesia. Balai Besar Litbang Sumberdaya Lahan Pertanian, Kementerian Pertanian ✔ (via [Nikonovas2022]) |
| Google Earth Engine | Gorelick, N. et al. (2017). Google Earth Engine. *Remote Sensing of Environment*, 202, 18–27. doi:10.1016/j.rse.2017.06.031 ✔ |

---

# Referensi

**✔** = halaman/artikel diakses saat penyusunan; **○** = dari pengetahuan penyusun, **cek DOI dan rincian sebelum submit**.

- **[Ban2020]** ○ Ban, Y. et al. (2020). Near real-time wildfire progression monitoring with Sentinel-1 SAR time series and deep learning. *Scientific Reports*, 10, 1322. doi:10.1038/s41598-019-56967-x (dirujuk pada daftar pustaka [Gerard2023]).
- **[Boeing2017]** ○ Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*, 65, 126–139.
- **[Cattau2016]** ✔ Cattau, M.E., Harrison, M.E., Shinyo, I., Tungau, S., Uriarte, M., DeFries, R. (2016). Sources of anthropogenic fire ignitions on the peat-swamp landscape in Kalimantan, Indonesia. *Global Environmental Change*, 39, 205–219. doi:10.1016/j.gloenvcha.2016.05.005 (via daftar pustaka [Nikonovas2022]).
- **[Chen2018]** ○ Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *ECCV 2018*.
- **[Descals2021]** ✔ Descals, A., Wich, S., Meijaard, E., Gaveau, D.L.A., Peedell, S., Szantoi, Z. (2021). *ESSD*, 13, 1211–1231. doi:10.5194/essd-13-1211-2021.
- **[Descals2024]** ✔ Descals, A., Gaveau, D.L.A., Wich, S., Szantoi, Z., Meijaard, E. (2024). *ESSD*, 16, 5111–5129. doi:10.5194/essd-16-5111-2024 (ringkasan pada halaman terkait artikel 2021).
- **[Eddin2022]** ✔ Eddin, M.H.S., Roscher, R., Gall, J. (2022). Location-aware adaptive denormalization: a deep learning approach for wildfire danger forecasting. arXiv:2212.08208 (via [Gerard2023]).
- **[ERA5Land-docs]** ✔ ECMWF. ERA5-Land data documentation. https://confluence.ecmwf.int/spaces/CKB/pages/140385202/ERA5-Land+data+documentation ; katalog GEE `ECMWF/ERA5_LAND/DAILY_AGGR`.
- **[Farr2007]** ✔ Farr, T.G. et al. (2007). *Reviews of Geophysics*, 45, RG2004. doi:10.1029/2005RG000183.
- **[Field2015-GFWED]** ✔ Field, R.D. et al. (2015). Development of a Global Fire Weather Database. *NHESS*, 15, 1407–1423. doi:10.5194/nhess-15-1407-2015.
- **[Field2016]** ✔ Field, R.D. et al. (2016). Indonesian fire activity and smoke pollution in 2015 show persistent nonlinear sensitivity to El Niño-induced drought. *PNAS*, 113, 9204–9209. doi:10.1073/pnas.1524888113.
- **[FIRMS-archive]** ✔ NASA FIRMS. Archive download; arsip VIIRS 375 m tersedia mulai 20 Januari 2012 (lihat dokumentasi FIRMS). https://firms.modaps.eosdis.nasa.gov/download/
- **[FIRMS-attr]** ✔ NASA Earthdata. Active fire data attributes (MODIS and VIIRS). https://www.earthdata.nasa.gov/data/tools/firms/active-fire-data-attributes-modis-viirs
- **[ForGeo2025]** ✔ Linking weather variability and wildfires across tropical ecoregions: evidence from Sumatra, Indonesia. *Forum Geografi*. doi:10.23917/forgeo.15980 (judul dan DOI dari halaman jurnal).
- **[Friedl2022]** ✔ Friedl, M., Sulla-Menashe, D. (2022). MCD12Q1 V061. doi:10.5067/MODIS/MCD12Q1.061.
- **[Funk2015]** ○ Funk, C. et al. (2015). The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. *Scientific Data*, 2, 150066.
- **[Gaveau2014]** ✔ Gaveau, D.L.A. et al. (2014). Major atmospheric emissions from peat fires in Southeast Asia during non-drought years. *Scientific Reports*, 4, 6112. doi:10.1038/srep06112.
- **[Gerard2023]** ✔ Gerard, S., Zhao, Y., Sullivan, J. (2023). WildfireSpreadTS: a dataset of multi-modal time series for wildfire spread prediction. *NeurIPS 2023 Datasets and Benchmarks*. Data: doi:10.5281/zenodo.8006177. Kode: https://github.com/SebastianGer/WildfireSpreadTS
- **[Giglio2016]** ✔ Giglio, L., Schroeder, W., Justice, C.O. (2016). The collection 6 MODIS active fire detection algorithm and fire products. *RSE*, 178, 31–41. doi:10.1016/j.rse.2016.02.054.
- **[Gneiting2007]** ✔ Gneiting, T., Raftery, A.E. (2007). Strictly proper scoring rules, prediction, and estimation. *JASA*, 102, 359–378. doi:10.1198/016214506000001437.
- **[Gorelick2017]** ✔ Gorelick, N. et al. (2017). *RSE*, 202, 18–27. doi:10.1016/j.rse.2017.06.031.
- **[GFWED-SVS]** ✔ NASA SVS. Global Fire Weather Database (visualisasi). https://svs.gsfc.nasa.gov/4634/
- **[GPM2.0]** ✔ Global Peatland Database, katalog komunitas GEE. https://gee-community-catalog.org/projects/peatland/
- **[Guo2017]** ✔ Guo, C., Pleiss, G., Sun, Y., Weinberger, K.Q. (2017). On calibration of modern neural networks. *PMLR*, 70, 1321–1330.
- **[Hansen2013]** ✔ Hansen, M.C. et al. (2013). *Science*, 342, 850–853. doi:10.1126/science.1244693.
- **[Haryono2011]** ✔ Haryono, S.M. et al. (2011). Peatland Map of Indonesia (data set), Kementerian Pertanian RI; tersedia di Global Forest Watch.
- **[Horton2022]** ✔ Horton, A.J., Lehtinen, J., Kummu, M. (2022). Targeted land management strategies could halve peatland fire occurrences in Central Kalimantan, Indonesia. *Communications Earth & Environment*, 3, 204. doi:10.1038/s43247-022-00534-2.
- **[Huffman-IMERG]** ✔ GPM IMERG V07, `NASA/GPM_L3/IMERG_V07`. doi:10.5067/GPM/IMERG/3B-HH/07.
- **[Huot2022]** ✔ Huot, F., Hu, R.L., Goyal, N., Sankar, T., Ihme, M., Chen, Y.-F. (2022). Next Day Wildfire Spread: a machine learning dataset to predict wildfire spreading from remote-sensing data. *IEEE TGRS*, 60. doi:10.1109/TGRS.2022.3192974. arXiv:2112.02447.
- **[LANCE-GEE]** ✔ VNP14IMGTDL_NRT Daily Raster, `NASA/LANCE/SNPP_VIIRS/C2`, katalog GEE. https://developers.google.com/earth-engine/datasets/catalog/NASA_LANCE_SNPP_VIIRS_C2
- **[LANCE-GEE-N20]** ✔ VJ114IMGTDL_NRT, `NASA/LANCE/NOAA20_VIIRS/C2`, katalog GEE.
- **[LAADS-spec]** ✔ LAADS DAAC. VNP14IMG V2.5.4 file specification (kelas fire mask). https://ladsweb.modaps.eosdis.nasa.gov/filespec/VIIRS/1/VNP14IMG
- **[LAPAN2011]** ✔ Roswintiarti, O. (LAPAN). Presentasi Indonesia pada pertemuan GOFC-GOLD Fire, 2011 (hotspot vs FFMC; sebaran kebakaran di lahan gambut). Materi presentasi, bukan artikel sejawat; bobot bukti rendah.
- **[Melton2022]** ✔ Melton, J.R. et al. (2022). Peat-ML. Zenodo doi:10.5281/zenodo.7352284.
- **[Meijer2018]** ○ Meijer, J.R. et al. (2018). Global patterns of current and future road infrastructure. *ERL*, 13, 064006.
- **[Murphy1973]** ✔ Murphy, A.H. (1973). A new vector partition of the probability score. *J. Appl. Meteorol.*, 12, 595–600.
- **[MunozSabater2019]** ○ Muñoz-Sabater, J. (2019). ERA5-Land hourly data from 1950 to present. C3S CDS. doi:10.24381/cds.e2161bac.
- **[Nikonovas2020]** ✔ Nikonovas, T. et al. (2020). Near-complete loss of fire-resistant primary tropical forest cover in Sumatra and Kalimantan. *Communications Earth & Environment*, 1, 65. doi:10.1038/s43247-020-00069-4.
- **[Nikonovas2022]** ✔ Nikonovas, T., Spessa, A., Doerr, S.H., Clay, G.D., Mezbahuddin, S. (2022). ProbFire: a probabilistic fire early warning system for Indonesia. *NHESS*, 22, 303–322. doi:10.5194/nhess-22-303-2022.
- **[Oliva2015]** ✔ Oliva, P., Schroeder, W. (2015). Assessment of VIIRS 375 m active fire detection product for direct burned area mapping. *RSE*, 160, 144–155.
- **[Page2002]** ✔ Page, S.E. et al. (2002). The amount of carbon released from peat and forest fires in Indonesia during 1997. *Nature*, 420, 61–65. doi:10.1038/nature01131.
- **[Pan2018]** ✔ Pan, X., Chin, M., Ichoku, C.M., Field, R.D. (2018). Connecting Indonesian fires and drought with the type of El Niño and phase of the Indian Ocean dipole during 1979–2016. *JGR Atmospheres*, 123, 7974–7988. doi:10.1029/2018JD028402.
- **[Purnomo2021-KAPAS]** ✔ Purnomo, D.M.J., Bonner, M., Moafi, S., Rein, G. (2021). Using cellular automata to simulate field-scale flaming and smouldering wildfires in tropical peatlands. *Proceedings of the Combustion Institute* (PII S1540748920306490).
- **[Purnomo2023-KAPAS2]** ✔ KAPAS II: simulation of peatland wildfires with daily variations of peat moisture content. *International Journal of Wildland Fire*, 32(6), 823. doi:10.1071/WF22109 (dari URL penerbit).
- **[Quaest2023]** ✔ The analysis of fire hotspot distribution in Kalimantan and its relationship with ENSO phases. *Quaestiones Geographicae*, 42(1), 2023. doi:10.14746/quageo-2023-0006.
- **[Roberts2008]** ○ Roberts, N.M., Lean, H.W. (2008). Scale-selective verification of rainfall accumulations from high-resolution forecasts of convective events. *Monthly Weather Review*, 136, 78–97.
- **[Roberts2017]** ○ Roberts, D.R. et al. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40, 913–929.
- **[Ronneberger2015]** ✔ Ronneberger, O., Fischer, P., Brox, T. (2015). U-Net. *MICCAI* (via [Gerard2023]).
- **[Saji1999]** ○ Saji, N.H., Goswami, B.N., Vinayachandran, P.N., Yamagata, T. (1999). A dipole mode in the tropical Indian Ocean. *Nature*, 401, 360–363.
- **[Schroeder2014]** ✔ Schroeder, W., Oliva, P., Giglio, L., Csiszar, I.A. (2014). *RSE*, 143, 85–96. doi:10.1016/j.rse.2013.12.008.
- **[Schroeder2025-UG]** ✔ Schroeder, W. (2025). Collection 2 VIIRS 375-m Active Fire Product User's Guide, v1.2. NASA Earthdata.
- **[Shawki2017]** ✔ Shawki, D. et al. (2017). Long-lead prediction of the 2015 fire and haze episode in Indonesia. *GRL*, 44, 9996–10005. doi:10.1002/2017GL073660.
- **[Spessa2015]** ✔ Spessa, A.C. et al. (2015). Seasonal forecasting of fire over Kalimantan, Indonesia. *NHESS*, 15, 429–442. doi:10.5194/nhess-15-429-2015.
- **[Tatem2017]** ○ Tatem, A.J. (2017). WorldPop, open data for spatial demography. *Scientific Data*, 4, 170004.
- **[Vetrita2020]** ✔ Vetrita, Y., Cochrane, M.A. (2020). Fire frequency and related land-use and land-cover changes in Indonesia's peatlands. *Remote Sensing*, 12, 5. doi:10.3390/rs12010005.
- **[Vitolo2020]** ✔ Vitolo, C. et al. (2020). ERA5-based global meteorological wildfire danger maps. *Scientific Data*, 7. Katalog GEE komunitas: https://gee-community-catalog.org/projects/cems_fire/
- **[VNP14A1-GEE]** ✔ VNP14A1 V002, `NASA/VIIRS/002/VNP14A1`. https://developers.google.com/earth-engine/datasets/catalog/NASA_VIIRS_002_VNP14A1
- **[VNP14IMG-NRT]** ✔ NASA LANCE. VNP14IMG_NRT product description. https://modaps.modaps.eosdis.nasa.gov/services/about/products/viirs-c2-nrt/VNP14IMG_NRT.html
- **[Xia2024]** ✔ Xia, Z., Cheng, S. (2024). Data for PyTorchFire. Mendeley Data V1. doi:10.17632/nx2wsksp9k.1. dan Xia & Cheng (2025). *Environmental Modelling & Software*, 188, 106401. doi:10.1016/j.envsoft.2025.106401.
- **[Zanaga2022]** ○ Zanaga, D. et al. (2022). ESA WorldCover 10 m 2021 v200. Zenodo. doi:10.5281/zenodo.7254221.
- **[Zhao-TSSatFire]** ○ Zhao, Y. et al. TS-SatFire (dirujuk pada PRD v1 dan pada arXiv:2503.08580). **Lengkapi rincian sitasi dari paper aslinya**; tidak diverifikasi pada sesi ini.
- **[Zhenirovskyy2026]** ✔ Zhenirovskyy, M., Matei, I., Vuppala, R., Kurihana, T., Wong, H.Y. (2026). Neural-parameterized cellular automata for wildfire spread. *Ecological Informatics*, 97, 103928. doi:10.1016/j.ecoinf.2026.103928. Kode: https://github.com/mzhen77/neural-ca-wildfire
