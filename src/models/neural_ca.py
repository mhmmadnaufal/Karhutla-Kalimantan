"""
Modul Arsitektur DeepLabV3+ Parameter Generator, MS-CNN Baseline (Capacity-Matched),
dan 3-State Probabilistic Cellular Automata (CA) dengan Varian Gambut Adaptif (M1).
Sesuai PRD v2 Bagian 2.1 - 2.11 dan [Zhenirovskyy2026].
"""

from typing import Dict, List, Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ModelCAConfig


class BasicBlock2D(nn.Module):
    """Residual BasicBlock standar ResNet18 dalam PyTorch murni (tanpa ketergantungan torchvision)."""

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


def make_resnet_layer(in_planes: int, planes: int, num_blocks: int = 2, stride: int = 1) -> nn.Sequential:
    layers = [BasicBlock2D(in_planes, planes, stride=stride)]
    for _ in range(1, num_blocks):
        layers.append(BasicBlock2D(planes, planes, stride=1))
    return nn.Sequential(*layers)


class CategoricalLandcoverEmbedding(nn.Module):
    """
    Mengubah indeks kategorik tutupan lahan (IGBP 0..17) menjadi vektor embedding kontinu
    seperti embedding FBFM40 pada [Zhenirovskyy2026] (PRD Bagian 1.4.1 & 2.6).
    """

    def __init__(self, num_classes: int = 18, embed_dim: int = 4):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=num_classes, embedding_dim=embed_dim)

    def forward(self, lc_idx: torch.Tensor) -> torch.Tensor:
        # lc_idx: [B, H, W] -> output: [B, embed_dim, H, W]
        emb = self.embedding(lc_idx.clamp(0, self.embedding.num_embeddings - 1))
        return emb.permute(0, 3, 1, 2).contiguous()


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int):
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, padding=dilation,
                dilation=dilation, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=min(16, out_channels), num_channels=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling yang disesuaikan untuk patch 256x256 (PRD Bagian 2.4).
    Pada output stride 8 (peta fitur 32x32) atau stride 16 (peta fitur 16x16),
    laju atrous dikalibrasi menjadi (2, 4, 8) agar tidak melampaui lebar peta fitur.
    """

    def __init__(self, in_channels: int, atrous_rates: Tuple[int, int, int] = (2, 4, 8), out_channels: int = 128):
        super().__init__()
        modules = [
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = [conv(x) for conv in self.convs]
        res = torch.cat(res, dim=1)
        return self.project(res)


class BoundedCAParameterHead(nn.Module):
    """
    Kepala aktivasi berbatas untuk ke-6 parameter spasial CA (PRD Bagian 2.3):
      Theta_t = [p_base, alpha_w1, alpha_w2, alpha_s, gamma, fuel_factor]
    Menjaga stabilitas gradien dan makna fisik:
      - p_base      in (0.001, 0.95)  via Sigmoid
      - alpha_w1    in (0.0, 0.25)    via Softplus terpotong / Sigmoid
      - alpha_w2    in (0.0, 0.40)    via Sigmoid
      - alpha_s     in (0.0, 2.00)    via Sigmoid
      - gamma       in (0.05, 4.00)   via Sigmoid
      - fuel_factor in (0.05, 3.00)   via Sigmoid
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.conv_out = nn.Conv2d(in_channels, 6, kernel_size=1)
        # Inisialisasi bias agar nilai awal parameter berada di rezim fisik moderat
        nn.init.normal_(self.conv_out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.conv_out.bias, 0.0)

    def forward(self, feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw = self.conv_out(feat)  # [B, 6, H, W]
        p_base = 0.005 + 0.945 * torch.sigmoid(raw[:, 0:1])
        alpha_w1 = 0.25 * torch.sigmoid(raw[:, 1:2])
        alpha_w2 = 0.40 * torch.sigmoid(raw[:, 2:3])
        alpha_s = 2.00 * torch.sigmoid(raw[:, 3:4])
        gamma = 0.20 + 3.80 * torch.sigmoid(raw[:, 4:5])
        fuel_factor = 0.20 + 2.80 * torch.sigmoid(raw[:, 5:6])

        theta = torch.cat([p_base, alpha_w1, alpha_w2, alpha_s, gamma, fuel_factor], dim=1)
        return {
            "theta": theta,
            "p_base": p_base,
            "alpha_w1": alpha_w1,
            "alpha_w2": alpha_w2,
            "alpha_s": alpha_s,
            "gamma": gamma,
            "fuel_factor": fuel_factor,
        }


class DeepLabV3PlusParameterGenerator(nn.Module):
    """
    Generator Parameter CA berbasis DeepLabV3+ dengan backbone ResNet18 ringan,
    embedding tutupan lahan, dan ASPP terkalibrasi untuk patch 256x256 (PRD Bagian 2.3 & 2.4).
    """

    def __init__(
        self,
        in_channels: int = 17,
        num_lc_classes: int = 18,
        lc_embed_dim: int = 4,
        aspp_out: int = 128,
    ):
        super().__init__()
        self.lc_embed = CategoricalLandcoverEmbedding(num_lc_classes, lc_embed_dim)
        # Kanal efektif: in_channels - 1 (karena landcover_idx diganti embedding) + lc_embed_dim
        eff_in_channels = (in_channels - 1) + lc_embed_dim

        self.stem = nn.Sequential(
            nn.Conv2d(eff_in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.layer1 = make_resnet_layer(64, 64, num_blocks=2, stride=1)    # [B, 64, 64, 64] (stride 4)
        self.layer2 = make_resnet_layer(64, 128, num_blocks=2, stride=2)   # [B, 128, 32, 32] (stride 8)
        self.layer3 = make_resnet_layer(128, 256, num_blocks=2, stride=2)  # [B, 256, 16, 16] (stride 16)

        # ASPP pada peta fitur 16x16 dengan atrous_rates=(2, 4, 6) sesuai PRD 2.4
        self.aspp = ASPP(in_channels=256, atrous_rates=(2, 4, 6), out_channels=aspp_out)

        # Proyeksi fitur low-level (stride 4, 64x64)
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Decoder DeepLabV3+
        self.decoder = nn.Sequential(
            nn.Conv2d(aspp_out + 32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.param_head = BoundedCAParameterHead(64)

    def forward(
        self,
        x: torch.Tensor,
        landcover_idx: torch.Tensor,
        lc_channel_pos: int = 7,
    ) -> Dict[str, torch.Tensor]:
        # Ganti kanal landcover_idx dengan vektor embeddingnya
        lc_emb = self.lc_embed(landcover_idx)  # [B, E, H, W]
        x_without_lc = torch.cat([x[:, :lc_channel_pos], x[:, lc_channel_pos + 1:]], dim=1)
        x_in = torch.cat([x_without_lc, lc_emb], dim=1)

        input_size = x.shape[-2:]
        s = self.stem(x_in)
        low_feat = self.layer1(s)       # 64 x 64
        x2 = self.layer2(low_feat)      # 32 x 32
        high_feat = self.layer3(x2)     # 16 x 16

        aspp_feat = self.aspp(high_feat)
        aspp_up = F.interpolate(aspp_feat, size=low_feat.shape[-2:], mode="bilinear", align_corners=False)
        low_proj = self.low_level_proj(low_feat)

        dec = self.decoder(torch.cat([aspp_up, low_proj], dim=1))
        dec_full = F.interpolate(dec, size=input_size, mode="bilinear", align_corners=False)
        return self.param_head(dec_full)


class MSCNNParameterGenerator(nn.Module):
    """
    Baseline B7: Multi-Scale CNN Parameter Generator (jalur paralel kernel 3x3, 5x5, 7x7)
    sesuai paper rujukan [Zhenirovskyy2026] untuk uji komparasi adil terhadap DeepLabV3+ (PRD 3.3 & 3.8).
    """

    def __init__(
        self,
        in_channels: int = 17,
        num_lc_classes: int = 18,
        lc_embed_dim: int = 4,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.lc_embed = CategoricalLandcoverEmbedding(num_lc_classes, lc_embed_dim)
        eff_in = (in_channels - 1) + lc_embed_dim

        self.branch3 = nn.Sequential(
            nn.Conv2d(eff_in, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(eff_in, hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.branch7 = nn.Sequential(
            nn.Conv2d(eff_in, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.param_head = BoundedCAParameterHead(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        landcover_idx: torch.Tensor,
        lc_channel_pos: int = 7,
    ) -> Dict[str, torch.Tensor]:
        lc_emb = self.lc_embed(landcover_idx)
        x_without_lc = torch.cat([x[:, :lc_channel_pos], x[:, lc_channel_pos + 1:]], dim=1)
        x_in = torch.cat([x_without_lc, lc_emb], dim=1)
        b3 = self.branch3(x_in)
        b5 = self.branch5(x_in)
        b7 = self.branch7(x_in)
        fused = self.fusion(torch.cat([b3, b5, b7], dim=1))
        return self.param_head(fused)


class ProbabilisticCellularAutomata(nn.Module):
    """
    Mekanisme Fisik 3-State Probabilistic Cellular Automata (Unburned, Burning, Burned)
    dengan agregasi ignisi Poisson pada ketetanggaan Moore 3x3 (8 arah) sesuai PRD Bagian 2.5 - 2.8.
    Dilengkapi:
      - Pembobotan eksplisit potensi propagasi oleh p_F(u) (probabilitas sumber sedang membara)
        dan koreksi jarak diagonal (1 / sqrt(2)).
      - Varian M0 (p_continue skalar tetap) dan M1 (p_continue adaptif gambut sebagai fungsi
        fraksi gambut, Drought Code, dan curah hujan).
      - Ablasi suku angin dan suku slope.
    """

    def __init__(
        self,
        cfg: ModelCAConfig = ModelCAConfig(),
        enable_wind: bool = True,
        enable_slope: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        self.enable_wind = enable_wind
        self.enable_slope = enable_slope

        # Definisi 8 tetangga Moore (dr, dc) dari sumber u ke target x:
        # Jika sel sumber u berada di (r + dr, c + dc), maka vektor rambatan dari u ke x adalah (-dr, -dc).
        # Dalam koordinat kartesian (dx = -dc, dy = +dr karena indeks baris meningkat ke selatan/bawah):
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
        self.neighbor_offsets = neighbor_offsets
        spread_angles = []
        dist_weights = []
        for dr, dc in neighbor_offsets:
            dx = float(-dc)
            dy = float(dr)  # jika u di atas (dr=-1), api merambat ke bawah (selatan, dy=-1)
            theta_spread = math.atan2(dy, dx)
            dist = math.sqrt(dx * dx + dy * dy)
            spread_angles.append(theta_spread)
            dist_weights.append(1.0 / dist)  # 1.0 untuk sumbu kardinal, 1/sqrt(2) untuk diagonal

        self.register_buffer("spread_angles", torch.tensor(spread_angles, dtype=torch.float32).view(1, 8, 1, 1))
        self.register_buffer("dist_weights", torch.tensor(dist_weights, dtype=torch.float32).view(1, 8, 1, 1))

        # Parameter yang dapat dipelajari untuk Varian M1: p_continue adaptif gambut (PRD Bagian 2.5)
        # p_continue(x) = sigmoid(a_0 + a_peat * peat + a_dc * DC_norm - a_rain * rain_norm)
        self.a0_logit = nn.Parameter(torch.tensor(math.log(cfg.p_continue_base / (1.0 - cfg.p_continue_base))))
        self.a_peat = nn.Parameter(torch.tensor(1.20))
        self.a_dc = nn.Parameter(torch.tensor(0.85))
        self.a_rain = nn.Parameter(torch.tensor(1.50))

    def compute_p_continue(self, peat_hydro: Optional[torch.Tensor], H: int, W: int, device: torch.device) -> torch.Tensor:
        """
        Menghitung probabilitas api terus membara (p_continue) per micro-step:
        - Mode M0: konstan di seluruh grid.
        - Mode M1: meningkat pada lahan gambut dengan Drought Code tinggi dan menurun saat hujan.
        """
        if not self.cfg.use_adaptive_p_continue or peat_hydro is None:
            return torch.sigmoid(self.a0_logit).view(1, 1, 1, 1).expand(1, 1, H, W).to(device)

        peat = peat_hydro[:, 0:1]
        dc_norm = peat_hydro[:, 1:2]
        rain_norm = peat_hydro[:, 2:3]
        logit = self.a0_logit + self.a_peat * peat + self.a_dc * dc_norm - self.a_rain * rain_norm
        return torch.sigmoid(logit)

    @staticmethod
    def _extract_moore_neighbors(tensor_4d: torch.Tensor, pad_mode: str = "replicate") -> torch.Tensor:
        """
        Mengekstrak nilai pada ke-8 tetangga Moore u di sekitar setiap sel x:
        Input : [B, 1, H, W]
        Output: [B, 8, H, W] sesuai urutan neighbor_offsets.
        Perlakuan batas mengikuti PRD Bagian 1.10: medan api di-zero-pad, medan lingkungan di-edge-pad.
        """
        if pad_mode == "zeros":
            padded = F.pad(tensor_4d, (1, 1, 1, 1), mode="constant", value=0.0)
        else:
            padded = F.pad(tensor_4d, (1, 1, 1, 1), mode="replicate")

        _, _, H, W = tensor_4d.shape
        neighbors = [
            padded[:, :, 0:H, 0:W],      # (-1, -1) top-left
            padded[:, :, 0:H, 1:W+1],    # (-1,  0) top
            padded[:, :, 0:H, 2:W+2],    # (-1, +1) top-right
            padded[:, :, 1:H+1, 0:W],    # ( 0, -1) left
            padded[:, :, 1:H+1, 2:W+2],  # ( 0, +1) right
            padded[:, :, 2:H+2, 0:W],    # (+1, -1) bottom-left
            padded[:, :, 2:H+2, 1:W+1],  # (+1,  0) bottom
            padded[:, :, 2:H+2, 2:W+2],  # (+1, +1) bottom-right
        ]
        return torch.cat(neighbors, dim=1)  # [B, 8, H, W]

    def forward(
        self,
        state_t: torch.Tensor,
        param_dict: Dict[str, torch.Tensor],
        ca_phys: torch.Tensor,
        peat_hydro: Optional[torch.Tensor] = None,
        n_micro_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Menjalankan simulasi 1 macro-step (24 jam) yang terdiri atas `n_micro_steps` langkah CA.
        Input:
          - state_t    : [B, 3, H, W] -> (P_U, P_F, P_B) dengan P_U + P_F + P_B = 1
          - param_dict : keluaran generator parameter (p_base, alpha_w1, alpha_w2, alpha_s, gamma, fuel_factor)
          - ca_phys    : [B, 5, H, W] -> (V, theta_wind, slope_norm, theta_upslope, M_fuel)
          - peat_hydro : [B, 3, H, W] -> (peat, dc_norm, rain_norm) untuk varian M1
        """
        steps = n_micro_steps if n_micro_steps is not None else self.cfg.n_micro_steps
        B, _, H, W = state_t.shape

        p_U = state_t[:, 0:1].clone()
        p_F = state_t[:, 1:2].clone()
        p_B = state_t[:, 2:3].clone()
        initial_fire = (p_F + p_B).detach()

        p_base = param_dict["p_base"]
        alpha_w1 = param_dict["alpha_w1"]
        alpha_w2 = param_dict["alpha_w2"]
        alpha_s = param_dict["alpha_s"]
        gamma = param_dict["gamma"]
        fuel_factor = param_dict["fuel_factor"]

        V = ca_phys[:, 0:1].clamp(0.0, self.cfg.max_wind_speed_mps)
        theta_wind = ca_phys[:, 1:2]
        slope_norm = ca_phys[:, 2:3].clamp(0.0, 1.0)
        theta_upslope = ca_phys[:, 3:4]
        M_fuel = ca_phys[:, 4:5].clamp(0.0, 1.0)

        # Pra-hitung suku arah dari setiap tetangga sumber u menuju sel x (PRD Bagian 2.5):
        # Semua kuantitas dievaluasi pada sel sumber u (edge-padded), dikali p_base(x) & M_fuel(x)
        fuel_u = self._extract_moore_neighbors(fuel_factor, pad_mode="replicate")  # [B, 8, H, W]
        V_u = self._extract_moore_neighbors(V, pad_mode="replicate")
        th_wind_u = self._extract_moore_neighbors(theta_wind, pad_mode="replicate")
        aw1_u = self._extract_moore_neighbors(alpha_w1, pad_mode="replicate")
        aw2_u = self._extract_moore_neighbors(alpha_w2, pad_mode="replicate")

        if self.enable_wind:
            # kappa_wind(u -> x) = exp(alpha_w1 * V) * exp(alpha_w2 * V * [cos(theta_spread - theta_wind) - 1])
            cos_wind = torch.cos(self.spread_angles - th_wind_u)
            wind_exponent = (aw1_u * V_u + aw2_u * V_u * (cos_wind - 1.0)).clamp(-10.0, 6.0)
            kappa_wind = torch.exp(wind_exponent)
        else:
            kappa_wind = torch.ones_like(fuel_u)

        if self.enable_slope:
            slope_u = self._extract_moore_neighbors(slope_norm, pad_mode="replicate")
            th_up_u = self._extract_moore_neighbors(theta_upslope, pad_mode="replicate")
            as_u = self._extract_moore_neighbors(alpha_s, pad_mode="replicate")
            cos_slope = torch.cos(self.spread_angles - th_up_u)
            slope_exponent = (as_u * slope_u * cos_slope).clamp(-6.0, 6.0)
            kappa_slope = torch.exp(slope_exponent)
        else:
            kappa_slope = torch.ones_like(fuel_u)

        # Kernel transmisi statis dari u ke x (sebelum dikalikan p_F(u) yang berubah tiap micro-step)
        # phi_static(u -> x) = p_base(x) * M_fuel(x) * fuel_factor(u) * kappa_wind(u->x) * kappa_slope(u->x) * dist_weight
        phi_static = (
            p_base * M_fuel * fuel_u * kappa_wind * kappa_slope * self.dist_weights
        )  # [B, 8, H, W]

        p_continue = self.compute_p_continue(peat_hydro, H, W, state_t.device)
        last_p_ignite = torch.zeros_like(p_U)

        for _ in range(steps):
            # Medan api p_F di-zero-pad pada batas patch sesuai PRD Bagian 1.10
            p_F_neighbors = self._extract_moore_neighbors(p_F, pad_mode="zeros")  # [B, 8, H, W]
            lam = torch.sum(phi_static * p_F_neighbors, dim=1, keepdim=True)     # [B, 1, H, W]
            p_ignite = 1.0 - torch.exp(-gamma * lam.clamp(min=0.0))

            newly_burning = p_U * p_ignite
            burned_now = p_F * (1.0 - p_continue)

            p_U = (p_U - newly_burning).clamp(0.0, 1.0)
            p_F = (p_F + newly_burning - burned_now).clamp(0.0, 1.0)
            p_B = (p_B + burned_now).clamp(0.0, 1.0)

            # Jaga invarian P_U + P_F + P_B = 1 (PRD Bagian 2.5)
            norm_sum = (p_U + p_F + p_B).clamp(min=1e-6)
            p_U = p_U / norm_sum
            p_F = p_F / norm_sum
            p_B = p_B / norm_sum
            last_p_ignite = p_ignite

        state_next = torch.cat([p_U, p_F, p_B], dim=1)
        p_fire_cum = (p_F + p_B).clamp(0.0, 1.0)
        delta_p_fire = (p_fire_cum - initial_fire).clamp(0.0, 1.0)

        return {
            "state_next": state_next,
            "p_U": p_U,
            "p_F": p_F,
            "p_B": p_B,
            "p_fire_cum": p_fire_cum,
            "delta_p_fire": delta_p_fire,
            "p_ignite": last_p_ignite,
            "p_continue_map": p_continue,
        }


class NeuralCAWildfireModel(nn.Module):
    """
    Model Terpadu: Parameter Generator (DeepLabV3+ atau MS-CNN) + Probabilistic CA.
    Mendukung prediksi satu langkah (t -> t+24) maupun rollout autoregresif bebas
    hingga t+48 dan t+72 jam (PRD Bagian 2.9).
    """

    def __init__(
        self,
        cfg: ModelCAConfig = ModelCAConfig(),
        generator_type: str = "deeplabv3plus",
        in_channels: int = 17,
        enable_wind: bool = True,
        enable_slope: bool = True,
    ):
        super().__init__()
        self.cfg = cfg
        if generator_type.lower() == "deeplabv3plus":
            self.generator = DeepLabV3PlusParameterGenerator(
                in_channels=in_channels,
                num_lc_classes=cfg.num_landcover_classes,
                lc_embed_dim=cfg.landcover_embed_dim,
            )
        elif generator_type.lower() == "mscnn":
            self.generator = MSCNNParameterGenerator(
                in_channels=in_channels,
                num_lc_classes=cfg.num_landcover_classes,
                lc_embed_dim=cfg.landcover_embed_dim,
            )
        else:
            raise ValueError(f"Generator tidak dikenal: {generator_type}")

        self.ca = ProbabilisticCellularAutomata(
            cfg=cfg,
            enable_wind=enable_wind,
            enable_slope=enable_slope,
        )

    def forward(
        self,
        x: torch.Tensor,
        landcover_idx: torch.Tensor,
        state_t: torch.Tensor,
        ca_phys: torch.Tensor,
        peat_hydro: Optional[torch.Tensor] = None,
        n_micro_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        param_dict = self.generator(x, landcover_idx)
        ca_out = self.ca(
            state_t=state_t,
            param_dict=param_dict,
            ca_phys=ca_phys,
            peat_hydro=peat_hydro,
            n_micro_steps=n_micro_steps,
        )
        return {**param_dict, **ca_out}

    def rollout_72h(
        self,
        x_seq: List[torch.Tensor],
        landcover_idx: torch.Tensor,
        state_t0: torch.Tensor,
        ca_phys_seq: List[torch.Tensor],
        peat_hydro_seq: Optional[List[torch.Tensor]] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Rollout autoregresif murni t -> t+24 -> t+48 -> t+72 jam (PRD Bagian 2.9).
        Tidak ada state api teramati dari masa depan yang dimasukkan kembali.
        Pada setiap macro-step, kanal fire_t pada prediktor diperbarui dengan P_F prediksi sebelumnya.
        """
        outputs = []
        curr_state = state_t0
        for step_idx in range(len(x_seq)):
            x_curr = x_seq[step_idx].clone()
            if step_idx > 0:
                # Perbarui kanal fire_t (indeks 0) dengan probabilitas burning prediksi langkah sebelumnya
                x_curr[:, 0:1] = curr_state[:, 1:2]

            ph = peat_hydro_seq[step_idx] if peat_hydro_seq is not None else None
            step_out = self.forward(
                x=x_curr,
                landcover_idx=landcover_idx,
                state_t=curr_state,
                ca_phys=ca_phys_seq[step_idx],
                peat_hydro=ph,
            )
            outputs.append(step_out)
            curr_state = step_out["state_next"]
        return outputs
