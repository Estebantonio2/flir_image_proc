import torch
import torch.nn as nn
from src.models.dstfs import SoftThresholdPReLU, SpatialAttention

class SPPChannelAttention(nn.Module):
    """
    MTDE-Net: Channel Attention (CA) con Spatial Pyramid Pooling (SPP) multiescala.
    """
    def __init__(self, channels: int, pool_scales=(1, 2, 4)):
        super().__init__()
        self.scales = pool_scales
        hidden = max(channels // 16, 4)
        self.fc1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.size(2), x.size(3)
        spp_feats = []
        for scale in self.scales:
            stride = (h // scale, w // scale)
            kernel = (h // scale + (h % scale > 0), w // scale + (w % scale > 0))
            avg_p = nn.functional.adaptive_avg_pool2d(nn.functional.avg_pool2d(x, kernel_size=kernel, stride=stride), 1)
            max_p = nn.functional.adaptive_max_pool2d(nn.functional.max_pool2d(x, kernel_size=kernel, stride=stride), 1)
            spp_feats.append(avg_p + max_p)
        spp_sum = sum(spp_feats) / len(self.scales)
        return x * torch.sigmoid(self.fc2(self.relu(self.fc1(spp_sum))))

class ResidualAttentionBlock(nn.Module):
    """
    MTDE-Net: Bloque residual que combina SPRelu con atención dual (SA + CA con SPP).
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(out_c), nn.BatchNorm2d(out_c)
        self.act1 = SoftThresholdPReLU(out_c) if stride == 1 else nn.PReLU(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.att_c, self.att_s = SPPChannelAttention(out_c), SpatialAttention()
        self.act2 = nn.PReLU(out_c)
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride, bias=False), 
            nn.BatchNorm2d(out_c)
        ) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        att_res = self.att_c(res) + self.att_s(res)
        return self.act2(att_res + self.skip(x))

class MTDE_Net(nn.Module):
    """
    Multimodal Thermal Decay Estimation Network (MTDE-Net).
    Integra características espaciales de la imagen térmica con un vector
    tabular enriquecido de 10 variables (ambientales, físicas y estadísticas de contraste, excluyendo género por contexto forense).
    """
    def __init__(self, tabular_dim=10, dropout=0.2):
        super().__init__()
        # 1. Rama Visual Lite
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, 7, 2, 3, bias=False), 
            nn.BatchNorm2d(32), 
            nn.PReLU(32)
        )
        self.stage1 = ResidualAttentionBlock(32, 32, stride=1)  # 56x56x32
        self.stage2 = ResidualAttentionBlock(32, 64, stride=2)  # 28x28x64
        self.stage3 = ResidualAttentionBlock(64, 128, stride=2) # 14x14x128
        self.stage4 = ResidualAttentionBlock(128, 256, stride=2) # 7x7x256
        self.visual_gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # 2. Rama Tabular
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.Linear(32, 64),
            nn.PReLU(64)
        )
        
        # 3. Cabezal de Regresión Multimodal
        self.multimodal_head = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(256),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Softplus()
        )

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        feat_vis = self.visual_gap(self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x_img))))))
        feat_tab = self.tabular_branch(x_tab)
        feat_fused = torch.cat([feat_vis, feat_tab], dim=1)
        return self.multimodal_head(feat_fused).squeeze(1)
