import torch
import torch.nn as nn
from src.models.mtde_net import ResidualAttentionBlock

class MTDE_Net_v2(nn.Module):
    """
    Multimodal Thermal Decay Estimation Network version 2 (MTDE-Net v2).
    Integra características espaciales de la imagen térmica con un vector
    tabular enriquecido de 12 variables (ambientales, físicas y estadísticas de contraste).
    """
    def __init__(self, tabular_dim=12, dropout=0.2):
        super().__init__()
        # 1. Rama Visual Lite (Backbone extractor de 256 dimensiones)
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
        
        # 2. Rama Tabular MLP (Procesa las 12 características termofísicas)
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
            nn.Softplus() # Garantiza predicción de tiempo estrictamente positiva
        )

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        feat_vis = self.visual_gap(self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x_img))))))
        feat_tab = self.tabular_branch(x_tab)
        
        # Fusión intermedia por concatenación
        feat_fused = torch.cat([feat_vis, feat_tab], dim=1)
        return self.multimodal_head(feat_fused).squeeze(1)
