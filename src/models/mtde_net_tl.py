import torch
import torch.nn as nn
import torchvision.models as models

class MTDE_Net_TL(nn.Module):
    """
    MTDE-Net adaptada para Transfer Learning utilizando un backbone preentrenado de ImageNet (ResNet).
    Acepta imágenes térmicas de 1 canal (escala de grises) y las expande automáticamente a 3 canales
    para aprovechar al 100% los pesos preentrenados de ImageNet.
    """
    def __init__(self, backbone_name="resnet18", pretrained=True, freeze_backbone=False, tabular_dim=4, dropout=0.2):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        
        # 1. Cargar el Backbone Preentrenado de ImageNet
        if self.backbone_name == "resnet18":
            if hasattr(models, "ResNet18_Weights"):
                weights = models.ResNet18_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet18(weights=weights)
            else:
                self.backbone = models.resnet18(pretrained=pretrained)
            self.visual_dim = self.backbone.fc.in_features  # Generalmente 512
            self.backbone.fc = nn.Identity()
            
        elif self.backbone_name == "resnet50":
            if hasattr(models, "ResNet50_Weights"):
                weights = models.ResNet50_Weights.DEFAULT if pretrained else None
                self.backbone = models.resnet50(weights=weights)
            else:
                self.backbone = models.resnet50(pretrained=pretrained)
            self.visual_dim = self.backbone.fc.in_features  # Generalmente 2048
            self.backbone.fc = nn.Identity()
            
        else:
            raise ValueError(f"Backbone no soportado: {backbone_name}. Elige 'resnet18' o 'resnet50'.")
            
        # Opcional: Congelar los pesos del backbone para extracción estática de características
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        # 2. Rama Tabular
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 32),
            nn.BatchNorm1d(32),
            nn.PReLU(32),
            nn.Linear(32, 64),
            nn.PReLU(64)
        )
        
        # 3. Cabezal de Regresión Fused (Multimodal)
        self.multimodal_head = nn.Sequential(
            nn.Linear(self.visual_dim + 64, 256),
            nn.BatchNorm1d(256),
            nn.PReLU(256),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Softplus()  # Regresión estrictamente positiva
        )

    def forward(self, x_img: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        # Si la imagen viene con 1 canal, expandirla a 3 canales para compatibilidad con ImageNet
        if x_img.size(1) == 1:
            x_img = x_img.expand(-1, 3, -1, -1)
            
        feat_vis = self.backbone(x_img)
        feat_tab = self.tabular_branch(x_tab)
        
        feat_fused = torch.cat([feat_vis, feat_tab], dim=1)
        return self.multimodal_head(feat_fused).squeeze(1)
