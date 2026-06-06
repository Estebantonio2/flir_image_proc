import torch
import torch.nn as nn

class SoftThresholdPReLU(nn.Module):
    """
    DSTFS: Umbral Suave PReLU (SPRelu) dinámico y adaptativo continuo.
    Calcula un umbral t dependiente de la muestra mediante GAP y un MLP de 2 capas,
    y aplica el umbral de manera continua para evitar discontinuidades de gradiente.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.prelu = nn.PReLU(channels)
        hidden = max(channels // 16, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. Calcular el umbral t adaptativo sobre el valor absoluto de la entrada original
        abs_x = torch.abs(x)
        g = torch.mean(abs_x, dim=(2, 3), keepdim=True)
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(g))))
        t = scale * g
        
        # 2. Aplicar umbral de manera continua preservando la pendiente de PReLU
        pos_part = torch.relu(x - t)
        neg_part = -torch.relu(-x - t)
        return pos_part + self.prelu(neg_part)

class ChannelAttention(nn.Module):
    """
    DSTFS: Channel Attention (CA) con pooling promedio y máximo globales.
    """
    def __init__(self, channels: int):
        super().__init__()
        hidden = max(channels // 16, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False), 
            nn.ReLU(True), 
            nn.Conv2d(hidden, channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(self.mlp(x.mean((2, 3), keepdim=True)) + self.mlp(x.amax((2, 3), keepdim=True)))

class SpatialAttention(nn.Module):
    """
    DSTFS: Spatial Attention (SA) mediante convolución 7x7.
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))

class ResidualBlock(nn.Module):
    """
    DSTFS: Bloque residual estándar de ResNet adaptado con SPRelu para stride=1.
    No incluye atención dual interna (conforme al diseño original del paper).
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act1 = SoftThresholdPReLU(out_c) if stride == 1 else nn.PReLU(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.act2 = nn.PReLU(out_c)
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride, bias=False), 
            nn.BatchNorm2d(out_c)
        ) if stride != 1 or in_c != out_c else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.act1(self.bn1(self.conv1(x)))
        res = self.bn2(self.conv2(res))
        return self.act2(res + self.skip(x))

class ThermalDepartureTimeNet(nn.Module):
    """
    DSTFS Temporal (Tarea única): Backbone compartido (Stem -> Stage 1-3) -> 
    Branch temporal (Stage 4 -> Módulo de atención dual paralelo -> Head de regresión lineal).
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.PReLU(64)
        )
        self._stage = lambda in_c, out_c, blocks, stride: nn.Sequential(
            ResidualBlock(in_c, out_c, stride),
            *[ResidualBlock(out_c, out_c, 1) for _ in range(blocks - 1)]
        )
        # Backbone compartido
        self.stage1 = self._stage(64, 64, 2, 1)   # 56x56x64
        self.stage2 = self._stage(64, 128, 2, 2)  # 28x28x128
        self.stage3 = self._stage(128, 256, 2, 2) # 14x14x256
        
        # Rama Temporal
        self.stage4 = self._stage(256, 512, 2, 2) # 7x7x512
        
        # Módulos de atención dual paralelos aplicados tras Stage 4
        self.att_c = ChannelAttention(512)
        self.att_s = SpatialAttention()
        
        # Head de regresión lineal (dos capas lineales)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.Dropout(0.25), 
            nn.Linear(512, 512), 
            nn.PReLU(512), 
            nn.Linear(512, 1) # Salida lineal continua sin Softplus
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Extracción del backbone y rama temporal
        x = self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))
        # Aplicar el módulo de atención dual paralelo: Xtime = X2 * (fCA(X2) + fSA(X2))
        x = self.att_c(x) + self.att_s(x)
        return self.head(x).squeeze(1)
