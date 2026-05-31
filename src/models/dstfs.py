import torch
import torch.nn as nn

class SoftThresholdPReLU(nn.Module):
    """
    DSTFS: Umbral Suave PReLU (SPRelu) dinámico y adaptativo.
    Calcula un umbral t dependiente de la muestra mediante GAP y un MLP de 2 capas.
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
        x_act = self.prelu(x)
        abs_x = torch.abs(x_act)
        g = torch.mean(abs_x, dim=(2, 3), keepdim=True)
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(g))))
        t = scale * g
        return torch.sign(x_act) * torch.relu(abs_x - t)

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

class ResidualAttentionBlock(nn.Module):
    """
    DSTFS: Bloque residual que combina SPRelu con atención dual (SA + CA).
    """
    def __init__(self, in_c: int, out_c: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1, self.bn2 = nn.BatchNorm2d(out_c), nn.BatchNorm2d(out_c)
        self.act1 = SoftThresholdPReLU(out_c) if stride == 1 else nn.PReLU(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.att_c, self.att_s = ChannelAttention(out_c), SpatialAttention()
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

class ThermalDepartureTimeNet(nn.Module):
    """
    DSTFS Temporal (Tarea única): Stem -> 4 Stages -> Head.
    """
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, 7, 2, 3, bias=False), 
            nn.BatchNorm2d(64), 
            nn.PReLU(64)
        )
        self.stage1 = self._stage(64, 64, 2, 1)  # 56x56x64
        self.stage2 = self._stage(64, 128, 2, 2) # 28x28x128
        self.stage3 = self._stage(128, 256, 2, 2) # 14x14x256
        self.stage4 = self._stage(256, 512, 2, 2) # 7x7x512
        
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), 
            nn.Flatten(), 
            nn.Dropout(0.25), 
            nn.Linear(512, 512), 
            nn.PReLU(512), 
            nn.Linear(512, 1), 
            nn.Softplus()
        )

    def _stage(self, in_c, out_c, blocks, stride):
        layers = [ResidualAttentionBlock(in_c, out_c, stride)] + [ResidualAttentionBlock(out_c, out_c, 1) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.stage4(self.stage3(self.stage2(self.stage1(self.stem(x)))))).squeeze(1)
