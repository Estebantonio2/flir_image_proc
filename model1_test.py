from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split


class ThermalTraceDataset(Dataset):
    def __init__(
        self,
        metadata_csv: str | Path = "processed_data/metadata.csv",
        processed_root: str | Path = "processed_data",
        target_column: str = "label_time_s",
        image_size: tuple[int, int] | None = (224, 224),
    ) -> None:
        self.metadata_csv = Path(metadata_csv)
        self.processed_root = Path(processed_root)
        self.target_column = target_column
        self.image_size = image_size

        self.df = pd.read_csv(self.metadata_csv)
        self.df = self.df.dropna(subset=["deltaT_path", target_column]).reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[index]
        delta_t_path = self.processed_root / row["deltaT_path"]

        delta_t = np.load(delta_t_path).astype(np.float32)
        delta_t = self._normalize_trace(delta_t)

        x = torch.from_numpy(delta_t).unsqueeze(0)
        if self.image_size is not None:
            x = nn.functional.interpolate(
                x.unsqueeze(0),
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        y = torch.tensor(float(row[self.target_column]), dtype=torch.float32)
        return x, y

    @staticmethod
    def _normalize_trace(trace: np.ndarray) -> np.ndarray:
        mean = float(np.mean(trace))
        std = float(np.std(trace))
        if std < 1e-6:
            return trace - mean
        return (trace - mean) / std


class SoftThresholdPReLU(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.prelu = nn.PReLU(num_parameters=channels)
        self.threshold = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prelu(x)
        threshold = torch.relu(self.threshold)
        return torch.sign(x) * torch.relu(torch.abs(x) - threshold)


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden_channels = max(channels // reduction, 4)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        max_values = torch.amax(x, dim=(2, 3), keepdim=True)
        weights = self.sigmoid(self.mlp(avg) + self.mlp(max_values))
        return x * weights


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        max_values = torch.amax(x, dim=1, keepdim=True)
        weights = self.sigmoid(self.conv(torch.cat([avg, max_values], dim=1)))
        return x * weights


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        use_soft_threshold: bool = True,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = SoftThresholdPReLU(out_channels) if use_soft_threshold else nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.channel_attention = ChannelAttention(out_channels)
        self.spatial_attention = SpatialAttention()
        self.act2 = nn.PReLU(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.channel_attention(out)
        out = self.spatial_attention(out)

        out = out + identity
        return self.act2(out)


class ThermalDepartureTimeNet(nn.Module):
    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.PReLU(32),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage1 = self._make_stage(32, 32, blocks=2, stride=1)
        self.stage2 = self._make_stage(32, 64, blocks=2, stride=2)
        self.stage3 = self._make_stage(64, 128, blocks=2, stride=2)
        self.stage4 = self._make_stage(128, 256, blocks=2, stride=2)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.25),
            nn.Linear(256, 64),
            nn.PReLU(64),
            nn.Linear(64, 1),
            nn.Softplus(),
        )

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
    ) -> nn.Sequential:
        layers = [
            ResidualAttentionBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                use_soft_threshold=stride == 1,
            )
        ]
        for _ in range(1, blocks):
            layers.append(
                ResidualAttentionBlock(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    use_soft_threshold=True,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return self.head(x).squeeze(1)


class SqrtScaledMSELoss(nn.Module):
    def __init__(self, time_scale: float = 30.0, eps: float = 1e-6) -> None:
        super().__init__()
        self.time_scale = time_scale
        self.eps = eps

    def forward(self, pred_seconds: torch.Tensor, target_seconds: torch.Tensor) -> torch.Tensor:
        pred = torch.sqrt(torch.clamp(pred_seconds / self.time_scale, min=self.eps))
        target = torch.sqrt(torch.clamp(target_seconds / self.time_scale, min=self.eps))
        return nn.functional.mse_loss(pred, target)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_samples += x.size(0)

    return total_loss / total_samples


@torch.no_grad()
def evaluate_mae_seconds(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_abs_error = 0.0
    total_samples = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)
        total_abs_error += torch.sum(torch.abs(pred - y)).item()
        total_samples += x.size(0)

    return total_abs_error / total_samples


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ThermalTraceDataset(
        metadata_csv="processed_data/metadata.csv",
        processed_root="processed_data",
        target_column="label_time_s",
        image_size=(224, 224),
    )

    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = ThermalDepartureTimeNet(input_channels=1).to(device)
    criterion = SqrtScaledMSELoss(time_scale=30.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(1, 21):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_mae = evaluate_mae_seconds(model, val_loader, device)
        print(f"epoch={epoch:02d} train_loss={train_loss:.6f} val_mae_s={val_mae:.2f}")

    torch.save(model.state_dict(), "model1_departure_time.pt")


if __name__ == "__main__":
    main()
