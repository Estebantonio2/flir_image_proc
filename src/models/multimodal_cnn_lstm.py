import torch
import torch.nn as nn
from src.models.cnn_lstm import LiteDSTFSBackbone

class MultimodalThermalCNNLSTM(nn.Module):
    """
    Modelo Multimodal CNN-LSTM que fusiona imágenes secuenciales (rama visual)
    con variables contextuales tabulares (rama tabular) usando Fusión Tardía (Late Fusion).
    """
    def __init__(self, lstm_hidden_dim: int = 128, lstm_layers: int = 1, dropout: float = 0.2, tabular_dim: int = 4):
        super().__init__()
        self.backbone = LiteDSTFSBackbone()
        
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True
        )
        
        # Cabeza de regresión multimodal (Fusión Tardía):
        # Toma el estado del LSTM (lstm_hidden_dim) + variables tabulares (tabular_dim)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_dim + tabular_dim, 64),
            nn.PReLU(64),
            nn.Linear(64, 1),
            nn.Softplus()
        )

    def forward(self, x_seq: torch.Tensor, x_tab: torch.Tensor) -> torch.Tensor:
        # x_seq: (B, L, 1, H, W)
        # x_tab: (B, tabular_dim)
        batch_size, seq_len, channels, height, width = x_seq.size()
        
        # 1. Extracción de características visuales por frame
        x_flat = x_seq.view(batch_size * seq_len, channels, height, width)
        feats_flat = self.backbone(x_flat) # (B * L, 256)
        
        # 2. Paso por la celda recurrent LSTM
        feats_seq = feats_flat.view(batch_size, seq_len, -1) # (B, L, 256)
        lstm_out, _ = self.lstm(feats_seq) # (B, L, lstm_hidden_dim)
        last_step_feat = lstm_out[:, -1, :] # (B, lstm_hidden_dim)
        
        # 3. Concatenación multimodal (Fusión Tardía)
        x_combined = torch.cat([last_step_feat, x_tab], dim=1) # (B, lstm_hidden_dim + tabular_dim)
        
        # 4. Cabeza de Regresión
        return self.regressor(x_combined).squeeze(1)
