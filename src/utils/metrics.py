import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def eval_dstfs_metrics(model: nn.Module, loader: DataLoader, device: str, scale: float = 1.0) -> dict[str, float]:
    """
    Calcula las métricas unificadas para el modelo DSTFS en segundos reales.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            p = model(x)
            all_preds.append(p.cpu() * scale)
            all_targets.append(y.cpu() * scale)
            
    return _compute_metrics(torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0))

def eval_mtde_net_metrics(model: nn.Module, loader: DataLoader, device: str, scale: float = 30.0) -> dict[str, float]:
    """
    Calcula las métricas unificadas para el modelo MTDE-Net desescalando las predicciones a segundos reales.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_img, x_tab, y in loader:
            x_img, x_tab, y = x_img.to(device), x_tab.to(device), y.to(device)
            p = model(x_img, x_tab)
            all_preds.append(p.cpu() * scale)
            all_targets.append(y.cpu() * scale)
            
    return _compute_metrics(torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0))

def eval_cnn_lstm_metrics(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    """
    Calcula las métricas unificadas para el modelo CNN-LSTM en segundos reales.
    """
    return eval_dstfs_metrics(model, loader, device)

def eval_cnn_1d_metrics(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    """
    Calcula las métricas unificadas para el modelo CNN 1D en segundos reales.
    """
    return eval_dstfs_metrics(model, loader, device)


def eval_multimodal_cnn_lstm_metrics(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    """
    Calcula las métricas unificadas para el modelo Multimodal CNN-LSTM en segundos reales.
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x_seq, x_tab, y in loader:
            x_seq, x_tab, y = x_seq.to(device), x_tab.to(device), y.to(device)
            p = model(x_seq, x_tab)
            all_preds.append(p.cpu())
            all_targets.append(y.cpu())
            
    return _compute_metrics(torch.cat(all_preds, dim=0), torch.cat(all_targets, dim=0))

def _compute_metrics(all_p: torch.Tensor, all_y: torch.Tensor) -> dict[str, float]:
    """
    Función interna para calcular MAE, RMSE, R2, MAPE, Acc60, Acc120.
    """
    abs_diff = (all_p - all_y).abs()
    sq_diff = (all_p - all_y)**2
    
    mae = abs_diff.mean().item()
    rmse = sq_diff.mean().sqrt().item()
    
    # Coeficiente de determinación R2
    y_mean = all_y.mean()
    ss_tot = ((all_y - y_mean)**2).sum().item()
    ss_res = sq_diff.sum().item()
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-6 else 0.0
    
    # MAPE (%)
    mape = (abs_diff / all_y.clamp_min(1e-5)).mean().item() * 100.0
    
    # Tolerancias
    acc60 = (abs_diff <= 60.0).float().mean().item() * 100.0
    acc120 = (abs_diff <= 120.0).float().mean().item() * 100.0
    
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
        "acc60": acc60,
        "acc120": acc120
    }
