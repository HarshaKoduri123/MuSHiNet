import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import xarray as xr
import numpy as np

from model import MuSHiNet
from dataloader import OceanDataset
from utils import gaussian_nll_loss, compute_rmse, compute_picp
from test import evaluate as test_evaluate   


def load_nc_files(wave_path, wind_path):
    """Load ERA5 netCDF files for SWH and 10m winds, with ocean masking."""
    ds_wave = xr.open_dataset(wave_path)
    ds_wind = xr.open_dataset(wind_path)

    # Extract variables
    swh = ds_wave["swh"]
    u10 = ds_wind["u10"]
    v10 = ds_wind["v10"]

    # Mask land points
    ocean_mask = ~swh.isnull().any(dim="valid_time")
    swh = np.nan_to_num(swh.where(ocean_mask, drop=True), nan=0.0)
    u10 = np.nan_to_num(u10.where(ocean_mask, drop=True), nan=0.0)
    v10 = np.nan_to_num(v10.where(ocean_mask, drop=True), nan=0.0)

    return swh, u10, v10


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss, total_rmse, total_picp = 0.0, 0.0, 0.0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        mean, log_var = model(x)
        var = torch.exp(log_var)

        loss = gaussian_nll_loss(mean, var, y)
        rmse = compute_rmse(mean, y)
        picp = compute_picp(mean, var, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rmse += rmse.item()
        total_picp += picp.item()

    n_batches = len(loader)
    return total_loss / n_batches, total_rmse / n_batches, total_picp / n_batches


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_rmse, total_picp = 0.0, 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)

            mean, log_var = model(x)
            var = torch.exp(log_var)

            loss = gaussian_nll_loss(mean, var, y)
            rmse = compute_rmse(mean, y)
            picp = compute_picp(mean, var, y)

            total_loss += loss.item()
            total_rmse += rmse.item()
            total_picp += picp.item()

    n_batches = len(loader)
    return total_loss / n_batches, total_rmse / n_batches, total_picp / n_batches


def main():
    # Load config
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    swh, u10, v10 = load_nc_files(cfg["dataset"]["paths"]["wave"], cfg["dataset"]["paths"]["wind"])

    dataset = OceanDataset(swh, u10, v10, seq_len=cfg["dataset"]["seq_len"])

    # Train/Val/Test split
    total_size = len(dataset)
    train_size = int(total_size * cfg["dataset"]["train_split"])
    val_size = int(total_size * cfg["dataset"]["val_split"])
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

    # Model
    model = MuSHiNet(in_channels=cfg["model"]["input_dim"]
                     ).to(device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )

    # Training loop
    for epoch in range(cfg["training"]["epochs"]):
        train_loss, train_rmse, train_picp = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_rmse, val_picp = evaluate(model, val_loader, device)

        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} "
              f"| Train Loss: {train_loss:.4f}, RMSE: {train_rmse:.4f}, PICP: {train_picp:.4f} "
              f"| Val Loss: {val_loss:.4f}, RMSE: {val_rmse:.4f}, PICP: {val_picp:.4f}")

    # Save model
    torch.save(model.state_dict(), cfg["training"]["save_path"])
    print(f"Model saved to {cfg['training']['save_path']}")

    # --- Run Test Evaluation ---
    print("\nRunning final test evaluation...")
    test_evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()

