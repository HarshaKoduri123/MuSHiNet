import torch
from utils import compute_rmse, compute_picp

def evaluate(model, test_loader, device="cuda"):
    model.eval()
    rmse, picp, count = 0, 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            yb_step = yb[:, -1]
            mean, var = model(xb)
            B = xb.size(0)
            rmse += compute_rmse(mean, yb_step).item() * B
            picp += compute_picp(mean, var, yb_step).item() * B
            count += B
    print(f"Test RMSE: {rmse/count:.4f}, Test PICP: {picp/count:.4f}")
