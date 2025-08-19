import torch

def gaussian_nll_loss(pred_mean, pred_var, target):
    pred_var = pred_var.clamp(min=1e-3, max=1e3)
    loss = 0.5 * torch.mean(
        torch.log(pred_var) + (target - pred_mean)**2 / pred_var
    )
    return loss

def compute_rmse(pred_mean, target):
    return torch.sqrt(torch.mean((pred_mean - target) ** 2))

def compute_picp(pred_mean, pred_var, target, alpha=0.05):
    std = torch.sqrt(pred_var)
    z = torch.tensor(torch.distributions.Normal(0,1).icdf(torch.tensor(1-alpha/2)))
    lower = pred_mean - z * std
    upper = pred_mean + z * std
    inside = (target >= lower) & (target <= upper)
    return inside.float().mean()
