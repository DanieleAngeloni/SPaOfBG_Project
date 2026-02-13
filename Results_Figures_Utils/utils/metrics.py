import torch


def compute_nmse(L, L0):
    return torch.norm(L - L0)**2 / (torch.norm(L0)**2 + 1e-8)


def compute_metrics(L, S, L0, S0):
    nmse_L = compute_nmse(L, L0)
    nmse_S = compute_nmse(S, S0)
    return nmse_L, nmse_S
