import torch


def soft_threshold(X, tau):
    return torch.sign(X) * torch.relu(torch.abs(X) - tau)


def svt(X, tau):
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    S_thresh = torch.relu(S - tau)
    return U @ torch.diag_embed(S_thresh) @ Vh


def admm_rpca(Y, lambda_val, rho, max_iter=2000, tol=1e-4):

    L = torch.zeros_like(Y)
    S = torch.zeros_like(Y)
    U = torch.zeros_like(Y)

    convergence = []
    norm_Y = torch.norm(Y) + 1e-8

    for k in range(max_iter):

        # ADMM updates
        L = svt(Y - S + U, 1.0 / rho)
        S = soft_threshold(Y - L + U, lambda_val / rho)

        residual = Y - L - S
        U = U + residual

        # Relative residual
        err = torch.norm(residual) / norm_Y
        convergence.append(err.item())

        if err < tol:
            break

    return L, S, k + 1, convergence
