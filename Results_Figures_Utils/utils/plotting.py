import os
import numpy as np
import matplotlib.pyplot as plt


def save_run_results(run_name,
                     nmse_L_list,
                     nmse_S_list,
                     iterations_list,
                     L0_rep,
                     L_hat_rep,
                     S0_rep,
                     S_hat_rep,
                     convergence_curve=None,
                     layerwise_nmse_L=None,
                     layerwise_nmse_S=None,
                     extra_params=None,
                     base_path=None,
                     zoom=None,
                     show_plots=True):

    if base_path is None:
        raise ValueError("Please provide base_path for saving results.")

    metrics_path = os.path.join(base_path, "results_metrics")
    figures_path = os.path.join(base_path, "results_figures")
    os.makedirs(metrics_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    # Metrics aggregation

    nmse_L_mean = float(np.mean(nmse_L_list))
    nmse_S_mean = float(np.mean(nmse_S_list))
    iter_mean = float(np.mean(iterations_list))

    metrics = {
        "NMSE_L_mean": nmse_L_mean,
        "NMSE_L_std": float(np.std(nmse_L_list)),
        "NMSE_S_mean": nmse_S_mean,
        "NMSE_S_std": float(np.std(nmse_S_list)),
        "Iterations_mean": iter_mean,
        "Iterations_std": float(np.std(iterations_list)),
    }

    if extra_params is not None:
        metrics.update(extra_params)

    np.savez(os.path.join(metrics_path, f"{run_name}_metrics.npz"), **metrics)

    # Optional zoom

    if zoom is not None:
        r1, r2, c1, c2 = zoom
        L0_rep = L0_rep[r1:r2, c1:c2]
        L_hat_rep = L_hat_rep[r1:r2, c1:c2]
        S0_rep = S0_rep[r1:r2, c1:c2]
        S_hat_rep = S_hat_rep[r1:r2, c1:c2]

    # Scaling

    vmax_L = np.percentile(np.abs(L0_rep), 99)
    vmax_S = np.percentile(np.abs(S0_rep), 99)
    vmax_EL = 0.05
    vmax_ES = 0.05

    aspect_ratio = L0_rep.shape[1] / L0_rep.shape[0]

    # =========================================================
    # Figure 1 — Low Rank
    # =========================================================

    fig1 = plt.figure(figsize=(14, 5))

    panels = [
        (L0_rep, "True $L_0$", "viridis", -vmax_L, vmax_L),
        (L_hat_rep, "Estimated $\hat{L}$", "viridis", -vmax_L, vmax_L),
        (np.abs(L0_rep - L_hat_rep), r"$|L_0 - \hat{L}|$", "Reds", 0, vmax_EL)
    ]

    for i, (data, title, cmap, vmin, vmax) in enumerate(panels):
        ax = plt.subplot(1, 3, i + 1)
        im = ax.imshow(data, aspect=aspect_ratio,
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.suptitle(f"{run_name} — LOW RANK", fontsize=14)
    plt.tight_layout()
    fig1.savefig(os.path.join(figures_path, f"{run_name}_LOWRANK.png"), dpi=300)

    if show_plots:
        plt.show()
    plt.close(fig1)

    # =========================================================
    # Figure 2 — Sparse
    # =========================================================

    fig2 = plt.figure(figsize=(14, 5))

    panels = [
        (S0_rep, "True $S_0$", "seismic", -vmax_S, vmax_S),
        (S_hat_rep, "Estimated $\hat{S}$", "seismic", -vmax_S, vmax_S),
        (np.abs(S0_rep - S_hat_rep), r"$|S_0 - \hat{S}|$", "Reds", 0, vmax_ES)
    ]

    for i, (data, title, cmap, vmin, vmax) in enumerate(panels):
        ax = plt.subplot(1, 3, i + 1)
        im = ax.imshow(data, aspect=aspect_ratio,
                       cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.suptitle(f"{run_name} — SPARSE", fontsize=14)
    plt.tight_layout()
    fig2.savefig(os.path.join(figures_path, f"{run_name}_SPARSE.png"), dpi=300)

    if show_plots:
        plt.show()
    plt.close(fig2)

    # =========================================================
    # Figure 3 — Diagnostics
    # =========================================================

    fig3 = plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 2, 1)

    if convergence_curve is not None:
        ax1.semilogy(convergence_curve, linewidth=2)
        ax1.set_title("ADMM Convergence")
        ax1.set_xlabel("Iterations")
        ax1.set_ylabel("Relative Residual")
        label_word = "Iterations"

    elif layerwise_nmse_L is not None:
        ax1.plot(layerwise_nmse_L, 'o-', label="NMSE_L")
        ax1.plot(layerwise_nmse_S, 's-', label="NMSE_S")
        ax1.set_title("Layer-wise NMSE")
        ax1.set_xlabel("Layers")
        ax1.set_ylabel("NMSE")
        ax1.legend()
        label_word = "Layers"

    else:
        label_word = "Iterations"

    ax1.grid(True)

    textstr = '\n'.join((
        f'NMSE_L = {nmse_L_mean:.6f}',
        f'NMSE_S = {nmse_S_mean:.6f}',
        f'{label_word} = {int(iter_mean)}'
    ))

    ax1.text(
        0.60, 0.95, textstr,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round',
                  facecolor='white',
                  alpha=0.9)
    )

    # Singular values panel

    ax2 = plt.subplot(1, 2, 2)
    s = np.linalg.svd(L_hat_rep, compute_uv=False)

    ax2.semilogy(s, 'o-')
    ax2.set_title("Singular Values")
    ax2.set_xlabel("Index")
    ax2.set_ylabel("σ_i")
    ax2.grid(True)

    singular_line = 5 if "RUN1" in run_name else 13

    ax2.axvline(x=singular_line,
                color='red',
                linestyle='--',
                linewidth=2)

    if singular_line <= len(s):
        ax2.text(singular_line + 1,
                 s[singular_line - 1],
                 f"{singular_line}",
                 color='red')

    plt.suptitle(f"{run_name} — DIAGNOSTICS", fontsize=14)
    plt.tight_layout()

    fig3.savefig(os.path.join(figures_path,
                              f"{run_name}_DIAGNOSTICS.png"),
                 dpi=300)

    if show_plots:
        plt.show()
    plt.close(fig3)

    print(f"Saved structured figures for {run_name}")
