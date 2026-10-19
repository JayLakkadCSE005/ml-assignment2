#!/usr/bin/env python3
"""
Figure Generation Script:
Generates publication-quality figures for the CVPR two-column report.
Saves all figures in both PDF (for LaTeX inclusion) and PNG (for quick preview).
"""
import os
import sys
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# Ensure project root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data_loader import load_splits
from src.bagging import train_bagging_ensemble, evaluate_bagging_metrics, evaluate_ensemble
from src.samme import fit_samme_full, predict_samme
from src.bias_variance import generate_stratified_resamples, compute_bias_variance

def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "grid.alpha": 0.5,
        "grid.linestyle": "--"
    })

def main():
    setup_style()
    config_path = os.path.join(os.path.dirname(__file__), "../configs/config.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config.get("random_seed", 26005)
    splits = load_splits()
    x_train, y_train = splits["x_train"], splits["y_train"]
    x_test, y_test   = splits["x_test"], splits["y_test"]
    x_test_masked    = splits["x_test_masked"]

    out_dir = "figures"
    os.makedirs(out_dir, exist_ok=True)
    print("Generating publication-ready figures in figures/ ...")

    # =========================================================================
    # Figure 1: Q1(c) - Bagging Metrics vs B
    # =========================================================================
    print("1. Plotting Fig 1: Q1(c) Bagging Metrics vs B...")
    B_values = config["question_1"]["bagging"]["b_values"]
    if os.path.exists("logs/q1_results.json"):
        import json
        with open("logs/q1_results.json") as jf:
            df_q1bc = pd.DataFrame(json.load(jf)["q1bc_bagging"])
    else:
        trees, oob_masks = train_bagging_ensemble(x_train, y_train, B_values=B_values, base_seed=seed)
        df_q1bc = evaluate_bagging_metrics(trees, oob_masks, x_train, y_train, B_values=B_values, num_classes=40)

    fig1, ax1 = plt.subplots(figsize=(6, 3.8))
    ax1.plot(df_q1bc["B"], df_q1bc["oob_acc"], marker="o", color="#1f77b4", label="OOB Accuracy")
    ax1.plot(df_q1bc["B"], df_q1bc["disagreement"], marker="s", color="#ff7f0e", label="Avg Pairwise Disagreement")
    ax1.plot(df_q1bc["B"], df_q1bc["vote_margin"], marker="^", color="#2ca02c", label="Mean Vote Margin")
    ax1.set_xlabel("Number of Trees ($B$)")
    ax1.set_ylabel("Metric Value")
    ax1.set_title("OOB Accuracy, Disagreement, and Vote Margin vs. Ensemble Size ($B$)")
    ax1.set_xticks(B_values)
    ax1.grid(True)
    ax1.legend(loc="center right")
    fig1.tight_layout()
    fig1.savefig(f"{out_dir}/fig1_q1c_bagging_metrics.pdf", bbox_inches="tight")
    fig1.savefig(f"{out_dir}/fig1_q1c_bagging_metrics.png", dpi=300, bbox_inches="tight")
    plt.close(fig1)

    # =========================================================================
    # Figure 2: Q1(d) - Qualitative Occlusion Cases
    # =========================================================================
    print("2. Plotting Fig 2: Q1(d) Qualitative Occlusion Cases...")
    all_four = [2, 4, 0, 17]
    st_preds_map  = {2: 15, 4: 3, 0: 7, 17: 37}
    bag_preds_map = {2: 1,  4: 2, 0: 7, 17: 14}
    titles = ["Corrected #1", "Corrected #2", "Both Wrong #1", "Both Wrong #2"]

    fig2, axes2 = plt.subplots(1, 4, figsize=(10, 2.8))
    for ax, idx, t_label in zip(axes2, all_four, titles):
        ax.imshow(x_test[idx].reshape(64, 64), cmap="gray")
        ax.set_title(f"{t_label} (Idx {idx})\nTrue: {y_test[idx]} | Tree: {st_preds_map[idx]} | Bag: {bag_preds_map[idx]}", fontsize=8)
        ax.axis("off")
    fig2.tight_layout()
    fig2.savefig(f"{out_dir}/fig2_q1d_occlusion_cases.pdf", bbox_inches="tight")
    fig2.savefig(f"{out_dir}/fig2_q1d_occlusion_cases.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)

    # =========================================================================
    # Figure 3: Q2(b) - 4-Panel SAMME Diagnostics Across 100 Rounds
    # =========================================================================
    print("3. Plotting Fig 3: Q2(b) SAMME Multi-Round Diagnostics...")
    boost_depths = [1, 2, 3, 4]
    models = {}
    colors = {1: "#7f7f7f", 2: "#2ca02c", 3: "#ff7f0e", 4: "#d62728"}
    cache_path = "logs/samme_models.npz"

    if os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=True)
        for d in boost_depths:
            models[d] = {
                "errors": cached[f"err_{d}"],
                "alphas": cached[f"alpha_{d}"],
                "max_weights": cached[f"max_w_{d}"],
                "top10_weights": cached[f"top10_w_{d}"],
                "weight_history": cached[f"w_hist_{d}"],
                "trees": cached[f"trees_{d}"]
            }
    else:
        save_dict = {}
        for d in boost_depths:
            m = fit_samme_full(x_train, y_train, max_depth=d, n_rounds=100, K=40, base_seed=seed)
            models[d] = m
            save_dict[f"err_{d}"] = m["errors"]
            save_dict[f"alpha_{d}"] = m["alphas"]
            save_dict[f"max_w_{d}"] = m["max_weights"]
            save_dict[f"top10_w_{d}"] = m["top10_weights"]
            save_dict[f"w_hist_{d}"] = m["weight_history"]
            save_dict[f"trees_{d}"] = np.array(m["trees"], dtype=object)
        np.savez(cache_path, **save_dict)

    fig3, axes3 = plt.subplots(2, 2, figsize=(9, 6))
    rounds = np.arange(1, 101)

    for d in boost_depths:
        m = models[d]
        axes3[0, 0].plot(rounds, m["errors"][:100], color=colors[d], label=f"Depth {d}")
        axes3[0, 1].plot(rounds, m["alphas"][:100], color=colors[d], label=f"Depth {d}")
        axes3[1, 0].plot(rounds, m["max_weights"][:100], color=colors[d], label=f"Depth {d}")
        axes3[1, 1].plot(rounds, m["top10_weights"][:100], color=colors[d], label=f"Depth {d}")

    # Theoretical random guessing threshold: 1 - 1/40 = 0.975
    axes3[0, 0].axhline(0.975, color="black", linestyle=":", label=r"Threshold $1 - 1/K$")
    axes3[0, 0].set_title(r"(a) Weighted Training Error ($\epsilon_t$)")
    axes3[0, 0].set_xlabel("Round ($t$)")
    axes3[0, 0].set_ylabel(r"$\epsilon_t$")
    axes3[0, 0].grid(True)
    axes3[0, 0].legend(loc="lower right")

    axes3[0, 1].set_title(r"(b) Learner Weight ($\alpha_t$)")
    axes3[0, 1].set_xlabel("Round ($t$)")
    axes3[0, 1].set_ylabel(r"$\alpha_t$")
    axes3[0, 1].grid(True)
    axes3[0, 1].legend(loc="upper right")

    axes3[1, 0].set_title(r"(c) Maximum Sample Weight ($\max_i w_i^{(t)}$)")
    axes3[1, 0].set_xlabel("Round ($t$)")
    axes3[1, 0].set_ylabel(r"$\max w_i$")
    axes3[1, 0].grid(True)
    axes3[1, 0].legend(loc="upper left")

    axes3[1, 1].set_title("(d) Total Weight of 10 Heaviest Samples")
    axes3[1, 1].set_xlabel("Round ($t$)")
    axes3[1, 1].set_ylabel("Top-10 Weight Sum")
    axes3[1, 1].grid(True)
    axes3[1, 1].legend(loc="upper left")

    fig3.tight_layout()
    fig3.savefig(f"{out_dir}/fig3_q2b_samme_diagnostics.pdf", bbox_inches="tight")
    fig3.savefig(f"{out_dir}/fig3_q2b_samme_diagnostics.png", dpi=300, bbox_inches="tight")
    plt.close(fig3)

    # =========================================================================
    # Figure 4: Q2(d) - Heaviest Training Sample and Trajectory
    # =========================================================================
    print("4. Plotting Fig 4: Q2(d) Heaviest Training Sample...")
    sel_model = models[4]
    w_hist = sel_model["weight_history"][:-1]  # (100, 240)
    peak_round, hardest_sample_idx = np.unravel_index(np.argmax(w_hist), w_hist.shape)
    true_hard_class = int(y_train[hardest_sample_idx])

    hard_sample_x = x_train[hardest_sample_idx:hardest_sample_idx+1]
    weak_preds_hard = np.array([t.predict(hard_sample_x)[0] for t in sel_model["trees"]])
    was_correct = (weak_preds_hard == true_hard_class)

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(8.5, 3.2), gridspec_kw={'width_ratios': [1, 2.2]})
    ax4a.imshow(x_train[hardest_sample_idx].reshape(64, 64), cmap="gray")
    ax4a.set_title(f"Heaviest Face #{hardest_sample_idx}\n(True Class: {true_hard_class})")
    ax4a.axis("off")

    ax4b.plot(np.arange(1, 101), w_hist[:, hardest_sample_idx], color="#d62728", label=r"Sample Weight $w_i^{(t)}$")
    correct_rounds = np.where(was_correct)[0] + 1
    ax4b.scatter(correct_rounds, w_hist[was_correct, hardest_sample_idx], color="#2ca02c", zorder=5, label="Weak Learner Correct")
    ax4b.set_title(f"Weight Trajectory Across 100 SAMME Rounds")
    ax4b.set_xlabel("Round ($t$)")
    ax4b.set_ylabel("Normalized Weight")
    ax4b.grid(True)
    ax4b.legend()

    fig4.tight_layout()
    fig4.savefig(f"{out_dir}/fig4_q2d_hardest_sample.pdf", bbox_inches="tight")
    fig4.savefig(f"{out_dir}/fig4_q2d_hardest_sample.png", dpi=300, bbox_inches="tight")
    plt.close(fig4)

    # =========================================================================
    # Figure 5: Q3(b) - Bias^2 and Variance across Tree Depths
    # =========================================================================
    print("5. Plotting Fig 5: Q3(b) Bias^2 and Variance vs Depth...")
    depth_labels = ["Depth 4", "Depth 16", "Unbounded (None)"]
    st_bias_sq  = [0.8692, 0.6655, 0.3984]
    bg_bias_sq  = [0.8959, 0.6809, 0.5107]
    st_variance = [0.0706, 0.2290, 0.6491]
    bg_variance = [0.0131, 0.0210, 0.0255]

    x = np.arange(len(depth_labels))
    width = 0.2

    fig5, ax5 = plt.subplots(figsize=(7, 3.5))
    ax5.bar(x - 1.5*width, st_bias_sq, width, label=r"Single Tree Bias$^2$", color="#3182bd")
    ax5.bar(x - 0.5*width, bg_bias_sq, width, label=r"Bagging(100) Bias$^2$", color="#9ecae1")
    ax5.bar(x + 0.5*width, st_variance, width, label="Single Tree Variance", color="#e6550d")
    ax5.bar(x + 1.5*width, bg_variance, width, label="Bagging(100) Variance", color="#fdae6b")

    ax5.set_ylabel("Metric Value")
    ax5.set_title(r"Empirical Bias$^2$ and Variance: Single Tree vs. Bagging(100) across Depths")
    ax5.set_xticks(x)
    ax5.set_xticklabels(depth_labels)
    ax5.grid(True, axis="y")
    ax5.legend(loc="upper right", ncol=2)

    fig5.tight_layout()
    fig5.savefig(f"{out_dir}/fig5_q3b_bias_variance_depth.pdf", bbox_inches="tight")
    fig5.savefig(f"{out_dir}/fig5_q3b_bias_variance_depth.png", dpi=300, bbox_inches="tight")
    plt.close(fig5)

    print(f"\nAll figures successfully generated in {out_dir}/")

if __name__ == "__main__":
    main()
