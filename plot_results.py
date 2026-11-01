"""
Visualization & Plotting Script for Machine Translation Assignment 3.
Generates 6 high-resolution publication-quality figures in results/plots/.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_all_plots():
    os.makedirs("results/plots", exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["figure.dpi"] = 180

    with open("results/metrics_summary.json", "r", encoding="utf-8") as f:
        summary_data = json.load(f)
    with open("results/temperature_analysis.json", "r", encoding="utf-8") as f:
        temp_data = json.load(f)
    with open("results/ethical_evaluation.json", "r", encoding="utf-8") as f:
        ethical_data = json.load(f)

    all_configs = summary_data["all_configurations"]
    primary_models = summary_data["primary_models"]
    histories = summary_data["training_histories"]

    # =========================================================================
    # Plot 1: BLEU and chrF Comparison Across All 4 Models (+ Prompting vs LoRA)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    order_4 = ["Basic Transformer", "mT5-small", "LLaMA-3.2-1B", "Gemma-3-1B"]
    labels_4 = [
        "Basic Transformer\n(Enc-Dec, Scratch)",
        "mT5-small\n(Enc-Dec, Fine-Tuned)",
        "LLaMA-3.2-1B\n(Dec-Only, LoRA)",
        "Gemma-3-1B\n(Dec-Only, LoRA)",
    ]
    bleu_4 = [primary_models[m]["bleu"] for m in order_4]
    chrf_4 = [primary_models[m]["chrf"] for m in order_4]

    x = np.arange(len(order_4))
    w = 0.36
    bars1 = axes[0].bar(x - w / 2, bleu_4, w, label="BLEU Score", color="#1f77b4", edgecolor="black", alpha=0.9)
    bars2 = axes[0].bar(x + w / 2, chrf_4, w, label="chrF Score", color="#ff7f0e", edgecolor="black", alpha=0.9)
    axes[0].set_title("A. Primary Models: BLEU & chrF Comparison", fontsize=12, fontweight="bold")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels_4, fontsize=9.5)
    axes[0].set_ylabel("Score (0 - 100)", fontsize=10.5)
    axes[0].set_ylim(0, max(chrf_4) * 1.18)
    axes[0].legend(frameon=True)
    for b in list(bars1) + list(bars2):
        h = b.get_height()
        axes[0].annotate(
            f"{h:.1f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    # Panel B: Adaptation comparison (3-Shot Prompting vs. LoRA Fine-Tuning on Decoder-Only)
    adapt_models = [
        "LLaMA-3.2-1B (Prompting)",
        "LLaMA-3.2-1B",
        "Gemma-3-1B (Prompting)",
        "Gemma-3-1B",
    ]
    adapt_labels = [
        "LLaMA-3.2-1B\n(3-Shot Prompt)",
        "LLaMA-3.2-1B\n(LoRA Tuned)",
        "Gemma-3-1B\n(3-Shot Prompt)",
        "Gemma-3-1B\n(LoRA Tuned)",
    ]
    adapt_bleu = [all_configs[m]["bleu"] for m in adapt_models]
    adapt_chrf = [all_configs[m]["chrf"] for m in adapt_models]
    x2 = np.arange(len(adapt_models))
    b3 = axes[1].bar(x2 - w / 2, adapt_bleu, w, label="BLEU Score", color="#2ca02c", edgecolor="black", alpha=0.9)
    b4 = axes[1].bar(x2 + w / 2, adapt_chrf, w, label="chrF Score", color="#d62728", edgecolor="black", alpha=0.9)
    axes[1].set_title("B. Decoder-Only Adaptation: Prompting vs. LoRA", fontsize=12, fontweight="bold")
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(adapt_labels, fontsize=9.5)
    axes[1].set_ylabel("Score (0 - 100)", fontsize=10.5)
    axes[1].set_ylim(0, max(adapt_chrf) * 1.18)
    axes[1].legend(frameon=True)
    for b in list(b3) + list(b4):
        h = b.get_height()
        axes[1].annotate(
            f"{h:.1f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig("results/plots/bleu_chrf_comparison.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Plot 2: Computational Requirements (Train Time, Inference Latency, Parameters)
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors_4 = ["#3b5998", "#008080", "#e67e22", "#8e44ad"]
    short_names = ["Basic\nTransformer", "mT5-small", "LLaMA-3.2-1B\n(LoRA)", "Gemma-3-1B\n(LoRA)"]

    train_times = [primary_models[m]["train_time_sec"] for m in order_4]
    latencies = [primary_models[m]["latency_ms_per_sent"] for m in order_4]
    total_p = [primary_models[m]["total_params"] / 1e6 for m in order_4]
    train_p = [primary_models[m]["trainable_params"] / 1e6 for m in order_4]

    # Panel 1: Training / Fine-tuning Time
    b_t = axes[0].bar(short_names, train_times, color=colors_4, edgecolor="black", width=0.55)
    axes[0].set_title("A. Training / Fine-Tuning Time (s)", fontsize=11.5, fontweight="bold")
    axes[0].set_ylabel("Time (seconds)", fontsize=10)
    axes[0].set_ylim(0, max(train_times) * 1.18)
    for b in b_t:
        h = b.get_height()
        axes[0].annotate(f"{h:.1f}s", (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    # Panel 2: Inference Latency
    b_l = axes[1].bar(short_names, latencies, color=colors_4, edgecolor="black", width=0.55)
    axes[1].set_title("B. Inference Latency (ms / sentence)", fontsize=11.5, fontweight="bold")
    axes[1].set_ylabel("Latency (ms / sentence)", fontsize=10)
    axes[1].set_ylim(0, max(latencies) * 1.18)
    for b in b_l:
        h = b.get_height()
        axes[1].annotate(f"{h:.1f} ms", (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    # Panel 3: Parameter Count (Total vs Trainable in Millions, Log Scale)
    x_p = np.arange(len(order_4))
    b_p1 = axes[2].bar(x_p - 0.18, total_p, 0.35, label="Total Params (M)", color="#34495e", edgecolor="black")
    b_p2 = axes[2].bar(x_p + 0.18, train_p, 0.35, label="Trainable Params (M)", color="#e74c3c", edgecolor="black")
    axes[2].set_yscale("log")
    axes[2].set_title("C. Model Parameters (Millions, Log Scale)", fontsize=11.5, fontweight="bold")
    axes[2].set_xticks(x_p)
    axes[2].set_xticklabels(short_names)
    axes[2].set_ylabel("Parameters (Millions)", fontsize=10)
    axes[2].set_ylim(0.5, max(total_p) * 5)
    axes[2].legend(frameon=True, fontsize=8.5)
    for b in list(b_p1) + list(b_p2):
        h = b.get_height()
        axes[2].annotate(f"{h:.1f}M", (b.get_x() + b.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/plots/compute_efficiency.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Plot 3: Training & Validation Convergence Curves (Basic Transformer & mT5)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    bt_hist = histories["Basic Transformer"]
    ep_bt = np.arange(1, len(bt_hist["train_loss"]) + 1)
    axes[0].plot(ep_bt, bt_hist["train_loss"], "o-", label="Train Loss (Label Smooth)", color="#1f77b4", linewidth=2)
    axes[0].plot(ep_bt, bt_hist["val_loss"], "s--", label="Validation Loss", color="#d62728", linewidth=2)
    axes[0].set_title("A. Basic Transformer Learning Curve (20 Epochs)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].legend(frameon=True)

    mt5_hist = histories["mT5-small"]
    ep_mt5 = np.arange(1, len(mt5_hist["train_loss"]) + 1)
    axes[1].plot(ep_mt5, mt5_hist["train_loss"], "o-", label="Train Loss", color="#008080", linewidth=2)
    axes[1].plot(ep_mt5, mt5_hist["val_loss"], "s--", label="Validation Loss", color="#e67e22", linewidth=2)
    axes[1].set_title(f"B. mT5-small Fine-Tuning Convergence ({len(ep_mt5)} Epochs)", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Seq2Seq Cross-Entropy Loss")
    axes[1].set_xticks(ep_mt5)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plt.savefig("results/plots/learning_curves.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Plot 4: Sampling-Temperature Analysis (T = 0.0, 0.3, 0.7, 1.0)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
    temps_str = ["0.0", "0.3", "0.7", "1.0"]
    temps_num = [0.0, 0.3, 0.7, 1.0]
    markers = ["o", "s", "^", "D"]

    for idx, m_name in enumerate(order_4):
        m_temp = temp_data[m_name]
        bleu_vals = [m_temp[t]["bleu"] for t in temps_str]
        dist2_vals = [m_temp[t]["distinct_2"] for t in temps_str]
        axes[0].plot(temps_num, bleu_vals, marker=markers[idx], linewidth=2.2, markersize=7, label=m_name, color=colors_4[idx])
        axes[1].plot(temps_num, dist2_vals, marker=markers[idx], linewidth=2.2, markersize=7, label=m_name, color=colors_4[idx])

    axes[0].set_title("A. Effect of Sampling Temperature on BLEU Score", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("Sampling Temperature (T)  [0.0 = Greedy]")
    axes[0].set_ylabel("Corpus BLEU Score")
    axes[0].set_xticks(temps_num)
    axes[0].legend(frameon=True)

    axes[1].set_title("B. Effect of Sampling Temperature on Lexical Diversity (Distinct-2)", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Sampling Temperature (T)  [0.0 = Greedy]")
    axes[1].set_ylabel("Distinct-2 Bigram Ratio")
    axes[1].set_xticks(temps_num)
    axes[1].legend(frameon=True)

    plt.tight_layout()
    plt.savefig("results/plots/temperature_sensitivity.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Plot 5: Architecture Trade-Off Scatter (Quality vs. Throughput vs. Size)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for idx, m_name in enumerate(order_4):
        info = primary_models[m_name]
        tp = info["throughput_sents_per_sec"]
        chrf = info["chrf"]
        size_m = info["total_params"] / 1e6
        arch = info["architecture"]
        bubble_size = max(120, size_m * 0.65)
        ax.scatter(tp, chrf, s=bubble_size, color=colors_4[idx], alpha=0.82, edgecolors="black", linewidth=1.5, label=f"{m_name} ({arch}, {size_m:.1f}M params)")
        ax.annotate(
            f"{m_name}\n(BLEU={info['bleu']:.1f}, chrF={chrf:.1f})",
            (tp, chrf),
            xytext=(8, -6),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_title("Architecture Trade-Off: Translation Quality (chrF) vs. Throughput", fontsize=12, fontweight="bold")
    ax.set_xlabel("Inference Throughput (sentences / second)", fontsize=10.5)
    ax.set_ylabel("chrF Score (0 - 100)", fontsize=10.5)
    ax.set_xlim(0, max(primary_models[m]["throughput_sents_per_sec"] for m in order_4) * 1.3)
    ax.set_ylim(min(primary_models[m]["chrf"] for m in order_4) * 0.85, max(primary_models[m]["chrf"] for m in order_4) * 1.1)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5)
    plt.tight_layout()
    plt.savefig("results/plots/architecture_tradeoff_scatter.png", bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Plot 6: Responsible AI - Gender Marker Distribution on Neutral Occupations
    # =========================================================================
    fig, ax = plt.subplots(figsize=(9, 5))
    g_stats = ethical_data["gender_bias_summary"]
    masc = [g_stats[m].get("Masculine", 0) for m in order_4]
    fem = [g_stats[m].get("Feminine", 0) for m in order_4]
    mixed = [g_stats[m].get("Mixed (Asymmetric Role Stereotype)", 0) for m in order_4]
    neut = [g_stats[m].get("Neutral / Unclear", 0) for m in order_4]

    x_g = np.arange(len(order_4))
    ax.bar(x_g, masc, width=0.5, label="Masculine Default (e.g., le médecin, l'ingénieur)", color="#2980b9", edgecolor="black")
    ax.bar(x_g, fem, width=0.5, bottom=masc, label="Feminine Default (e.g., l'infirmière)", color="#e84393", edgecolor="black")
    b_mix = np.array(masc) + np.array(fem)
    ax.bar(x_g, mixed, width=0.5, bottom=b_mix, label="Asymmetric Stereotype (e.g., la secrétaire + au directeur)", color="#f39c12", edgecolor="black")
    b_neut = b_mix + np.array(mixed)
    ax.bar(x_g, neut, width=0.5, bottom=b_neut, label="Neutral / Other", color="#95a5a6", edgecolor="black")

    ax.set_title("Responsible AI Audit: Grammatical Gender Assignment on Gender-Neutral English Occupations", fontsize=11, fontweight="bold")
    ax.set_xticks(x_g)
    ax.set_xticklabels(short_names)
    ax.set_ylabel("Number of Occupational Test Cases (out of 4)")
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.legend(frameon=True, fontsize=8.5)
    plt.tight_layout()
    plt.savefig("results/plots/ethical_gender_bias.png", bbox_inches="tight")
    plt.close()

    print("Successfully generated all 6 visual plots in results/plots/")


if __name__ == "__main__":
    generate_all_plots()

