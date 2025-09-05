import os
from typing import Tuple

import pandas as pd
import numpy as np
import matplotlib
from matplotlib import font_manager
import matplotlib.pyplot as plt
import seaborn as sns

# 全局中文字体与负号设置
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False
matplotlib.rcParams["pdf.fonttype"] = 42  # embed TrueType fonts in PDF
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["svg.fonttype"] = "none"

def _setup_chinese_font() -> str:
    """Register a workable Chinese font and return its family name."""
    # Candidate font files (Windows) and generic names
    candidate_paths = [
        r"C:\\Windows\\Fonts\\msyh.ttc",   # Microsoft YaHei
        r"C:\\Windows\\Fonts\\msyhbd.ttc",
        r"C:\\Windows\\Fonts\\simhei.ttf", # SimHei
        r"C:\\Windows\\Fonts\\simsun.ttc", # SimSun
        r"C:\\Windows\\Fonts\\Deng.ttf",   # DengXian
    ]
    candidate_names = [
        "Microsoft YaHei", "SimHei", "SimSun", "DengXian",
        "Noto Sans CJK SC", "Source Han Sans CN",
    ]

    chosen_name = None
    # Try to register font files if present
    for p in candidate_paths:
        try:
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
        except Exception:
            pass

    # After registration, check available families
    available = set(f.name for f in font_manager.fontManager.ttflist)
    for name in candidate_names:
        if name in available:
            chosen_name = name
            break

    # Fallback to DejaVu Sans (ASCII only) if none found
    if not chosen_name:
        chosen_name = "DejaVu Sans"

    # Set as default
    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        chosen_name,
        "Microsoft YaHei", "SimHei", "SimSun", "DengXian",
        "Noto Sans CJK SC", "Source Han Sans CN",
        "DejaVu Sans",
    ]
    return chosen_name

def detect_dirs() -> Tuple[str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    return project_dir, script_dir


def setup_style():
    # 统一论文风格
    chosen = _setup_chinese_font()
    sns.set_theme(style="whitegrid", context="talk", font=chosen)
    plt.rcParams.update({
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "axes.titlesize": 16,
        "axes.labelsize": 14,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })


def fig_elbow_and_silhouette(metrics_csv: str, out_dir: str):
    met = pd.read_csv(metrics_csv)
    ks = met["k"].values
    sse = met["sse"].values
    sil = met["silhouette"].values

    # Elbow
    plt.figure(figsize=(6, 4))
    plt.plot(ks, sse, marker="o")
    plt.xlabel("K")
    plt.ylabel("组内平方和 (SSE)")
    plt.title("肘部法则 (达标孕周)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paper_elbow_tmin.png"))
    plt.savefig(os.path.join(out_dir, "paper_elbow_tmin.pdf"))
    plt.close()

    # Silhouette
    plt.figure(figsize=(6, 4))
    plt.plot(ks, sil, marker="o")
    plt.xlabel("K")
    plt.ylabel("轮廓系数")
    plt.title("轮廓系数 (达标孕周)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "paper_silhouette_tmin.png"))
    plt.savefig(os.path.join(out_dir, "paper_silhouette_tmin.pdf"))
    plt.close()


def fig_tmin_violin(assign_csv: str, out_dir: str, k: int = 3):
    df = pd.read_csv(assign_csv)
    sub = df[df["k"] == k].copy()
    if sub.empty:
        return
    plt.figure(figsize=(7.5, 4.5))
    sns.violinplot(data=sub, x="cluster", y="t_min_week", inner=None, cut=0)
    sns.boxplot(data=sub, x="cluster", y="t_min_week", width=0.18, showcaps=True, boxprops={'facecolor':'white'}, showfliers=False)
    sns.stripplot(data=sub, x="cluster", y="t_min_week", color="black", size=2, alpha=0.3)
    plt.xlabel("聚类（按中心早-晚排序）")
    plt.ylabel("达标孕周 t_min (周)")
    plt.title(f"达标孕周分布（K={k}）")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"paper_tmin_violin_k{k}.png"))
    plt.savefig(os.path.join(out_dir, f"paper_tmin_violin_k{k}.pdf"))
    plt.close()


def fig_bmi_box(assign_csv: str, out_dir: str, k: int = 3):
    df = pd.read_csv(assign_csv)
    sub = df[df["k"] == k].copy()
    if sub.empty:
        return
    plt.figure(figsize=(7.5, 4.5))
    sns.boxplot(data=sub, x="cluster", y="bmi_subject")
    sns.stripplot(data=sub, x="cluster", y="bmi_subject", color="black", size=2, alpha=0.3)
    plt.xlabel("聚类（按中心早-晚排序）")
    plt.ylabel("BMI（受试者均值）")
    plt.title(f"BMI按聚类分布（K={k}）")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"paper_bmi_box_k{k}.png"))
    plt.savefig(os.path.join(out_dir, f"paper_bmi_box_k{k}.pdf"))
    plt.close()


def fig_scatter_tmin_bmi(assign_csv: str, out_dir: str, k: int = 3):
    df = pd.read_csv(assign_csv)
    sub = df[df["k"] == k].copy()
    if sub.empty:
        return
    plt.figure(figsize=(6.2, 4.6))
    sns.scatterplot(data=sub, x="bmi_subject", y="t_min_week", hue="cluster", palette="viridis", s=32, alpha=0.9)
    plt.xlabel("BMI（受试者均值）")
    plt.ylabel("达标孕周 t_min (周)")
    plt.title(f"达标孕周 vs BMI（K={k}）")
    plt.legend(title="簇")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"paper_scatter_tmin_bmi_k{k}.png"))
    plt.savefig(os.path.join(out_dir, f"paper_scatter_tmin_bmi_k{k}.pdf"))
    plt.close()


def main():
    project_dir, script_dir = detect_dirs()
    setup_style()

    metrics_csv = os.path.join(script_dir, "kmeans_metrics_tmin.csv")
    assign_csv = os.path.join(script_dir, "kmeans_assignments_tmin.csv")

    fig_elbow_and_silhouette(metrics_csv, script_dir)
    for k in (3, 2, 4):
        fig_tmin_violin(assign_csv, script_dir, k=k)
        fig_bmi_box(assign_csv, script_dir, k=k)
        fig_scatter_tmin_bmi(assign_csv, script_dir, k=k)

    print("Saved paper-ready figures to:", script_dir)


if __name__ == "__main__":
    main()


