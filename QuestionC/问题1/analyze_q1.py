import os
from typing import Tuple

import pandas as pd
import numpy as np

# Use non-interactive backend for environments without display
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def get_project_paths() -> Tuple[str, str, str]:
    """
    Return (project_root, data_path, output_dir)
    优先使用 数据预处理/male_cleaned.csv；若不存在则回退到 问题1/q1_male_cleaned.csv。
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, os.pardir))
    preferred_path = os.path.join(project_root, "数据预处理", "male_cleaned.csv")
    fallback_path = os.path.join(current_dir, "q1_male_cleaned.csv")
    data_path = preferred_path if os.path.exists(preferred_path) else fallback_path
    output_dir = os.path.join(current_dir, "figures")
    os.makedirs(output_dir, exist_ok=True)
    return project_root, data_path, output_dir


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Ensure required columns exist
    required = ["gestational_week", "bmi", "y_concentration"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in cleaned data: {missing}")
    # Coerce numeric
    for c in required:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)
    # Prepare y for plotting: convert to percentage scale if data is in proportion
    y = df["y_concentration"].astype(float)
    if np.nanmax(y.values) <= 1.0 + 1e-9:
        df["y_plot"] = y * 100.0
    else:
        df["y_plot"] = y
    return df


def add_bmi_group(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 20, 28, 32, 36, 40, np.inf]
    labels = ["<20", "[20,28)", "[28,32)", "[32,36)", "[36,40)", "40+"]
    df = df.copy()
    df["bmi_group"] = pd.cut(df["bmi"], bins=bins, labels=labels, right=False, include_lowest=True)
    return df


def _set_ylim_quantile(ax: plt.Axes, values: pd.Series) -> None:
    v = values.dropna().values
    if v.size == 0:
        return
    q_low = np.nanpercentile(v, 1)
    q_high = np.nanpercentile(v, 99)
    span = max(q_high - q_low, 1e-6)
    ymin = max(0.0, q_low - 0.05 * span)
    ymax = q_high + 0.05 * span
    if ymax <= ymin:
        ymax = ymin + 1.0
    ax.set_ylim(ymin, ymax)


def _maybe_draw_threshold(ax: plt.Axes, ymin: float, ymax: float, threshold_pct: float = 4.0) -> None:
    if ymin <= threshold_pct <= ymax:
        ax.axhline(threshold_pct, color="red", linestyle="--", linewidth=1.2, label="4% threshold")
    else:
        # Annotate if out of view
        direction = "top" if threshold_pct > ymax else "bottom"
        y_ann = ymax if threshold_pct > ymax else ymin
        ax.annotate(
            "4% threshold",
            xy=(0.99, 0.98 if direction == "top" else 0.02),
            xycoords="axes fraction",
            ha="right",
            va="top" if direction == "top" else "bottom",
            color="red",
            fontsize=9,
        )


def plot_scatter_week_vs_y(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(7.5, 5.5))
    sns.scatterplot(
        data=df,
        x="gestational_week",
        y="y_plot",
        hue="bmi_group",
        palette="viridis",
        alpha=0.7,
        edgecolor=None,
    )
    plt.xlabel("Gestational Week")
    plt.ylabel("Y Concentration (%)")
    plt.title("Y Concentration vs Gestational Week (colored by BMI group)")
    plt.legend(title="BMI Group", fontsize=8)
    ax = plt.gca()
    _set_ylim_quantile(ax, df["y_plot"])
    ymin, ymax = ax.get_ylim()
    _maybe_draw_threshold(ax, ymin, ymax, 4.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_week_vs_y.png"), dpi=180)
    plt.close()


def plot_scatter_bmi_vs_y(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(7.5, 5.5))
    sns.scatterplot(
        data=df,
        x="bmi",
        y="y_plot",
        hue="gestational_week",
        palette="coolwarm",
        alpha=0.7,
        edgecolor=None,
    )
    plt.xlabel("BMI")
    plt.ylabel("Y Concentration (%)")
    plt.title("Y Concentration vs BMI (colored by Gestational Week)")
    plt.legend(title="Gestational Week", fontsize=8)
    ax = plt.gca()
    _set_ylim_quantile(ax, df["y_plot"])
    ymin, ymax = ax.get_ylim()
    _maybe_draw_threshold(ax, ymin, ymax, 4.0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "scatter_bmi_vs_y.png"), dpi=180)
    plt.close()


def plot_distributions(df: pd.DataFrame, out_dir: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    sns.histplot(df["gestational_week"], kde=True, ax=axes[0], color="#4C72B0")
    axes[0].set_title("Gestational Week distribution")
    sns.histplot(df["bmi"], kde=True, ax=axes[1], color="#55A868")
    axes[1].set_title("BMI distribution")
    sns.histplot(df["y_plot"], kde=True, ax=axes[2], color="#C44E52")
    # Threshold line only if in view
    x_vals = df["y_plot"].dropna().values
    if x_vals.size:
        q_low = np.nanpercentile(x_vals, 1)
        q_high = np.nanpercentile(x_vals, 99)
        span = max(q_high - q_low, 1e-6)
        xmin = max(0.0, q_low - 0.05 * span)
        xmax = q_high + 0.05 * span
        axes[2].set_xlim(xmin, xmax)
        if xmin <= 4.0 <= xmax:
            axes[2].axvline(4.0, color="red", linestyle="--", linewidth=1.2)
        else:
            axes[2].annotate(
                "4% threshold",
                xy=(0.98, 0.9),
                xycoords="axes fraction",
                ha="right",
                va="top",
                color="red",
                fontsize=9,
            )
    axes[2].set_title("Y Concentration distribution")
    for ax in axes:
        ax.grid(alpha=0.15)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "distributions.png"), dpi=180)
    plt.close(fig)


def plot_box_by_bmi_group(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(8, 4.5))
    sns.boxplot(data=df, x="bmi_group", y="y_plot", palette="pastel")
    ax = plt.gca()
    _set_ylim_quantile(ax, df["y_plot"])
    ymin, ymax = ax.get_ylim()
    _maybe_draw_threshold(ax, ymin, ymax, 4.0)
    plt.xlabel("BMI Group")
    plt.ylabel("Y Concentration (%)")
    plt.title("Y Concentration by BMI Group")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "box_y_by_bmi_group.png"), dpi=180)
    plt.close()


def compute_correlations_basic(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["gestational_week", "bmi", "y_concentration"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr(method="pearson")
    return corr


def _select_extended_numeric_columns(df: pd.DataFrame) -> list:
    """选择用于扩展相关性的数值列（若存在）。"""
    candidates = [
        "gestational_week",
        "bmi",
        "y_concentration",
        "y_zscore",
        "maternal_age",
        "maternal_height_cm",
        "maternal_weight_kg",
        "bmi_recalc",
        "ivf_flag",
        # 质控/测序相关
        "L", "M", "N", "O", "P", "X", "Y", "Z", "AA",
        # 其他可能存在的辅助列
        "draw_index",
    ]
    cols = [c for c in candidates if c in df.columns]
    return cols


def compute_correlations_extended(df: pd.DataFrame) -> pd.DataFrame:
    cols = _select_extended_numeric_columns(df)
    if not cols:
        return pd.DataFrame()
    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    return corr


def save_corr_heatmap(corr: pd.DataFrame, out_path_png: str, title: str, figsize=(4.2, 3.6)) -> None:
    if corr.empty:
        return
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1, square=True, cbar=True)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=180)
    plt.close()


def write_summary_markdown(df: pd.DataFrame, corr_basic: pd.DataFrame, corr_ext: pd.DataFrame, out_path: str) -> None:
    desc = df[["gestational_week", "bmi", "y_concentration"]].describe().T
    lines = []
    lines.append("# 问题1 EDA 摘要\n")
    lines.append("## 样本规模\n")
    lines.append(f"记录数: {len(df)}\n")
    lines.append("\n## 描述性统计（关键变量）\n")
    for idx, row in desc.iterrows():
        lines.append(f"- {idx}: count={int(row['count'])}, mean={row['mean']:.3f}, std={row['std']:.3f}, min={row['min']:.3f}, 25%={row['25%']:.3f}, 50%={row['50%']:.3f}, 75%={row['75%']:.3f}, max={row['max']:.3f}")
    lines.append("\n## 基础相关性 (Pearson)\n")
    for r in corr_basic.index:
        vals = ", ".join([f"{c}={corr_basic.loc[r, c]:.3f}" for c in corr_basic.columns])
        lines.append(f"- {r}: {vals}")
    if not corr_ext.empty and "y_concentration" in corr_ext.columns:
        lines.append("\n## 扩展相关性与Top相关（对 y_concentration）\n")
        s = corr_ext["y_concentration"].drop("y_concentration", errors="ignore")
        s = s.sort_values(key=lambda x: x.abs(), ascending=False)
        topk = s.head(10)
        for name, val in topk.items():
            lines.append(f"- {name}: r={val:.3f}")
        lines.append("\n详见: figures/corr_matrix_extended.csv 与 figures/corr_heatmap_extended.png\n")
    lines.append("\n## 生成图表\n")
    lines.append("- figures/distributions.png\n- figures/scatter_week_vs_y.png\n- figures/scatter_bmi_vs_y.png\n- figures/box_y_by_bmi_group.png\n- figures/corr_heatmap_basic.png\n- figures/corr_heatmap_extended.png\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    _, data_path, out_dir = get_project_paths()
    df = load_data(data_path)
    df = add_bmi_group(df)

    # Plots
    plot_distributions(df, out_dir)
    plot_scatter_week_vs_y(df, out_dir)
    plot_scatter_bmi_vs_y(df, out_dir)
    plot_box_by_bmi_group(df, out_dir)

    # Correlations and summary
    corr_basic = compute_correlations_basic(df)
    save_corr_heatmap(corr_basic, os.path.join(out_dir, "corr_heatmap_basic.png"), "Pearson Correlations (basic)")

    corr_ext = compute_correlations_extended(df)
    # 保存扩展相关矩阵 CSV
    if not corr_ext.empty:
        corr_csv = os.path.join(out_dir, "corr_matrix_extended.csv")
        corr_ext.to_csv(corr_csv, encoding="utf-8-sig")
        # 根据维度自适应图尺寸
        n = corr_ext.shape[0]
        fig_w = max(4.5, min(1.0 * n, 18.0))
        fig_h = max(4.0, min(1.0 * n, 18.0))
        save_corr_heatmap(corr_ext, os.path.join(out_dir, "corr_heatmap_extended.png"), "Pearson Correlations (extended)", figsize=(fig_w, fig_h))

    summary_md = os.path.join(os.path.dirname(data_path), "q1_eda_summary.md")
    write_summary_markdown(df, corr_basic, corr_ext, summary_md)
    print(f"EDA 完成。图表输出目录: {out_dir}")
    print(f"摘要: {summary_md}")


if __name__ == "__main__":
    main()


