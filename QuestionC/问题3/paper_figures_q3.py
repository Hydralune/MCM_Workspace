import os
import argparse
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _detect_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = script_dir
    return out_dir


def plot_prob_curves(pred_csv: str, out_png: str, threshold_lines=(0.8, 0.9)) -> None:
    df = pd.read_csv(pred_csv, encoding="utf-8-sig")
    # 中文字体与负号显示设置
    try:
        import matplotlib as _mpl
        _mpl.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "DejaVu Sans",
        ]
        _mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    plt.figure(figsize=(7.5, 5.0))
    groups = list(df["group"].unique())
    groups.sort()
    for g in groups:
        sub = df[df["group"] == g].copy()
        sub = sub.sort_values("week")
        plt.plot(sub["week"], sub["p_ge_4"], label=str(g))
    for thr in threshold_lines:
        plt.axhline(float(thr), color="gray", linestyle=":", linewidth=1.0)
        plt.text(27.8, float(thr) + 0.01, f"P={float(thr):.1f}", ha="right", va="bottom", fontsize=8, color="gray")
    plt.axhline(0.0, color="#dddddd", linewidth=0.8)
    plt.axhline(1.0, color="#dddddd", linewidth=0.8)
    plt.xlabel("孕周 (周)")
    plt.ylabel("P(Y≥4%)")
    plt.ylim(0, 1)
    plt.title("不同BMI组达到4%的概率曲线 (问题3)")
    plt.legend(fontsize=8, title="BMI分位组")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def plot_tstar_bar(rec_csv: str, out_png: str) -> None:
    df = pd.read_csv(rec_csv, encoding="utf-8-sig")
    # 中文字体与负号显示设置
    try:
        import matplotlib as _mpl
        _mpl.rcParams["font.sans-serif"] = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "Noto Sans CJK JP",
            "DejaVu Sans",
        ]
        _mpl.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    df = df.sort_values("group")
    groups = df["group"].astype(str).tolist()
    vals = pd.to_numeric(df["t_star"], errors="coerce").values

    # 设定合理y轴范围（若无有效值，使用临床常见范围）
    finite_vals = vals[np.isfinite(vals)]
    y_min = 10.0
    y_max = float(np.nanmax(finite_vals)) + 2.0 if finite_vals.size else 35.0
    y_max = max(y_max, 20.0)

    plt.figure(figsize=(6.5, 4.2))
    # 用0占位绘制柱，随后用文本标注实际数值/未达标
    heights = np.nan_to_num(vals, nan=0.0)
    bars = plt.bar(groups, heights)
    plt.xlabel("BMI分位组")
    plt.ylabel("最优/最早推荐孕周 (t*)")
    plt.title("各BMI组推荐时点 (问题3)")
    plt.ylim(y_min, y_max)

    for i, (g, y) in enumerate(zip(groups, vals)):
        if np.isfinite(y):
            plt.text(i, float(y) + 0.2, f"{float(y):.1f}", ha="center", va="bottom", fontsize=9)
        else:
            # 未达标：置顶标注
            plt.text(i, y_min + 0.15 * (y_max - y_min), "未达标", ha="center", va="bottom", fontsize=9, color="gray")

    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main() -> None:
    out_dir = _detect_paths()
    parser = argparse.ArgumentParser(description="问题3：论文图表输出")
    parser.add_argument("--pred_csv", default=os.path.join(out_dir, "q3_pred_prob_by_week.csv"))
    parser.add_argument("--rec_csv", default=os.path.join(out_dir, "q3_recommendations.csv"))
    parser.add_argument("--out_prob_png", default=os.path.join(out_dir, "paper_q3_prob_curves.png"))
    parser.add_argument("--out_rec_png", default=os.path.join(out_dir, "paper_q3_tstar_bar.png"))
    args = parser.parse_args()

    if os.path.exists(args.pred_csv):
        plot_prob_curves(args.pred_csv, args.out_prob_png)
        print("Saved:", args.out_prob_png)
    else:
        print("Skip prob curves: pred_csv not found:", args.pred_csv)

    if os.path.exists(args.rec_csv):
        plot_tstar_bar(args.rec_csv, args.out_rec_png)
        print("Saved:", args.out_rec_png)
    else:
        print("Skip t* bar: rec_csv not found:", args.rec_csv)


if __name__ == "__main__":
    main()


