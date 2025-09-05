import os
import pandas as pd


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    assign_csv = os.path.join(base, "kmeans_assignments_tmin.csv")
    out_csv = os.path.join(base, "bmi_group_recommendations.csv")

    df = pd.read_csv(assign_csv, encoding="utf-8-sig")
    df = df[df["k"] == 3].copy()

    # 计算每簇 BMI 和 t_min 的统计量
    grp = df.groupby("cluster")
    bmi_stats = grp["bmi_subject"].median().rename("bmi_median").to_frame()
    t_stats = grp["t_min_week"].agg(
        t_q25=lambda s: s.quantile(0.25),
        t_median="median",
        t_q75=lambda s: s.quantile(0.75),
        n="count",
    )
    stats = bmi_stats.join(t_stats)

    # 按 t_min 中位数从早到晚排序（更符合临床命名）
    stats = stats.sort_values("t_median").reset_index()
    stats["rank"] = range(len(stats))

    # 以相邻 BMI 中位数的中点作为分组阈值
    bmi_meds = stats["bmi_median"].values
    cut_points = []
    for a, b in zip(bmi_meds[:-1], bmi_meds[1:]):
        cut_points.append((a + b) / 2.0)

    # 构造区间
    bounds = []
    labels = []
    for i in range(len(stats)):
        if i == 0:
            lo, hi = float("-inf"), cut_points[0]
            labels.append("早达标群")
        elif i == len(stats) - 1:
            lo, hi = cut_points[-1], float("inf")
            labels.append("晚达标群")
        else:
            lo, hi = cut_points[i - 1], cut_points[i]
            labels.append("中间群")
        bounds.append((lo, hi))

    # 生成建议表
    rows = []
    for i, row in stats.iterrows():
        lo, hi = bounds[i]
        rows.append({
            "group": labels[i],
            "bmi_min": None if lo == float("-inf") else round(lo, 3),
            "bmi_max": None if hi == float("inf") else round(hi, 3),
            "t_min_q25": round(row["t_q25"], 2),
            "t_min_median": round(row["t_median"], 2),
            "t_min_q75": round(row["t_q75"], 2),
            "n": int(row["n"]),
        })

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print("Saved:", out_csv)
    print(out)


if __name__ == "__main__":
    main()



