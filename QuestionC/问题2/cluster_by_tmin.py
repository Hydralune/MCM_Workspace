import os
from typing import Tuple, Dict, List

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


THRESHOLD = 0.04  # 4%


def detect_project_dirs() -> Tuple[str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    return project_dir, script_dir


def compute_tmin(df: pd.DataFrame, threshold: float = THRESHOLD) -> pd.DataFrame:
    """
    Compute earliest gestational week where y_concentration >= threshold, for each subject.
    Also compute representative BMI for subject (mean across visits) for interpretation.
    Returns a DataFrame with columns: subject_id, t_min_week, bmi_subject, censored
    """
    req_cols = ["subject_id", "gestational_week", "y_concentration", "bmi"]
    for c in req_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    # Ensure numeric types
    df = df.copy()
    df["gestational_week"] = pd.to_numeric(df["gestational_week"], errors="coerce")
    df["y_concentration"] = pd.to_numeric(df["y_concentration"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

    # For each subject, sort by gestational week and find first time reaching threshold
    def _first_reach(sub: pd.DataFrame) -> pd.Series:
        sub = sub.sort_values("gestational_week")
        mask = sub["y_concentration"] >= threshold
        t_min = sub.loc[mask, "gestational_week"].min() if mask.any() else np.nan
        bmi_subject = sub["bmi"].mean() if sub["bmi"].notna().any() else np.nan
        censored = 0 if mask.any() else 1
        return pd.Series({"t_min_week": t_min, "bmi_subject": bmi_subject, "censored": censored})

    out = df.groupby("subject_id", as_index=False).apply(_first_reach).reset_index(drop=True)
    return out


def relabel_by_center(labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Map KMeans arbitrary labels to order by ascending cluster center (earlier t_min = 0).
    """
    order = np.argsort(centers.reshape(-1))
    mapping = {int(old): int(new_rank) for new_rank, old in enumerate(order)}
    relabeled = np.vectorize(lambda x: mapping[int(x)])(labels)
    return relabeled


def run_kmeans_on_tmin(
    tmin_df: pd.DataFrame, k_values: List[int] = [2, 3, 4]
) -> Tuple[pd.DataFrame, Dict[int, Dict[str, float]]]:
    """
    Run KMeans on t_min_week (drop NaN). Returns assignments and metrics per K.
    Metrics dict per K: {"sse": float, "silhouette": float or np.nan}
    """
    data = tmin_df.dropna(subset=["t_min_week"]).copy()
    X = data[["t_min_week"]].values
    results = {}
    all_assignments = []

    for k in k_values:
        if len(X) <= k:
            results[k] = {"sse": np.nan, "silhouette": np.nan}
            continue
        km = KMeans(n_clusters=k, n_init=20, random_state=42)
        labels = km.fit_predict(X)
        centers = km.cluster_centers_.reshape(-1)
        labels_ordered = relabel_by_center(labels, km.cluster_centers_)
        sse = float(km.inertia_)
        try:
            sil = float(silhouette_score(X, labels)) if len(np.unique(labels)) > 1 else np.nan
        except Exception:
            sil = np.nan

        results[k] = {"sse": sse, "silhouette": sil}

        tmp = data[["subject_id", "t_min_week", "bmi_subject"]].copy()
        tmp["k"] = k
        tmp["cluster"] = labels_ordered
        tmp["center"] = tmp["cluster"].map({i: centers[idx] for idx, i in enumerate(np.argsort(centers))})
        all_assignments.append(tmp)

    assign_df = pd.concat(all_assignments, axis=0, ignore_index=True) if all_assignments else pd.DataFrame()
    return assign_df, results


def save_plots(assign_df: pd.DataFrame, metrics: Dict[int, Dict[str, float]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Elbow plot (SSE)
    ks = sorted(metrics.keys())
    sse_vals = [metrics[k]["sse"] for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, sse_vals, marker="o")
    plt.xlabel("K")
    plt.ylabel("SSE (Inertia)")
    plt.title("Elbow on t_min_week")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_elbow_tmin.png"), dpi=160)
    plt.close()

    # Silhouette plot
    sil_vals = [metrics[k]["silhouette"] for k in ks]
    plt.figure(figsize=(6, 4))
    plt.plot(ks, sil_vals, marker="o")
    plt.xlabel("K")
    plt.ylabel("Silhouette score")
    plt.title("Silhouette on t_min_week")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fig_silhouette_tmin.png"), dpi=160)
    plt.close()

    # Boxplot per K
    if not assign_df.empty:
        for k in sorted(assign_df["k"].unique()):
            sub = assign_df[assign_df["k"] == k]
            plt.figure(figsize=(7, 4))
            sns.boxplot(data=sub, x="cluster", y="t_min_week")
            sns.stripplot(data=sub, x="cluster", y="t_min_week", color="black", size=2, alpha=0.4)
            plt.xlabel("Cluster (ordered by center)")
            plt.ylabel("t_min_week")
            plt.title(f"t_min_week by cluster (K={k})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"fig_box_tmin_by_cluster_k{k}.png"), dpi=160)
            plt.close()

            # BMI by cluster for interpretation
            plt.figure(figsize=(7, 4))
            sns.boxplot(data=sub, x="cluster", y="bmi_subject")
            sns.stripplot(data=sub, x="cluster", y="bmi_subject", color="black", size=2, alpha=0.4)
            plt.xlabel("Cluster (ordered by center)")
            plt.ylabel("BMI (subject mean)")
            plt.title(f"BMI by cluster (K={k})")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"fig_box_bmi_by_cluster_k{k}.png"), dpi=160)
            plt.close()


def save_tables(tmin_df: pd.DataFrame, assign_df: pd.DataFrame, metrics: Dict[int, Dict[str, float]], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    tmin_df.to_csv(os.path.join(out_dir, "tmin_per_subject.csv"), index=False, encoding="utf-8-sig")
    if not assign_df.empty:
        assign_df.to_csv(os.path.join(out_dir, "kmeans_assignments_tmin.csv"), index=False, encoding="utf-8-sig")

        # Summary per K, per cluster
        summaries = []
        for k in sorted(assign_df["k"].unique()):
            sub = assign_df[assign_df["k"] == k]
            g = sub.groupby("cluster")["t_min_week"]
            summ = g.agg(n="count", mean="mean", std="std", q25=lambda s: s.quantile(0.25), median="median", q75=lambda s: s.quantile(0.75)).reset_index()
            summ["k"] = k
            summaries.append(summ)
        summary_df = pd.concat(summaries, ignore_index=True)
        summary_df.to_csv(os.path.join(out_dir, "kmeans_summary_tmin.csv"), index=False, encoding="utf-8-sig")

    # Metrics
    met_df = pd.DataFrame([{"k": k, **v} for k, v in metrics.items()])
    met_df.to_csv(os.path.join(out_dir, "kmeans_metrics_tmin.csv"), index=False, encoding="utf-8-sig")


def main():
    project_dir, script_dir = detect_project_dirs()

    # Source data: male_cleaned.csv (问题2仅男胎)
    src_csv = os.path.join(project_dir, "数据预处理", "male_cleaned.csv")
    out_dir = script_dir

    df = pd.read_csv(src_csv, encoding="utf-8-sig")
    tmin_df = compute_tmin(df, threshold=THRESHOLD)

    # Save tmin
    os.makedirs(out_dir, exist_ok=True)
    tmin_path = os.path.join(out_dir, "tmin_per_subject.csv")
    tmin_df.to_csv(tmin_path, index=False, encoding="utf-8-sig")

    # Run KMeans for K=2..4 (on non-NaN t_min)
    assign_df, metrics = run_kmeans_on_tmin(tmin_df, k_values=[2, 3, 4])

    # Save plots and tables
    save_plots(assign_df, metrics, out_dir)
    save_tables(tmin_df, assign_df, metrics, out_dir)

    # Console summary
    n_total = len(tmin_df)
    n_reached = tmin_df["t_min_week"].notna().sum()
    print(f"Subjects: total={n_total}, reached>=4%={n_reached}, censored={n_total - n_reached}")
    for k in sorted(metrics.keys()):
        print(f"K={k}: SSE={metrics[k]['sse']}, silhouette={metrics[k]['silhouette']}")


if __name__ == "__main__":
    main()



