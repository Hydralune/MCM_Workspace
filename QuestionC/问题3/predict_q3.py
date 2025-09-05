import os
import json
import argparse
from typing import Dict, Tuple, List

import pandas as pd
import numpy as np
from scipy.stats import norm


def _detect_paths() -> Tuple[str, str, str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    data_path = os.path.join(project_root, "数据预处理", "male_cleaned.csv")
    out_dir = script_dir
    return project_root, data_path, out_dir


def _load_model_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj


def _extract_re_cov(cov_re: Dict) -> Tuple[float, float, float]:
    rows = cov_re.get("rows", [])
    cols = cov_re.get("cols", [])
    vals = cov_re.get("values", [])
    try:
        # 统一从名字中提取
        def get(a: str, b: str, ia: int, ib: int) -> float:
            if a in rows and b in cols:
                return float(vals[rows.index(a)][cols.index(b)])
            # 后备：按索引位置
            return float(vals[ia][ib])

        var_u0 = get("Intercept", "Intercept", 0, 0)
        cov_u0u1 = get("Intercept", "gestational_week", 0, 1)
        var_u1 = get("gestational_week", "gestational_week", 1, 1)
    except Exception:
        # 至少返回随机截距方差
        try:
            var_u0 = float(vals[0][0])
        except Exception:
            var_u0 = 0.0
        cov_u0u1, var_u1 = 0.0, 0.0
    return float(var_u0), float(cov_u0u1), float(var_u1)


def _sigma_tot(scale_resid: float, var_u0: float, cov_u0u1: float, var_u1: float, t: float) -> float:
    v = var_u0 + 2.0 * t * cov_u0u1 + (t ** 2) * var_u1
    return float(np.sqrt(max(scale_resid, 0.0) + max(v, 0.0)))


def _mu_pop(params: Dict[str, float], row: Dict[str, float], feature_means: Dict[str, float]) -> float:
    mu = 0.0
    # Intercept 名称可能为 Intercept 或 const
    mu += float(params.get("Intercept", params.get("const", 0.0)))
    for k, v in row.items():
        if k in params:
            v_centered = float(v) - float(feature_means.get(k, 0.0))
            mu += float(params[k]) * v_centered
    return float(mu)


def _group_bmi_age(df: pd.DataFrame, q: int = 4) -> pd.DataFrame:
    df = df.copy()
    df["bmi_q"] = pd.qcut(df["bmi"], q=q, labels=[f"Q{i+1}" for i in range(q)])
    grp = df.groupby("bmi_q")[["bmi", "maternal_age"]].mean(numeric_only=True)
    grp = grp.rename(columns={"bmi": "bmi_mean", "maternal_age": "age_mean"})
    grp = grp.reset_index()
    return grp


def main() -> None:
    project_root, data_path, out_dir = _detect_paths()

    parser = argparse.ArgumentParser(description="问题3：基于模型计算 P(Y>=4%) 曲线")
    parser.add_argument("--model_json", default=os.path.join(out_dir, "q3_model_summary.json"), help="模型摘要JSON路径")
    parser.add_argument("--weeks_from", type=float, default=10.0)
    parser.add_argument("--weeks_to", type=float, default=28.0)
    parser.add_argument("--weeks_step", type=float, default=0.25)
    parser.add_argument("--bmi_quantiles", type=int, default=4, help="BMI分位分组数量")
    parser.add_argument("--output_csv", default=os.path.join(out_dir, "q3_pred_prob_by_week.csv"))
    args = parser.parse_args()

    # 载入模型
    model = _load_model_json(args.model_json)
    params = {k: float(v) for k, v in model.get("params", {}).items()}
    scale = float(model.get("scale", 0.0))
    var_u0, cov_u0u1, var_u1 = _extract_re_cov(model.get("cov_re", {}))
    features = set(model.get("features", []))
    feature_means = {k: float(v) for k, v in model.get("feature_means", {}).items()}

    # 载入数据，用于估计各组代表 BMI/年龄
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    for c in ["gestational_week", "bmi", "maternal_age"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["bmi"]).reset_index(drop=True)
    if "maternal_age" not in df.columns:
        df["maternal_age"] = float(df["maternal_age"].mean()) if "maternal_age" in df.columns else 30.0

    grp_features = _group_bmi_age(df, q=max(2, args.bmi_quantiles))

    # 计算概率曲线（群体级：使用 sigma_tot）
    weeks = np.arange(args.weeks_from, args.weeks_to + 1e-9, args.weeks_step)
    rows: List[Dict[str, float]] = []
    for _, g in grp_features.iterrows():
        group_name = str(g["bmi_q"]) if "bmi_q" in g else "ALL"
        bmi_val = float(g["bmi_mean"]) if "bmi_mean" in g else float(df["bmi"].mean())
        age_val = float(g["age_mean"]) if "age_mean" in g else float(df.get("maternal_age", pd.Series([30.0])).mean())
        for t in weeks:
            x: Dict[str, float] = {"gestational_week": float(t), "bmi": bmi_val}
            if "maternal_age" in features:
                x["maternal_age"] = age_val
            mu = _mu_pop(params, x, feature_means)
            sigma = _sigma_tot(scale, var_u0, cov_u0u1, var_u1, t)
            p = 1.0 - float(norm.cdf((4.0 - mu) / max(sigma, 1e-9)))
            rows.append({
                "group": group_name,
                "week": float(t),
                "bmi_mean": bmi_val,
                "age_mean": age_val,
                "mu": mu,
                "sigma_tot": sigma,
                "p_ge_4": p,
            })

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print("Saved:", args.output_csv)


if __name__ == "__main__":
    main()


