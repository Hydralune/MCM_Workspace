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
        return json.load(f)


def _extract_re_cov(cov_re: Dict) -> Tuple[float, float, float]:
    rows = cov_re.get("rows", [])
    cols = cov_re.get("cols", [])
    vals = cov_re.get("values", [])
    try:
        def get(a: str, b: str, ia: int, ib: int) -> float:
            if a in rows and b in cols:
                return float(vals[rows.index(a)][cols.index(b)])
            return float(vals[ia][ib])
        var_u0 = get("Intercept", "Intercept", 0, 0)
        cov_u0u1 = get("Intercept", "gestational_week", 0, 1)
        var_u1 = get("gestational_week", "gestational_week", 1, 1)
    except Exception:
        var_u0 = float(vals[0][0]) if vals else 0.0
        cov_u0u1, var_u1 = 0.0, 0.0
    return float(var_u0), float(cov_u0u1), float(var_u1)


def _sigma_tot(scale_resid: float, var_u0: float, cov_u0u1: float, var_u1: float, t: float) -> float:
    v = var_u0 + 2.0 * t * cov_u0u1 + (t ** 2) * var_u1
    return float(np.sqrt(max(scale_resid, 0.0) + max(v, 0.0)))


def _mu_pop(params: Dict[str, float], x: Dict[str, float], feature_means: Dict[str, float]) -> float:
    mu = float(params.get("Intercept", params.get("const", 0.0)))
    for k, v in x.items():
        if k in params:
            v_centered = float(v) - float(feature_means.get(k, 0.0))
            mu += float(params.get(k, 0.0)) * v_centered
    return float(mu)


def _group_bmi_age(df: pd.DataFrame, q: int = 3) -> pd.DataFrame:
    df = df.copy()
    df["bmi_q"] = pd.qcut(df["bmi"], q=q, labels=[f"Q{i+1}" for i in range(q)])
    grp = df.groupby("bmi_q")[["bmi", "maternal_age"]].mean(numeric_only=True)
    grp = grp.rename(columns={"bmi": "bmi_mean", "maternal_age": "age_mean"}).reset_index()
    return grp


def _risk_waiting(t: float) -> float:
    if t <= 12:
        return 1.0
    if 13 <= t <= 27:
        return 5.0
    return 20.0


def _cost_failure(t: float) -> float:
    return _risk_waiting(t + 3.0)


def main() -> None:
    project_root, data_path, out_dir = _detect_paths()

    parser = argparse.ArgumentParser(description="问题3：按分组优化最优检测时点")
    parser.add_argument("--model_json", default=os.path.join(out_dir, "q3_model_summary.json"))
    parser.add_argument("--bmi_quantiles", type=int, default=3)
    parser.add_argument("--weeks_from", type=float, default=11.0)
    parser.add_argument("--weeks_to", type=float, default=28.0)
    parser.add_argument("--weeks_step", type=float, default=0.25)
    parser.add_argument("--p_threshold", type=float, default=0.80, help="概率阈值法的P*")
    parser.add_argument(
        "--sigma_mode",
        choices=["population", "residual"],
        default="population",
        help="不确定性使用方式：population=σ_tot(更保守)，residual=σ_ε(个体级)",
    )
    parser.add_argument("--error_multiplier", type=float, default=1.0, help="敏感性：放大sigma_tot")
    parser.add_argument("--mode", choices=["threshold", "risk"], default="threshold")
    parser.add_argument("--output_csv", default=os.path.join(out_dir, "q3_recommendations.csv"))
    args = parser.parse_args()

    model = _load_model_json(args.model_json)
    params = {k: float(v) for k, v in model.get("params", {}).items()}
    scale = float(model.get("scale", 0.0))
    var_u0, cov_u0u1, var_u1 = _extract_re_cov(model.get("cov_re", {}))
    features = set(model.get("features", []))
    feature_means = {k: float(v) for k, v in model.get("feature_means", {}).items()}

    df = pd.read_csv(data_path, encoding="utf-8-sig")
    for c in ["bmi", "maternal_age"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["bmi"]).reset_index(drop=True)
    if "maternal_age" not in df.columns:
        df["maternal_age"] = float(df["maternal_age"].mean()) if "maternal_age" in df.columns else 30.0

    grp = _group_bmi_age(df, q=max(2, args.bmi_quantiles))
    weeks = np.arange(args.weeks_from, args.weeks_to + 1e-9, args.weeks_step)

    recs: List[Dict[str, float]] = []
    for _, g in grp.iterrows():
        group_name = str(g["bmi_q"]) if "bmi_q" in g else "ALL"
        bmi_val = float(g["bmi_mean"]) if "bmi_mean" in g else float(df["bmi"].mean())
        age_val = float(g["age_mean"]) if "age_mean" in g else float(df.get("maternal_age", pd.Series([30.0])).mean())

        def _p_of_t(t: float, sigma_scale: float = 1.0) -> float:
            x: Dict[str, float] = {"gestational_week": float(t), "bmi": bmi_val}
            if "maternal_age" in features:
                x["maternal_age"] = age_val
            mu = _mu_pop(params, x, feature_means)
            if args.sigma_mode == "population":
                sigma = _sigma_tot(scale, var_u0, cov_u0u1, var_u1, t)
            else:
                sigma = float(np.sqrt(max(scale, 0.0)))
            sigma *= float(sigma_scale)
            return 1.0 - float(norm.cdf((4.0 - mu) / max(sigma, 1e-9)))

        t_star = np.nan
        obj_min = np.inf

        if args.mode == "threshold":
            for t in weeks:
                if _p_of_t(t, sigma_scale=args.error_multiplier) >= args.p_threshold:
                    t_star = float(t)
                    break
        else:
            # 风险最小化
            for t in weeks:
                p_fail = 1.0 - _p_of_t(t, sigma_scale=args.error_multiplier)
                total_risk = _risk_waiting(t) + p_fail * _cost_failure(t)
                if total_risk < obj_min:
                    obj_min = total_risk
                    t_star = float(t)

        recs.append({
            "group": group_name,
            "bmi_mean": bmi_val,
            "age_mean": age_val,
            "mode": args.mode,
            "p_threshold": args.p_threshold if args.mode == "threshold" else np.nan,
            "error_multiplier": args.error_multiplier,
            "t_star": t_star,
        })

    out_df = pd.DataFrame(recs)
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print("Saved:", args.output_csv)


if __name__ == "__main__":
    main()


