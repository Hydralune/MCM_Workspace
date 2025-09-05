import os
import json
from typing import Tuple, List, Dict

import pandas as pd
import numpy as np
import statsmodels.formula.api as smf


def _detect_paths() -> Tuple[str, str, str]:
    """
    Returns (project_root, data_path, out_dir)
    - data_path 优先使用 数据预处理/male_cleaned.csv
    - out_dir 为 当前脚本目录（问题3）
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
    preferred = os.path.join(project_root, "数据预处理", "male_cleaned.csv")
    data_path = preferred
    out_dir = script_dir
    return project_root, data_path, out_dir


def _load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    # 关键列存在性与类型
    req = [
        "subject_id",
        "gestational_week",
        "bmi",
        "y_concentration",
    ]
    miss = [c for c in req if c not in df.columns]
    if miss:
        raise ValueError(f"缺少必要列: {miss}")
    # 数值化
    for c in ["gestational_week", "bmi", "y_concentration"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 可选协变量
    optional_numeric = [
        "maternal_age",
        "maternal_height_cm",
        "maternal_weight_kg",
        # QC / 测序相关可选列（按数据存在与否自动加入）
        "L", "M", "N", "O", "P", "X", "Y", "Z", "AA",
    ]
    for c in optional_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["subject_id", "gestational_week", "bmi", "y_concentration"]).reset_index(drop=True)
    return df


def _maybe_convert_to_percent(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    df = df.copy()
    y = df["y_concentration"].astype(float)
    is_prop = np.nanmax(y.values) <= 1.0 + 1e-9
    df["y_target"] = y * 100.0 if is_prop else y
    return df, is_prop


def _choose_features(df: pd.DataFrame) -> List[str]:
    """
    选择用于固定效应的特征：
    - 基础必备：gestational_week, bmi
    - 优先加入：maternal_age
    - 若存在则加入：L, M, N（其余QC列按需可加，这里保守选常见三项）
    """
    features: List[str] = ["gestational_week", "bmi"]
    if "maternal_age" in df.columns:
        features.append("maternal_age")
    for qc in ["L", "M", "N"]:
        if qc in df.columns:
            features.append(qc)
    return features


def _fit_mixedlm_random_slope(df: pd.DataFrame, features: List[str]):
    if "gestational_week" not in features:
        raise ValueError("features 必须包含 gestational_week")
    formula = "y_target ~ " + " + ".join(features)
    md = smf.mixedlm(
        formula,
        data=df,
        groups=df["subject_id"],
        re_formula="~ gestational_week",
    )
    try:
        res = md.fit(reml=False, method="lbfgs")
    except Exception:
        res = md.fit(reml=False, method="nm")
    return res


def _export_model(res, features: List[str], out_dir: str, feature_means: Dict[str, float]) -> Dict[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    # 仅导出预测所需的最小信息，避免复杂对象序列化问题
    params = {k: float(v) for k, v in res.params.items()}
    try:
        cov_re = res.cov_re.copy()
        row_names = list(cov_re.index)
        col_names = list(cov_re.columns)
        cov_vals = cov_re.values.tolist()
    except Exception:
        # 部分情况下可能没有随机斜率，降级为仅截距
        row_names, col_names, cov_vals = ["Intercept"], ["Intercept"], [[float(getattr(res, "cov_re", 0.0))]]

    payload = {
        "features": features,
        "feature_means": {k: float(v) for k, v in feature_means.items()},
        "params": params,
        "cov_re": {
            "rows": row_names,
            "cols": col_names,
            "values": cov_vals,
        },
        "scale": float(getattr(res, "scale", np.nan)),  # 残差方差
        "aic": float(getattr(res, "aic", np.nan)),
        "bic": float(getattr(res, "bic", np.nan)),
        "llf": float(getattr(res, "llf", np.nan)),
        "model": "MixedLM_random_slope",
    }

    json_path = os.path.join(out_dir, "q3_model_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # 文本摘要
    md_path = os.path.join(out_dir, "q3_model_summary.md")
    lines = []
    lines.append("# 问题3 模型摘要\n")
    lines.append("## 固定效应与参数\n")
    for k, v in params.items():
        lines.append(f"- {k}: {v:+.6f}")
    lines.append("\n## 随机效应方差-协方差 (u0,u_week)\n")
    try:
        for i, r in enumerate(row_names):
            row_vals = cov_vals[i]
            lines.append("- " + r + ": " + ", ".join([f"{col_names[j]}={row_vals[j]:+.6f}" for j in range(len(col_names))]))
    except Exception:
        pass
    lines.append(f"\n## 残差方差 scale: {payload['scale']:.6f}\n")
    lines.append(f"AIC: {payload['aic']:.3f}, BIC: {payload['bic']:.3f}, llf: {payload['llf']:.3f}\n")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"json": json_path, "markdown": md_path}


def main() -> None:
    _, data_path, out_dir = _detect_paths()
    df = _load_data(data_path)
    df, _ = _maybe_convert_to_percent(df)
    features = _choose_features(df)
    # 数值特征做均值中心化，便于解释截距与稳定预测
    feature_means: Dict[str, float] = {}
    for f in features:
        mu = float(pd.to_numeric(df[f], errors="coerce").mean())
        feature_means[f] = mu
        df[f] = pd.to_numeric(df[f], errors="coerce") - mu
    res = _fit_mixedlm_random_slope(df, features)
    paths = _export_model(res, features, out_dir, feature_means)
    print("模型已训练并导出:")
    for k, v in paths.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()


