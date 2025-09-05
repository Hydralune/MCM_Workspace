import os
import json
import argparse
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np


def detect_project_dirs() -> Tuple[str, str]:
    """
    Return (project_dir, script_dir).
    project_dir: root folder that contains 数据预处理/ and 问题2/
    script_dir: directory of this script
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    return project_dir, script_dir


def choose_features(df: pd.DataFrame, user_features: List[str] = None) -> List[str]:
    """
    Choose numeric features to standardize. If user_features provided, use their
    intersection with df columns; otherwise, use a sensible default list.
    """
    if user_features:
        return [c for c in user_features if c in df.columns]

    # Default candidates for 问题2男胎聚类/分组前的标准化
    default_candidates = [
        "gestational_week",  # 孕周(周)
        "bmi",               # BMI(kg/m^2)
        "y_concentration",   # Y染色体浓度(比例)
    ]
    return [c for c in default_candidates if c in df.columns]


def compute_zscore(
    df: pd.DataFrame, features: List[str]
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    For each feature, compute z = (x - mean) / std (NaN-aware). If std≈0, set std to 1
    and the resulting z-scores to 0 to avoid division by zero.
    Returns: (df_with_z_columns, params)
    params format: { feature: {"mean": float, "std": float} }
    """
    params: Dict[str, Dict[str, float]] = {}
    out_df = df.copy()
    for col in features:
        col_values = pd.to_numeric(out_df[col], errors="coerce")
        mu = float(col_values.mean(skipna=True)) if col_values.notna().any() else 0.0
        sigma = float(col_values.std(ddof=0, skipna=True)) if col_values.notna().any() else 0.0
        if not np.isfinite(sigma) or sigma <= 1e-12:
            # Near-constant feature; define std=1, z=0 to be safe
            sigma = 1.0
            z = pd.Series(0.0, index=out_df.index)
        else:
            z = (col_values - mu) / sigma

        out_df[f"{col}_z"] = z
        params[col] = {"mean": mu, "std": sigma}

    return out_df, params


def main():
    project_dir, script_dir = detect_project_dirs()

    parser = argparse.ArgumentParser(description="Z-score 标准化（仅男胎）")
    parser.add_argument(
        "--input",
        default=os.path.join(project_dir, "数据预处理", "male_cleaned.csv"),
        help="输入CSV路径（male_cleaned.csv）",
    )
    parser.add_argument(
        "--output_csv",
        default=os.path.join(script_dir, "male_standardized.csv"),
        help="标准化后的输出CSV路径",
    )
    parser.add_argument(
        "--params_json",
        default=os.path.join(script_dir, "zscore_params.json"),
        help="保存均值/标准差参数的JSON路径",
    )
    parser.add_argument(
        "--features",
        nargs="*",
        default=None,
        help="显式指定需要标准化的列名（空则使用默认候选）",
    )
    args = parser.parse_args()

    # 读取数据
    df = pd.read_csv(args.input, encoding="utf-8-sig")

    # 选择特征
    feature_cols = choose_features(df, args.features)
    if not feature_cols:
        raise SystemExit("未找到可标准化的特征列，请检查输入数据或使用 --features 指定。")

    # 计算Z-score
    std_df, params = compute_zscore(df, feature_cols)

    # 保存结果
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    std_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    with open(args.params_json, "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)

    # 控制台提示
    print("Standardized features:", ", ".join(feature_cols))
    print("Saved:", args.output_csv)
    print("Params:", args.params_json)


if __name__ == "__main__":
    main()





