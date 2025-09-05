import os
import sys
from typing import Optional, List, Dict

import pandas as pd
import numpy as np


def _safe_import_preprocess_utils(project_dir: str):
    """Best-effort import of helpers from 数据预处理/preprocess_data.py."""
    utils: Dict[str, object] = {}
    dp_dir = os.path.join(project_dir, "数据预处理")
    if os.path.isdir(dp_dir):
        sys.path.insert(0, dp_dir)
        try:
            from preprocess_data import (
                try_load_csv_any,  # type: ignore
                normalize_columns,  # type: ignore
                parse_gestation_to_weeks,  # type: ignore
            )
            utils["try_load_csv_any"] = try_load_csv_any
            utils["normalize_columns"] = normalize_columns
            utils["parse_gestation_to_weeks"] = parse_gestation_to_weeks
        except Exception:
            pass
    return utils


def _fallback_try_load_csv_any(csv_path: str) -> pd.DataFrame:
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "cp936"]
    last_err: Optional[Exception] = None
    for enc in encodings:
        try:
            return pd.read_csv(csv_path, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Unable to load CSV {csv_path}. Last error: {last_err}")


def _fallback_normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Minimal noop fallback (assume列名已正确)
    return df


def _normalize_ab_value(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper().replace(" ", "")
    # 常见写法规范化
    if s in {"T13", "13"}:
        return "T13"
    if s in {"T18", "18"}:
        return "T18"
    if s in {"T21", "21"}:
        return "T21"
    # 其它一律视为空/正常
    return ""


def build_q4_female_dataset(raw_df: pd.DataFrame, utils: Dict[str, object]) -> pd.DataFrame:
    normalize_columns = utils.get("normalize_columns", _fallback_normalize_columns)  # type: ignore
    parse_gestation_to_weeks = utils.get("parse_gestation_to_weeks", None)

    df = normalize_columns(raw_df)

    # 仅保留女胎：V列(Y浓度)为空
    if "V" in df.columns:
        df = df[df["V"].isna()].copy()

    # 关键列存在性检查（尽量宽松，缺失则后续置空）
    for c in ["AB", "Q", "R", "S", "T", "X", "Y", "Z", "L", "M", "N", "O", "AA", "K", "C", "J", "G", "D", "E", "B", "H", "I"]:
        if c not in df.columns:
            df[c] = np.nan

    # 计算孕周
    if parse_gestation_to_weeks is not None:
        df["gestational_week"] = df["J"].apply(parse_gestation_to_weeks)
    else:
        # 兜底：尝试直接转为数值
        df["gestational_week"] = pd.to_numeric(df["J"], errors="coerce")

    # IVF 标记
    def _ivf_to_flag(x: object) -> Optional[int]:
        if pd.isna(x):
            return None
        s = str(x)
        s_upper = s.upper()
        return 1 if ("IVF" in s_upper or ("试管" in s) or ("人工" in s)) else 0

    df["ivf_flag"] = df["G"].apply(_ivf_to_flag)

    # BMI 复算并选择
    def _recalc_bmi(height_cm: object, weight_kg: object) -> Optional[float]:
        try:
            h = float(height_cm)
            w = float(weight_kg)
            if h <= 0 or w <= 0:
                return None
            return w / ((h / 100.0) ** 2)
        except Exception:  # noqa: BLE001
            return None

    df["bmi_recalc"] = df.apply(lambda r: _recalc_bmi(r.get("D"), r.get("E")), axis=1)

    def _choose_bmi(orig: object, rec: object) -> Optional[float]:
        if pd.isna(orig) and pd.notna(rec):
            return float(rec)
        try:
            if pd.notna(orig) and pd.notna(rec) and abs(float(orig) - float(rec)) > 0.5:
                return float(rec)
            return float(orig)
        except Exception:  # noqa: BLE001
            return float(rec) if pd.notna(rec) else None

    df["bmi_final"] = [
        _choose_bmi(o, r) for o, r in zip(df["K"], df["bmi_recalc"])
    ]

    # 目标编码
    df["ab_type"] = df["AB"].apply(_normalize_ab_value)
    df["is_abnormal"] = (df["ab_type"].isin(["T13", "T18", "T21"]).astype(int))

    # 数值化特征
    num_cols_map = {
        "q_z13": "Q",
        "r_z18": "R",
        "s_z21": "S",
        "t_zx": "T",
        "gc13": "X",
        "gc18": "Y",
        "gc21": "Z",
        "reads_total": "L",
        "align_ratio": "M",
        "duplicate_ratio": "N",
        "unique_reads": "O",
        "filtered_ratio": "AA",
        "bmi": "bmi_final",
        "maternal_age": "C",
        "gestational_week": "gestational_week",
        "ivf_flag": "ivf_flag",
    }

    out_df = pd.DataFrame()
    for new_name, src in num_cols_map.items():
        if src in df.columns:
            out_df[new_name] = pd.to_numeric(df[src], errors="coerce")
        else:
            out_df[new_name] = np.nan

    # 目标列放在最前
    out_df.insert(0, "ab_type", df["ab_type"].astype(str))
    out_df.insert(0, "is_abnormal", df["is_abnormal"].astype(int))

    # 标识列：subject_id 与 分箱孕周（0.1周）
    subj = df["B"].astype(str).fillna("") if "B" in df.columns else pd.Series([""] * len(df))
    out_df.insert(0, "subject_id", subj)
    out_df["gestational_week_round"] = out_df["gestational_week"].round(1)

    return out_df


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, os.pardir))

    utils = _safe_import_preprocess_utils(project_dir)

    csv_path = os.path.join(project_dir, "附件B.csv")
    if not os.path.exists(csv_path):
        print(f"未找到文件: {csv_path}")
        sys.exit(1)

    try_load_csv_any = utils.get("try_load_csv_any", None)
    if try_load_csv_any is None:
        raw_df = _fallback_try_load_csv_any(csv_path)
    else:
        raw_df = try_load_csv_any(csv_path)  # type: ignore

    out_df = build_q4_female_dataset(raw_df, utils)

    out_csv = os.path.join(base_dir, "female_q4_dataset.csv")
    out_md = os.path.join(base_dir, "female_q4_summary.md")

    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 简要摘要
    vc = out_df["is_abnormal"].value_counts(dropna=False).to_dict()
    ab_vc = out_df["ab_type"].value_counts(dropna=False).to_dict()
    null_counts = out_df.isna().sum().sort_values(ascending=False)

    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Q4 女胎数据集摘要\n\n")
        f.write(f"总样本数: {len(out_df)}\n\n")
        f.write("## 标签分布\n")
        f.write(f"is_abnormal: {vc}\n\n")
        f.write(f"ab_type: {ab_vc}\n\n")
        f.write("## 缺失值统计(Top 20)\n\n")
        f.write(null_counts.head(20).to_string())

    print(f"Saved dataset: {out_csv}")
    print(f"Saved summary: {out_md}")


if __name__ == "__main__":
    main()


