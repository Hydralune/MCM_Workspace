import os
import sys
from typing import Optional, Tuple, List, Dict

import pandas as pd
import numpy as np


def parse_gestation_to_weeks(value: str) -> Optional[float]:
    """
    Convert gestational age strings like "13w+6", "13w", "13W+6", "13+6", "13周+6天" to weeks (float).
    Returns None if cannot be parsed.
    """
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None

    # Normalize common markers
    text = (
        text.replace("周", "w")
        .replace("天", "")
        .replace("W", "w")
        .replace("周+", "w+")
        .replace("＋", "+")
    )

    # Accept forms: 13w+6, 13w, 13+6, 13
    weeks = None
    days = 0
    try:
        if "w" in text:
            main = text
            if "+" in main:
                w_part, d_part = main.split("+")
                w_part = w_part.replace("w", "")
                weeks = float(w_part)
                days = float(d_part)
            else:
                weeks = float(main.replace("w", ""))
        else:
            if "+" in text:
                w_part, d_part = text.split("+")
                weeks = float(w_part)
                days = float(d_part)
            else:
                weeks = float(text)
    except Exception:
        return None

    if weeks is None:
        return None
    # Convert days to week fraction (7 days per week)
    return weeks + (days / 7.0)


def coerce_geq3_to_int(value) -> Optional[int]:
    """Convert strings like '≥3', '>=3', '≧3' to int 3; pass through numeric; None for invalid."""
    if pd.isna(value):
        return None
    s = str(value).strip()
    if not s:
        return None
    for sym in ["≥", ">=", "≧"]:
        if s.startswith(sym):
            try:
                return int(s.replace(sym, ""))
            except Exception:
                return 3
    # Try direct int
    try:
        return int(float(s))
    except Exception:
        return None


def try_load_attachment(base_dir: str) -> pd.DataFrame:
    """
    Load 附件.xlsx if present; otherwise try 附件.csv with multiple encodings.
    """
    xlsx_path = os.path.join(base_dir, "附件.xlsx")
    csv_path = os.path.join(base_dir, "附件.csv")

    if os.path.exists(xlsx_path):
        try:
            df = pd.read_excel(xlsx_path)
            return df
        except Exception as e:
            print(f"Failed to read Excel: {e}")

    # Try CSV with common encodings
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb2312", "cp936"]
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            print(f"Loaded CSV with encoding: {enc}")
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Unable to load attachment from {xlsx_path} or {csv_path}. Last error: {last_err}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names to match the spec in 问题重述.md."""
    # Expected mapping hints by position when header may be garbled but order preserved
    # Fallback to positional rename if we detect likely first header cell is garbled (non-ASCII)
    col_map = {
        "样本序号": "A",
        "孕妇代码": "B",
        "孕妇年龄": "C",
        "孕妇身高": "D",
        "孕妇体重": "E",
        "末次月经时间": "F",
        "IVF 妊娠方式": "G",
        "检测时间": "H",
        "检测抽血次数": "I",
        "孕妇本次检测时的孕周（周数+天数）": "J",
        "孕妇 BMI 指标": "K",
        "原始测序数据的总读段数（个）": "L",
        "总读段数中在参考基因组上比对的比例": "M",
        "总读段数中重复读段的比例": "N",
        "总读段数中唯一比对的读段数（个）": "O",
        "GC 含量": "P",
        "13号染色体的Z值": "Q",
        "18号染色体的Z值": "R",
        "21号染色体的Z值": "S",
        "X染色体的Z值": "T",
        "Y染色体的Z值（女胎数据此列为空白）": "U",
        "Y染色体浓度, 即Y染色体游离 DNA 片段的比例（女胎数据此列为空白）": "V",
        "X染色体浓度（其数值是通过生物信息学在一定假设下通过数据分析估计得出, 可能出现负值）": "W",
        "13号染色体的GC含量": "X",
        "18号染色体的GC含量": "Y",
        "21号染色体的GC含量": "Z",
        "被过滤掉的读段数占总读段数的比例": "AA",
        "检测出的13号, 18号, 21号染色体非整倍体, 即数量异常, 空白即为无异常": "AB",
        "孕妇的怀孕次数": "AC",
        "孕妇的生产次数": "AD",
        "胎儿是否健康（婴儿出生后的结果）": "AE",
    }

    # Determine if header looks garbled by checking ASCII ratio of first header
    def is_garbled(s: str) -> bool:
        try:
            return sum(ord(ch) < 128 for ch in s) / max(len(s), 1) < 0.5
        except Exception:
            return False

    renames = {}
    if len(df.columns) >= 31 and is_garbled(str(df.columns[0])):
        # Attempt positional rename according to known order
        expected_order = [
            "A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P",
            "Q","R","S","T","U","V","W","X","Y","Z","AA","AB","AC","AD","AE",
        ]
        renames = {df.columns[i]: expected_order[i] for i in range(min(len(df.columns), len(expected_order)))}
    else:
        for c in df.columns:
            if c in col_map:
                renames[c] = col_map[c]
    df = df.rename(columns=renames)
    return df


def _recalc_bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    try:
        h = float(height_cm)
        w = float(weight_kg)
        if h <= 0 or w <= 0:
            return None
        return w / ((h / 100.0) ** 2)
    except Exception:
        return None


def _winsorize_series(s: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    if s.empty:
        return s
    q_low = s.quantile(lower)
    q_high = s.quantile(upper)
    return s.clip(q_low, q_high)


def preprocess_for_q1(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return cleaned male-fetus dataset and basic descriptive stats.
    """
    df = normalize_columns(df)

    # Convert key columns
    if "J" in df.columns:
        df["gestational_week"] = df["J"].apply(parse_gestation_to_weeks)
    else:
        df["gestational_week"] = np.nan

    if "AC" in df.columns:
        df["gravidity"] = df["AC"].apply(coerce_geq3_to_int)
    if "AD" in df.columns:
        df["parity"] = df["AD"].apply(coerce_geq3_to_int)

    # Filter male fetus: V not null
    if "V" not in df.columns:
        raise ValueError("Column V (Y chromosome concentration) not found after normalization.")
    male_df = df[df["V"].notna()].copy()

    # Keep relevant columns for Q1 (保留溯源信息)
    keep_cols = [c for c in [
        "B",  # subject_id
        "C",  # age
        "D",  # height
        "E",  # weight
        "K",  # bmi
        "J",  # 原孕周字符串
        "H",  # 检测时间
        "I",  # 抽血次数
        "gestational_week",
        "V",  # y concentration
        "U",  # y zscore
        "L","M","N","O","P","X","Y","Z","AA",  # 质控相关
    ] if c in male_df.columns]
    male_df = male_df[keep_cols]
    male_df = male_df.rename(columns={
        "B": "subject_id",
        "C": "maternal_age",
        "D": "maternal_height_cm",
        "E": "maternal_weight_kg",
        "K": "bmi",
        "J": "gestation_str",
        "H": "draw_date",
        "I": "draw_index",
        "V": "y_concentration",
        "U": "y_zscore",
    })

    # QC 策略（分级）：none / light / strict；默认 light
    qc_mode = os.environ.get("QC_MODE", "light").lower()
    qc_rules_applied: List[str] = []
    before_qc = len(male_df)
    if qc_mode not in {"none", "light", "strict"}:
        qc_mode = "light"

    if qc_mode == "strict":
        # 与先前相同的严格规则
        if "P" in male_df.columns:
            male_df = male_df[(male_df["P"].astype(float) >= 0.40) & (male_df["P"].astype(float) <= 0.60)]
            qc_rules_applied.append("40%≤GC≤60%")
        if "AA" in male_df.columns:
            male_df = male_df[male_df["AA"].astype(float) <= 0.30]
            qc_rules_applied.append("AA≤0.30")
        if "L" in male_df.columns:
            male_df = male_df[male_df["L"].astype(float) > 1_000_000]
            qc_rules_applied.append("L>1e6")
        if "M" in male_df.columns:
            male_df = male_df[(male_df["M"].astype(float) >= 0.60) & (male_df["M"].astype(float) <= 0.98)]
            qc_rules_applied.append("0.60≤M≤0.98")
        if "N" in male_df.columns:
            male_df = male_df[male_df["N"].astype(float) <= 0.50]
            qc_rules_applied.append("N≤0.50")
    elif qc_mode == "light":
        # 温和、必要的质控：放宽阈值，尽量少丢样本
        if "P" in male_df.columns:
            male_df = male_df[(male_df["P"].astype(float) >= 0.35) & (male_df["P"].astype(float) <= 0.65)]
            qc_rules_applied.append("35%≤GC≤65%")
        if "AA" in male_df.columns:
            male_df = male_df[male_df["AA"].astype(float) <= 0.60]
            qc_rules_applied.append("AA≤0.60")
        if "L" in male_df.columns:
            male_df = male_df[male_df["L"].astype(float) > 500_000]
            qc_rules_applied.append("L>5e5")
        # M/N 仅做极端过滤
        if "M" in male_df.columns:
            male_df = male_df[(male_df["M"].astype(float) >= 0.40) & (male_df["M"].astype(float) <= 0.995)]
            qc_rules_applied.append("0.40≤M≤0.995")
        if "N" in male_df.columns:
            male_df = male_df[male_df["N"].astype(float) <= 0.80]
            qc_rules_applied.append("N≤0.80")
    else:
        # none：不做质控
        qc_rules_applied.append("no_qc")

    after_qc = len(male_df)

    # Ensure numeric
    numeric_cols = ["maternal_age", "maternal_height_cm", "maternal_weight_kg", "bmi", "gestational_week", "y_concentration", "y_zscore"]
    for col in numeric_cols:
        if col in male_df.columns:
            male_df[col] = pd.to_numeric(male_df[col], errors="coerce")

    # BMI 复算与一致性校验（默认替换异常差异）
    if {"maternal_height_cm", "maternal_weight_kg"}.issubset(male_df.columns):
        male_df["bmi_recalc"] = male_df.apply(lambda r: _recalc_bmi(r.get("maternal_height_cm"), r.get("maternal_weight_kg")), axis=1)
        # 若原K缺失则用复算；若两者差异>0.5，以复算替换，同时标注
        def _choose_bmi(row):
            orig = row.get("bmi")
            rec = row.get("bmi_recalc")
            if pd.isna(orig) and pd.notna(rec):
                return rec
            try:
                if pd.notna(orig) and pd.notna(rec) and abs(float(orig) - float(rec)) > 0.5:
                    return rec
                return orig
            except Exception:
                return rec if pd.notna(rec) else orig
        male_df["bmi"] = male_df.apply(_choose_bmi, axis=1)
        male_df["bmi_mismatch"] = (male_df["bmi_recalc"].notna()) & (male_df["bmi"].notna()) & (abs(male_df["bmi_recalc"] - male_df["bmi"]) > 0.5)

    # Drop rows with missing essential fields
    male_df = male_df.dropna(subset=["gestational_week", "bmi", "y_concentration"])

    # 同孕周去重合并（按0.1周分箱），数值取均值；保留溯源字段首个
    male_df["gestational_week_round"] = male_df["gestational_week"].round(1)
    group_keys = ["subject_id", "gestational_week_round"]
    dedup_before = len(male_df)
    if set(group_keys).issubset(male_df.columns):
        num_cols = male_df.select_dtypes(include=[np.number]).columns.tolist()
        # 避免对索引键再次聚合
        for k in group_keys:
            if k in num_cols:
                num_cols.remove(k)
        agg_map: Dict[str, str] = {c: "mean" for c in num_cols}
        first_cols = [c for c in ["gestation_str", "draw_date", "draw_index"] if c in male_df.columns]
        for c in first_cols:
            agg_map[c] = "first"
        male_df = male_df.groupby(group_keys, as_index=False).agg(agg_map)
    dedup_after = len(male_df)

    # 温莎化：strict 模式开启 1%-99%；light/none 关闭
    if qc_mode == "strict":
        if "y_concentration" in male_df.columns:
            male_df["y_concentration"] = _winsorize_series(male_df["y_concentration"].astype(float), 0.01, 0.99)
        if "bmi" in male_df.columns:
            male_df["bmi"] = _winsorize_series(male_df["bmi"].astype(float), 0.01, 0.99)

    male_df = male_df.dropna(subset=["gestational_week", "bmi", "y_concentration"]).reset_index(drop=True)

    # Basic stats
    stats = male_df[[c for c in ["gestational_week", "bmi", "y_concentration"] if c in male_df.columns]].describe().T
    stats["missing"] = male_df[[c for c in ["gestational_week", "bmi", "y_concentration"] if c in male_df.columns]].isna().sum().values

    # 附：记录预处理日志（便于审计）
    stats.loc["rows_before_qc", "count"] = before_qc
    stats.loc["rows_after_qc", "count"] = after_qc
    stats.loc["rows_before_dedup", "count"] = dedup_before
    stats.loc["rows_after_dedup", "count"] = dedup_after
    stats.loc["qc_rules", "count"] = len(qc_rules_applied)

    return male_df, stats


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, os.pardir))
    try:
        raw_df = try_load_attachment(project_dir)
    except Exception as e:
        print(str(e))
        sys.exit(1)

    male_df, stats = preprocess_for_q1(raw_df)

    out_clean = os.path.join(base_dir, "q1_male_cleaned.csv")
    out_stats = os.path.join(base_dir, "q1_male_stats.csv")
    male_df.to_csv(out_clean, index=False, encoding="utf-8-sig")
    stats.to_csv(out_stats, encoding="utf-8-sig")

    # 预处理日志
    qc_mode = os.environ.get("QC_MODE", "light").lower()
    if qc_mode not in {"none", "light", "strict"}:
        qc_mode = "light"
    log_path = os.path.join(base_dir, "q1_prep_log.md")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("# 预处理日志\n")
        f.write(f"- QC模式：{qc_mode}\n")
        if qc_mode == "strict":
            f.write("- 启用：QC过滤（严格）、BMI复算、同孕周去重(0.1周)、1%-99%温莎化\n")
            f.write("- QC规则：40%≤GC≤60%，AA≤0.30，L>1e6，0.60≤M≤0.98，N≤0.50\n")
        elif qc_mode == "light":
            f.write("- 启用：QC过滤（温和）、BMI复算、同孕周去重(0.1周)、不做温莎化\n")
            f.write("- QC规则：35%≤GC≤65%，AA≤0.60，L>5e5，0.40≤M≤0.995，N≤0.80\n")
        else:
            f.write("- 启用：BMI复算、同孕周去重(0.1周)，未启用QC与温莎化\n")
        f.write(f"- 清洗后样本数：{len(male_df)}\n")
        f.write("\n## 统计摘要(含处理前后规模)\n\n")
        try:
            f.write(stats.to_markdown())
        except Exception:
            f.write(str(stats))

    print(f"Saved cleaned male-fetus dataset: {out_clean}")
    print(f"Saved basic stats: {out_stats}")
    print(f"Saved preprocess log: {log_path}")


if __name__ == "__main__":
    main()


