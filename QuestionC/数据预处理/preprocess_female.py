import os
import sys
import pandas as pd

from preprocess_data import try_load_csv_any, preprocess_female_for_q1


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.abspath(os.path.join(base_dir, os.pardir))

    csv_path = os.path.join(project_dir, "附件B.csv")
    if not os.path.exists(csv_path):
        print(f"未找到文件: {csv_path}")
        sys.exit(1)

    raw_df = try_load_csv_any(csv_path)
    cleaned_df, stats = preprocess_female_for_q1(raw_df)

    out_clean = os.path.join(base_dir, "female_cleaned.csv")
    out_stats = os.path.join(base_dir, "female_stats.csv")
    out_log = os.path.join(base_dir, "female_prep_log.md")

    cleaned_df.to_csv(out_clean, index=False, encoding="utf-8-sig")
    stats.to_csv(out_stats, encoding="utf-8-sig")

    with open(out_log, "w", encoding="utf-8") as f:
        f.write("# 女性数据预处理日志\n")
        f.write("- 规则：39%≤GC≤62%，L>5e5，0.40≤M≤0.995，N≤0.80\n")
        f.write(f"- 清洗后样本数：{len(cleaned_df)}\n")

    print(f"Saved: {out_clean}, {out_stats}, {out_log}")


if __name__ == "__main__":
    main()


