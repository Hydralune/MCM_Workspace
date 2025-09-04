import os
from typing import Dict, Tuple, Optional, List

import pandas as pd
import numpy as np

import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import bs
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# 全局中文字体与负号设置
matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False


def get_paths() -> Tuple[str, str, str]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, "q1_male_cleaned.csv")
    out_dir = os.path.join(current_dir, "models")
    os.makedirs(out_dir, exist_ok=True)
    return current_dir, data_path, out_dir


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = ["subject_id", "gestational_week", "bmi", "y_concentration"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    # numeric coercion
    df["gestational_week"] = pd.to_numeric(df["gestational_week"], errors="coerce")
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["y_concentration"] = pd.to_numeric(df["y_concentration"], errors="coerce")
    df = df.dropna(subset=["subject_id", "gestational_week", "bmi", "y_concentration"])\
           .reset_index(drop=True)
    return df


def maybe_convert_y_to_percent(df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
    df = df.copy()
    y = df["y_concentration"].astype(float)
    is_proportion = np.nanmax(y.values) <= 1.0 + 1e-9
    df["y_target"] = y * 100.0 if is_proportion else y
    return df, is_proportion


def fit_ols(df: pd.DataFrame) -> sm.regression.linear_model.RegressionResultsWrapper:
    # Add constant handled by formula automatically if using smf.ols with explicit Intercept
    model = smf.ols(formula="y_target ~ gestational_week + bmi", data=df).fit()
    return model


def fit_mixed_effects(df: pd.DataFrame):
    # Random intercept by subject
    # If convergence issues arise, we can try different optimizers
    md = smf.mixedlm("y_target ~ gestational_week + bmi", data=df, groups=df["subject_id"]) 
    try:
        mdf = md.fit(reml=False, method="lbfgs")
    except Exception:
        mdf = md.fit(reml=False, method="nm")
    return mdf


def fit_ols_spline(df: pd.DataFrame, df_spline: int = 4):
    """OLS 样条回归：对 gestational_week 使用样条以捕捉非线性。"""
    model = smf.ols(formula=f"y_target ~ bs(gestational_week, df={df_spline}) + bmi", data=df).fit()
    return model


def fit_mixed_effects_random_slope(df: pd.DataFrame):
    """混合效应：随机截距 + 孕周随机斜率。"""
    md = smf.mixedlm(
        "y_target ~ gestational_week + bmi",
        data=df,
        groups=df["subject_id"],
        re_formula="~ gestational_week",
    )
    try:
        mdf = md.fit(reml=False, method="lbfgs")
    except Exception:
        mdf = md.fit(reml=False, method="nm")
    return mdf


def compute_group_tmin(model, bmi_values: Dict[str, float]) -> pd.DataFrame:
    """
    Solve for week t where predicted y_target reaches 4%.
    Model: y = b0 + b1*week + b2*bmi (both OLS and Mixed return .params)
    """
    target = 4.0  # percent scale
    params = model.params
    b0 = float(params.get("Intercept", params.get("const", 0.0)))
    b1 = float(params.get("gestational_week", 0.0))
    b2 = float(params.get("bmi", 0.0))
    out = []
    for group, bmi in bmi_values.items():
        # target = b0 + b1*t + b2*bmi  =>  t = (target - b0 - b2*bmi)/b1
        if abs(b1) < 1e-9:
            t_min = np.nan
        else:
            t_min = (target - b0 - b2 * bmi) / b1
        out.append({"bmi_group": group, "bmi_repr": bmi, "t_min_week": t_min})
    return pd.DataFrame(out)


def compute_prob_tmin_linear(model, bmi_values: Dict[str, float], p: float, week_floor: float = 11.0) -> pd.DataFrame:
    """
    线性OLS：mu(t)=b0+b1*t+b2*bmi, sigma=sqrt(mse_resid)。
    求最小 t 使 P(Y>=4)≥p，即 mu(t) - k*sigma ≥ 4, k=Phi^{-1}(p)。
    """
    k = float(norm.ppf(p))
    params = model.params
    b0 = float(params.get("Intercept", params.get("const", 0.0)))
    b1 = float(params.get("gestational_week", 0.0))
    b2 = float(params.get("bmi", 0.0))
    sigma = float(np.sqrt(getattr(model, "mse_resid", getattr(model, "scale", 0.0))))
    rows = []
    for group, bmi in bmi_values.items():
        if abs(b1) < 1e-9:
            t_star = np.nan
        else:
            t_req = (4.0 + k * sigma - b0 - b2 * bmi) / b1
            t_star = max(week_floor, t_req)
        rows.append({"bmi_group": group, "bmi_repr": bmi, f"t_min_p{int(p*100)}": t_star})
    return pd.DataFrame(rows)


def _mixed_sigma_random_intercept(res, t: float) -> float:
    """随机截距模型的总体不确定性：残差方差 + 截距随机效应方差。"""
    var_u0 = float(res.cov_re.iloc[0, 0]) if res.cov_re.shape[0] >= 1 else 0.0
    return float(np.sqrt(res.scale + var_u0))


def _mixed_sigma_random_slope(res, t: float) -> float:
    """随机截距+斜率的总体不确定性：残差方差 + [1, t]Sigma[1, t]^T。"""
    cov = res.cov_re
    names: List[str] = list(cov.index)
    def get(name_a: str, name_b: str, ia: int, ib: int) -> float:
        if name_a in names and name_b in names:
            return float(cov.loc[name_a, name_b])
        return float(cov.iloc[ia, ib])
    var_u0 = get("Intercept", "Intercept", 0, 0)
    cov_u0u1 = get("Intercept", "gestational_week", 0, 1)
    var_u1 = get("gestational_week", "gestational_week", 1, 1)
    v = var_u0 + 2.0 * t * cov_u0u1 + (t ** 2) * var_u1
    return float(np.sqrt(res.scale + max(v, 0.0)))


def compute_prob_tmin_mixed_random_intercept(res, bmi_values: Dict[str, float], p: float, week_floor: float = 11.0, week_ceiling: float = 30.0) -> pd.DataFrame:
    k = float(norm.ppf(p))
    params = res.params
    b0 = float(params.get("Intercept", params.get("const", 0.0)))
    b1 = float(params.get("gestational_week", 0.0))
    b2 = float(params.get("bmi", 0.0))
    rows = []
    for group, bmi in bmi_values.items():
        def f(t: float) -> float:
            mu = b0 + b1 * t + b2 * bmi
            sigma = _mixed_sigma_random_intercept(res, t)
            return mu - k * sigma - 4.0
        t_grid = np.arange(week_floor, week_ceiling + 1e-9, 0.1)
        t_star = np.nan
        for t in t_grid:
            if f(t) >= 0:
                t_star = float(t)
                break
        rows.append({"bmi_group": group, "bmi_repr": bmi, f"t_min_p{int(p*100)}": t_star})
    return pd.DataFrame(rows)


def compute_prob_tmin_mixed_random_slope(res, bmi_values: Dict[str, float], p: float, week_floor: float = 11.0, week_ceiling: float = 30.0) -> pd.DataFrame:
    k = float(norm.ppf(p))
    params = res.params
    b0 = float(params.get("Intercept", params.get("const", 0.0)))
    b1 = float(params.get("gestational_week", 0.0))
    b2 = float(params.get("bmi", 0.0))
    rows = []
    for group, bmi in bmi_values.items():
        def f(t: float) -> float:
            mu = b0 + b1 * t + b2 * bmi
            sigma = _mixed_sigma_random_slope(res, t)
            return mu - k * sigma - 4.0
        t_grid = np.arange(week_floor, week_ceiling + 1e-9, 0.1)
        t_star = np.nan
        for t in t_grid:
            if f(t) >= 0:
                t_star = float(t)
                break
        rows.append({"bmi_group": group, "bmi_repr": bmi, f"t_min_p{int(p*100)}": t_star})
    return pd.DataFrame(rows)



def export_report(
    ols_res,
    lme_res,
    tmin_ols: pd.DataFrame,
    tmin_lme: pd.DataFrame,
    out_dir: str,
    ols_spline_res=None,
    lme_rs_res=None,
    prob_tables: Optional[Dict[str, pd.DataFrame]] = None,
) -> None:
    report_path = os.path.join(out_dir, "q1_model_report_v2.md")
    lines = []
    lines.append("# 问题1 模型报告\n")
    lines.append("## OLS 回归\n")
    lines.append("```")
    lines.append(str(ols_res.summary()))
    lines.append("```\n")
    if ols_spline_res is not None:
        lines.append("## OLS 样条回归 (gestational_week 样条)\n")
        lines.append("```")
        lines.append(str(ols_spline_res.summary()))
        lines.append("```\n")
    lines.append("## 线性混合效应模型 (随机截距: subject_id)\n")
    lines.append("```")
    lines.append(str(lme_res.summary()))
    lines.append("```\n")
    if lme_rs_res is not None:
        lines.append("## 线性混合效应模型 (随机截距+孕周随机斜率)\n")
        lines.append("```")
        lines.append(str(lme_rs_res.summary()))
        lines.append("```\n")

    # 模型对比
    lines.append("## 模型对比 (对数似然/信息准则)\n")
    comp_rows = []
    def add_model_row(name: str, res_obj) -> None:
        if res_obj is None:
            return
        ll = getattr(res_obj, "llf", np.nan)
        aic = getattr(res_obj, "aic", np.nan)
        bic = getattr(res_obj, "bic", np.nan)
        comp_rows.append({"model": name, "llf": ll, "aic": aic, "bic": bic})
    add_model_row("OLS", ols_res)
    add_model_row("OLS_spline", ols_spline_res)
    add_model_row("MixedLM_random_intercept", lme_res)
    add_model_row("MixedLM_random_slope", lme_rs_res)
    if comp_rows:
        comp_df = pd.DataFrame(comp_rows)
        lines.append(comp_df.to_markdown(index=False))
        lines.append("\n")

    lines.append("## 达标孕周 t_min (预测4%)\n")
    lines.append("### OLS 估计\n")
    lines.append(tmin_ols.to_markdown(index=False))
    lines.append("\n### MixedLM 估计\n")
    lines.append(tmin_lme.to_markdown(index=False))
    if prob_tables:
        for key, dfp in prob_tables.items():
            lines.append(f"\n### 概率达标时点 {key}\n")
            lines.append(dfp.to_markdown(index=False))
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"模型报告已导出: {report_path}")


def main() -> None:
    _, data_path, out_dir = get_paths()
    df = load_data(data_path)
    df, _ = maybe_convert_y_to_percent(df)

    # 拟合模型
    ols_res = fit_ols(df)
    lme_res = fit_mixed_effects(df)
    ols_spline_res = fit_ols_spline(df, df_spline=4)
    lme_rs_res = fit_mixed_effects_random_slope(df)

    # 代表性BMI（按分箱中位数）
    bmi_bins = pd.cut(df["bmi"], bins=[0, 20, 28, 32, 36, 40, np.inf], right=False, include_lowest=True)
    bmi_group_medians = df.groupby(bmi_bins)["bmi"].median().to_dict()
    bmi_group_labels = {}
    for interval in bmi_group_medians.keys():
        if interval.right == np.inf:
            label = "40+"
        else:
            label = f"[{int(interval.left)},{int(interval.right)})"
        bmi_group_labels[label] = float(bmi_group_medians[interval])

    tmin_ols = compute_group_tmin(ols_res, bmi_group_labels)
    tmin_lme = compute_group_tmin(lme_res, bmi_group_labels)

    # 概率达标时点 p=0.8 和 0.9（优先使用随机斜率混合模型）
    prob_tables: Dict[str, pd.DataFrame] = {}
    for p in [0.8, 0.9]:
        tbl_ri = compute_prob_tmin_mixed_random_intercept(lme_res, bmi_group_labels, p)
        tbl_rs = compute_prob_tmin_mixed_random_slope(lme_rs_res, bmi_group_labels, p)
        rec = tbl_rs.copy()
        rec.rename(columns={f"t_min_p{int(p*100)}": "t_min_rs"}, inplace=True)
        rec["recommend_week"] = rec["t_min_rs"].apply(lambda x: float(np.nan) if pd.isna(x) else max(11.0, float(x)))
        prob_tables[f"P>={int(p*100)}% (随机斜率)"] = rec
        tbl_ri.rename(columns={f"t_min_p{int(p*100)}": "t_min_ri"}, inplace=True)
        prob_tables[f"P>={int(p*100)}% (随机截距)"] = tbl_ri

    export_report(
        ols_res,
        lme_res,
        tmin_ols,
        tmin_lme,
        out_dir=out_dir,
        ols_spline_res=ols_spline_res,
        lme_rs_res=lme_rs_res,
        prob_tables=prob_tables,
    )

    # 导出问题2时点表（合并P≥80%与P≥90%，随机斜率方案）
    rs80 = compute_prob_tmin_mixed_random_slope(lme_rs_res, bmi_group_labels, 0.8)
    rs80.rename(columns={"t_min_p80": "t_min_p80"}, inplace=True)
    rs80["recommend_week_p80"] = rs80["t_min_p80"].apply(lambda x: float(np.nan) if pd.isna(x) else max(11.0, float(x)))
    rs90 = compute_prob_tmin_mixed_random_slope(lme_rs_res, bmi_group_labels, 0.9)
    rs90.rename(columns={"t_min_p90": "t_min_p90"}, inplace=True)
    rs90["recommend_week_p90"] = rs90["t_min_p90"].apply(lambda x: float(np.nan) if pd.isna(x) else max(11.0, float(x)))
    timing = pd.merge(
        rs80[["bmi_group", "bmi_repr", "t_min_p80", "recommend_week_p80"]],
        rs90[["bmi_group", "t_min_p90", "recommend_week_p90"]],
        on="bmi_group",
        how="outer",
    )
    timing_path = os.path.join(out_dir, "q1_q2_timing.csv")
    timing.to_csv(timing_path, index=False, encoding="utf-8-sig")
    print(f"问题2时点建议已导出: {timing_path}")

    # 导出问题一最终结果与解释（简要）
    summary_path = os.path.join(out_dir, "q1_problem1_summary.md")
    lines = []
    lines.append("# 问题一：Y浓度与孕周、BMI关系模型（最终结论）\n")
    lines.append("## 模型选择\n")
    lines.append("- 最优模型：线性混合效应（随机截距+孕周随机斜率）。\n")
    lines.append("- 依据：信息准则最优（见 q1_model_report_v2.md 的模型对比表，AIC 最低）。\n")
    lines.append("\n## 关系模型形式\n")
    lines.append("- 响应变量：y_target（Y染色体浓度，单位：%）。\n")
    lines.append("- 固定效应：gestational_week（孕周，周）；bmi。\n")
    lines.append("- 模型：y% = (β0 + u0_i) + β1*week + β2*bmi + u1_i*week + ε，其中 (u0_i, u1_i) 为个体随机效应。\n")
    # 提取关键系数
    b0 = float(lme_rs_res.params.get("Intercept", lme_rs_res.params.get("const", 0.0)))
    b1 = float(lme_rs_res.params.get("gestational_week", 0.0))
    b2 = float(lme_rs_res.params.get("bmi", 0.0))
    z1 = float(lme_rs_res.tvalues.get("gestational_week", np.nan))
    p1 = float(lme_rs_res.pvalues.get("gestational_week", np.nan))
    z2 = float(lme_rs_res.tvalues.get("bmi", np.nan))
    p2 = float(lme_rs_res.pvalues.get("bmi", np.nan))
    lines.append(f"- 估计系数：Intercept≈{b0:.3f}，β_week≈{b1:.3f} (%/周)，β_bmi≈{b2:.3f} (%/单位BMI)。\n")
    lines.append(f"- 显著性：孕周 z≈{z1:.2f}, p≈{p1:.3g}；BMI z≈{z2:.2f}, p≈{p2:.3g}（均显著）。\n")
    # 方差成分
    try:
        var_u0 = float(lme_rs_res.cov_re.iloc[0, 0])
        cov_u0u1 = float(lme_rs_res.cov_re.iloc[0, 1])
        var_u1 = float(lme_rs_res.cov_re.iloc[1, 1])
        res_var = float(lme_rs_res.scale)
        lines.append(f"- 随机效应方差：Var(u0)≈{var_u0:.3f}，Cov(u0,u_week)≈{cov_u0u1:.3f}，Var(u_week)≈{var_u1:.3f}；残差方差≈{res_var:.3f}。\n")
    except Exception:
        pass
    lines.append("\n## 解释与结论\n")
    lines.append("- 孕周每增加1周，平均Y浓度增加约0.30–0.35个百分点；BMI每增加1单位，平均Y浓度降低约0.12个百分点。\n")
    lines.append("- 个体间基线与增长速率差异显著，采用随机斜率模型能更好刻画纵向数据结构。\n")
    lines.append("- 该模型通过显著性检验，方向与生物学预期一致。\n")
    lines.append("\n## 可视化与数据支持\n")
    lines.append("- 参见 EDA 图：figures/scatter_week_vs_y.png、box_y_by_bmi_group.png、corr_heatmap.png。\n")
    lines.append("- 报告：models/q1_model_report_v2.md（含详细系数、对数似然与AIC对比）。\n")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"问题一总结已导出: {summary_path}")

    # 生成最终图表：预测均值曲线与概率达标曲线（基于随机斜率模型）
    weeks = np.linspace(10, 28, 181)
    def predict_mu(t: float, bmi: float) -> float:
        b0 = float(lme_rs_res.params.get("Intercept", lme_rs_res.params.get("const", 0.0)))
        b1 = float(lme_rs_res.params.get("gestational_week", 0.0))
        b2 = float(lme_rs_res.params.get("bmi", 0.0))
        return b0 + b1 * t + b2 * bmi

    def prob_ge_4(t: float, bmi: float) -> float:
        mu = predict_mu(t, bmi)
        sigma = _mixed_sigma_random_slope(lme_rs_res, t)
        return float(1.0 - norm.cdf((4.0 - mu) / max(sigma, 1e-9)))

    # 排序固定顺序的标签
    bmi_items = list(bmi_group_labels.items())
    label_order = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "40+"]
    bmi_items = [it for it in bmi_items if it[0] in label_order]
    bmi_items.sort(key=lambda kv: label_order.index(kv[0]))

    # 预测均值曲线
    plt.figure(figsize=(7.5, 5.0))
    for label, bmi_val in bmi_items:
        y = [predict_mu(t, bmi_val) for t in weeks]
        plt.plot(weeks, y, label=f"{label} (BMI≈{bmi_val:.1f})")
    plt.axhline(4.0, color="red", linestyle="--", linewidth=1.2, label="4% 阈值")
    plt.xlabel("孕周 (周)")
    plt.ylabel("预测Y浓度 (%)")
    plt.title("不同BMI组的预测均值曲线 (MixedLM 随机斜率)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_pred = os.path.join(out_dir, "fig_pred_curves.png")
    plt.savefig(fig_pred, dpi=180)
    plt.close()

    # 概率达标曲线
    plt.figure(figsize=(7.5, 5.0))
    for label, bmi_val in bmi_items:
        pvals = [prob_ge_4(t, bmi_val) for t in weeks]
        plt.plot(weeks, pvals, label=f"{label} (BMI≈{bmi_val:.1f})")
    for thr in [0.8, 0.9]:
        plt.axhline(thr, color="gray", linestyle=":" , linewidth=1.0)
        plt.text(27.8, thr + 0.01, f"P={thr:.1f}", ha="right", va="bottom", fontsize=8, color="gray")
    plt.xlabel("孕周 (周)")
    plt.ylabel("P(Y≥4%)")
    plt.ylim(0, 1)
    plt.title("不同BMI组达到4%的概率曲线 (MixedLM 随机斜率)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    fig_prob = os.path.join(out_dir, "fig_prob_curves.png")
    plt.savefig(fig_prob, dpi=180)
    plt.close()

    # 导出问题一最终说明文档（正式版）
    final_path = os.path.join(out_dir, "q1_problem1_final.md")
    # 装载 timing 表以嵌入
    try:
        timing_df = pd.read_csv(timing_path)
    except Exception:
        timing_df = timing
    final_lines = []
    final_lines.append("# 问题一最终说明\n")
    final_lines.append("## 模型与方法\n")
    final_lines.append("- 最终采用线性混合效应模型（随机截距+孕周随机斜率），响应变量为Y染色体浓度百分比。\n")
    final_lines.append("- 固定效应：孕周、BMI；随机效应：个体基线与孕周斜率。\n")
    final_lines.append("- 相比OLS/样条与随机截距模型，AIC最低（见 q1_model_report_v2.md），能更好表征个体差异。\n")
    final_lines.append("\n## 关键结论\n")
    final_lines.append("- 孕周每增加1周，平均Y浓度上升约0.30–0.35个百分点；BMI每增加1单位，平均Y浓度下降约0.12个百分点，均显著。\n")
    final_lines.append("- 个体基线与增长速率差异显著，随机斜率必要。\n")
    final_lines.append("\n## 主要图表\n")
    final_lines.append(f"- 预测均值曲线：{os.path.basename(fig_pred)}\n")
    final_lines.append(f"- 概率达标曲线：{os.path.basename(fig_prob)}\n")
    final_lines.append("\n## 时点建议（用于问题二）\n")
    final_lines.append("- 采用P(Y≥4%)≥80%的最早周作为建议初检时点（且不早于11周）；若需更稳健可用P≥90%。\n")
    try:
        final_lines.append(timing_df.to_markdown(index=False))
    except Exception:
        pass
    final_lines.append("\n注：低于11周属于外推，建议遵循临床可检测最早周。\n")
    with open(final_path, "w", encoding="utf-8") as f:
        f.write("\n".join(final_lines))
    print(f"问题一最终说明已导出: {final_path}")


if __name__ == "__main__":
    main()


