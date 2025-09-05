# 问题一：Y 染色体浓度与孕周、BMI 的关系模型（最终解答稿）

## 1. 研究目标与数据

- 目标：定量刻画男胎 Y 染色体浓度 y(%) 与孕周 t(周)、BMI 之间的关系，检验显著性，为后续最佳 NIPT 时点提供依据。
  - BMI 复算与一致性：由身高体重复算 `bmi_recalc`，若原 BMI 缺失或与复算差异 > 0.5，则替换，并保留不一致标记。
  - 基础质控筛选（QC）：`39% ≤ GC(P) ≤ 62%`，`L > 5e5`，`0.40 ≤ M ≤ 0.995`，`N ≤ 0.80`。
  - Y 浓度极端值：男性样本对 `y_concentration` 去除 P0.5 以下与 P99.5 以上极端值。
  - 缺失处理与必需字段：男性保留 `gestational_week / bmi / y_concentration` 三者齐全的记录。
  - 去重：计算 `gestational_week_round = round(gestational_week, 1)`，按 `subject_id + gestational_week_round` 分组，数值取均值，溯源字段取首个。

## 2. 方法与模型

- 采用线性混合效应模型（MixedLM），考虑同一孕妇的多次检测（随机效应），以避免样本相关性对显著性与区间估计的偏倚。
- 模型形式：
  y_i,j = (β0 + u0_i) + β1·t_i,j + β2·BMI_i,j + u1_i·t_i,j + ε_i,j
  其中 i 为孕妇，j 为第 j 次检测；(u0_i, u1_i) 为个体随机截距与孕周随机斜率，ε_i,j 为误差项；y 为 Y 浓度（单位：%）。
- 对比模型：
  - OLS 基线；
  - OLS+样条（孕周的自然样条）；
  - MixedLM 随机截距；
  - MixedLM 随机截距+随机斜率（最终选用）。

## 3. 主要结果

- 显著性与方向（最终模型：随机截距+随机斜率）：
  - 孕周系数 β1 ≈ +0.30–0.35 %/周（P<0.001）：孕周越大，Y 浓度越高；
  - BMI 系数 β2 ≈ −0.12 %/单位BMI（P≈0.02）：BMI 越大，Y 浓度越低；
  - 个体差异显著：基线与增长速率在不同个体间存在明显变异，采用随机斜率模型必要。
- 模型优选（对数似然/信息准则，详见 `q1_model_report_v2.md`）：
  MixedLM 随机斜率优于随机截距、OLS 与 OLS+样条（AIC 最低）。

### EDA 与相关性要点（Pearson）

- 关键相关：gestational_week 与 y 正相关 r≈0.116；BMI 与 y 负相关 r≈−0.153；体重与 y 负相关 r≈−0.182（但与 BMI 强共线，r≈0.84）。
- 技术/质控：M（比对比例）与 y r≈−0.111，L（总读段）与 y r≈−0.106，N（重复比例）与 y r≈+0.102，提示质控指标与 y 存在弱相关。
- draw_index 与 y r≈+0.337，但与孕周强相关 r≈0.768，属于“检测序号/时间”的代理，不与孕周同置于同一固定效应以避免多重共线。

## 4. 可视化与解释

- 预测均值曲线：不同 BMI 组的均值曲线近似平行，随孕周上升整体上移；阈值线 4% 作为达标参考（见 `fig_pred_curves.png`）。
- 概率达标曲线：P(Y≥4%) 随孕周单调上升，BMI 越高曲线越靠右（见 `fig_prob_curves.png`）。

## 5. 结论（问题一）

- 男胎 Y 浓度与孕周显著正相关、与 BMI 显著负相关，上述关系在控制个体内相关性后依然稳健；
- 线性混合效应模型（随机斜率）能最好地描述数据结构并提供可靠的统计推断；
- 基于该模型可为后续问题二计算各 BMI 组的“最早达标时点”和“概率达标时点”，相应结果已导出：`q1_q2_timing.csv`。

## 6. 稳健性与敏感性分析

- 稳健模型（随机斜率 MixedLM）对比：
  - RS_A：gestational_week + BMI；
  - RS_B：gestational_week + 体重（替代 BMI）；
  - RS_C：gestational_week + BMI + L + M + N（纳入代表性质控变量）。
- 结论：各模型的系数方向与显著性总体与主模型一致；在纳入质控变量后，孕周正向、BMI 负向的结论保持稳健。draw_index 因与孕周强相关而未纳入。

## 7. 附：关键文件（相对路径）

- 报告与模型对比：`问题1/models/q1_model_report_v2.md`
- 建议时点表：`问题1/models/q1_q2_timing.csv`
- 图表：`问题1/models/fig_pred_curves.png`、`问题1/models/fig_prob_curves.png`
- 相关性图与矩阵：`问题1/figures/corr_heatmap_basic.png`、`问题1/figures/corr_heatmap_extended.png`、`问题1/figures/corr_matrix_extended.csv`

注：若用于正式论文，可将本段直接作为“问题一结果与讨论”的主体文字，并在文中按需插入图表与表格。
