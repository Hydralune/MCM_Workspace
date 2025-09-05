# 问题一：Y染色体浓度与孕周、BMI的关系模型（完整过程稿）

1. 研究目标与数据

- 目标：定量刻画男胎 Y 染色体浓度 y(%) 与孕周 t(周)、BMI 的关系，检验显著性，并为后续最佳 NIPT 时点提供依据。
- 数据：`数据预处理/male_cleaned.csv`（本次运行约 1006 条），存在同一孕妇的多次检测（纵向数据）。原始数据包含人口学、测序质量、各染色体 Z 值与浓度等（见 `问题重述.md`）。

1. 数据预处理与特征工程

- 实现：`数据预处理/preprocess_male.py`（规则详见 `数据预处理/预处理流程与规则.md`）
  - 列名标准化：中文列映射到 `A`~`AE`（关键位点：`J` 孕周字符串、`K` BMI、`L/M/N/O` 读段/比对、`P` GC、`U/T` Z 值、`V` Y 浓度、`AA` 过滤比例）。
  - 孕周解析：`J` → `gestational_week`（支持“13w+6/13周+6天/13+6”等多格式）。
  - 性别划分：以 `V`（Y 浓度）非空为男胎，空为女胎。
  - BMI 复算与一致性：由身高体重复算 `bmi_recalc`；若原 BMI 缺失或与复算差异 > 0.5，则以复算替换，并保留不一致标记。
  - 基础质控（QC）：`39% ≤ GC(P) ≤ 62%`，`L > 5e5`，`0.40 ≤ M ≤ 0.995`，`N ≤ 0.80`。
  - 极端值与缺失：男性样本对 `y_concentration` 去除 P0.5 以下与 P99.5 以上极端值；保留 `gestational_week / bmi / y_concentration` 均不缺失的记录。
  - 去重合并：`gestational_week_round = round(gestational_week, 1)`；以 `subject_id + gestational_week_round` 分组，数值列取均值，溯源列取首个。
  - 导出清洗数据与统计：`数据预处理/male_cleaned.csv`、`数据预处理/male_stats.csv`、预处理日志。

1. 探索性数据分析（EDA）

- 实现：`问题1/analyze_q1.py`
  - 统一将 y_concentration 转为百分比显示（y_plot），纵轴依据 1%–99% 分位数自适应。
  - 绘制分布、散点（孕周 vs Y，BMI vs Y）、分组箱线图、相关性热图。
  - 生成图表：`问题1/figures/` 下多张 PNG；摘要：`q1_eda_summary.md`。

1. 建模与模型比较

- 实现：`问题1/model_q1.py`
  - 基线模型：OLS（y ~ week + bmi）。
  - 非线性检验：OLS+自然样条（week 的样条项）。
  - 纵向建模：MixedLM 随机截距；MixedLM 随机截距+随机斜率（最终）。
  - 模型比较依据：对数似然与信息准则（AIC/BIC）。
  - 结果要点（详见 `问题1/models/q1_model_report_v2.md`）：
    - OLS：R²≈0.046，未处理个体相关性；
    - OLS 样条：R²略升；
    - MixedLM 随机截距：显著改进；
    - MixedLM 随机斜率：AIC 最低（最优）。
  - 稳健性检验：在统一样本子集上对比 RS_A（week+BMI）、RS_B（week+体重）、RS_C（week+BMI+L/M/N），各模型系数方向与显著性与主模型一致。

1. 最终模型（问题一答案）

- 形式：
  y_i,j = (β0 + u0_i) + β1·t_i,j + β2·BMI_i,j + u1_i·t_i,j + ε_i,j
  其中 y 为 Y 浓度（%），t 为孕周（周）。(u0_i, u1_i) 分别为个体随机截距与随机斜率。
- 关键结论（来自最优模型）：
  - 孕周正效应：β1 ≈ +0.30–0.35 %/周，P<0.001；
  - BMI 负效应：β2 ≈ −0.12 %/单位BMI，P≈0.02；
  - 个体差异显著：基线与增长速率的方差均显著，采用随机斜率模型必要；
  - 方向与临床经验一致：孕周上升→Y 浓度升高；BMI 增加→Y 浓度降低。

1. 概率达标时点与可视化（为问题二服务）

- 基于最优模型，计算 P(Y≥4%)≥0.8/0.9 的最早孕周并给出建议时点（不早于 11 周）：`问题1/models/q1_q2_timing.csv`（已按本次运行更新）。
- 图表：
  - 预测均值曲线（不同 BMI 组）：`问题1/models/fig_pred_curves.png`
  - 概率达标曲线：`问题1/models/fig_prob_curves.png`

1. 论文撰写要点

- 方法选择理由：纵向数据→需处理个体内相关性；AIC 最优→随机斜率 MixedLM。
- 统计显著性：报告 β、标准误/置信区间与 P 值（取自 `q1_model_report_v2.md`）。
- 实证解释：孕周推进是主要正向驱动，BMI 增加带来负向修正。
- 与实践结合：将“概率达标时点”映射为建议检测时点，兼顾检测成功率与尽早检测的风险权衡。

1. 文件索引（相对路径）

- 清洗与统计：`数据预处理/male_cleaned.csv`、`数据预处理/male_stats.csv`
- EDA：`问题1/figures/`、`问题1/q1_eda_summary.md`
- 模型报告与比较：`问题1/models/q1_model_report_v2.md`
- 时点表：`问题1/models/q1_q2_timing.csv`
- 最终说明：`问题1/models/q1_problem1_final.md`
