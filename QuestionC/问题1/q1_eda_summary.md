# 问题1 EDA 摘要

## 样本规模

记录数: 1020


## 描述性统计（关键变量）

- gestational_week: count=1020, mean=16.675, std=3.995, min=11.000, 25%=13.286, 50%=15.857, 75%=19.857, max=29.000
- bmi: count=1020, mean=32.279, std=2.992, min=20.703, 25%=30.180, 50%=31.812, 75%=33.874, max=46.875
- y_concentration: count=1020, mean=0.078, std=0.033, min=0.010, 25%=0.052, 50%=0.076, 75%=0.099, max=0.234

## 相关性 (Pearson)

- gestational_week: gestational_week=1.000, bmi=0.143, y_concentration=0.138
- bmi: gestational_week=0.143, bmi=1.000, y_concentration=-0.146
- y_concentration: gestational_week=0.138, bmi=-0.146, y_concentration=1.000

## 生成图表

- figures/distributions.png
- figures/scatter_week_vs_y.png
- figures/scatter_bmi_vs_y.png
- figures/box_y_by_bmi_group.png
- figures/corr_heatmap.png
