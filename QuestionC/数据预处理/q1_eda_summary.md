# 问题1 EDA 摘要

## 样本规模

记录数: 1006


## 描述性统计（关键变量）

- gestational_week: count=1006, mean=16.646, std=3.984, min=11.000, 25%=13.286, 50%=15.857, 75%=19.857, max=29.000
- bmi: count=1006, mean=32.260, std=2.933, min=20.703, 25%=30.182, 50%=31.812, 75%=33.874, max=46.875
- y_concentration: count=1006, mean=0.078, std=0.031, min=0.020, 25%=0.052, 50%=0.076, 75%=0.099, max=0.192

## 基础相关性 (Pearson)

- gestational_week: gestational_week=1.000, bmi=0.148, y_concentration=0.116
- bmi: gestational_week=0.148, bmi=1.000, y_concentration=-0.153
- y_concentration: gestational_week=0.116, bmi=-0.153, y_concentration=1.000

## 扩展相关性与Top相关（对 y_concentration）

- draw_index: r=0.337
- maternal_weight_kg: r=-0.182
- bmi: r=-0.153
- bmi_recalc: r=-0.153
- gestational_week: r=0.116
- M: r=-0.111
- maternal_height_cm: r=-0.109
- L: r=-0.106
- maternal_age: r=-0.104
- N: r=0.102

详见: figures/corr_matrix_extended.csv 与 figures/corr_heatmap_extended.png


## 生成图表

- figures/distributions.png
- figures/scatter_week_vs_y.png
- figures/scatter_bmi_vs_y.png
- figures/box_y_by_bmi_group.png
- figures/corr_heatmap_basic.png
- figures/corr_heatmap_extended.png
