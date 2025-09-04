# 预处理日志
- QC模式：light
- 启用：QC过滤（温和）、BMI复算、同孕周去重(0.1周)、不做温莎化
- QC规则：35%≤GC≤65%，AA≤0.60，L>5e5，0.40≤M≤0.995，N≤0.80
- 清洗后样本数：1020

## 统计摘要(含处理前后规模)

|                   |   count |        mean |         std |         min |         25% |         50% |        75% |        max |   missing |
|:------------------|--------:|------------:|------------:|------------:|------------:|------------:|-----------:|-----------:|----------:|
| gestational_week  |    1020 |  16.6751    |   3.99499   |  11         |  13.2857    |  15.8571    |  19.8571   |  29        |         0 |
| bmi               |    1020 |  32.2791    |   2.99155   |  20.7031    |  30.1799    |  31.8116    |  33.8741   |  46.875    |         0 |
| y_concentration   |    1020 |   0.0779832 |   0.0333218 |   0.0100039 |   0.0521937 |   0.0759441 |   0.099242 |   0.234218 |         0 |
| rows_before_qc    |    1082 | nan         | nan         | nan         | nan         | nan         | nan        | nan        |       nan |
| rows_after_qc     |    1082 | nan         | nan         | nan         | nan         | nan         | nan        | nan        |       nan |
| rows_before_dedup |    1082 | nan         | nan         | nan         | nan         | nan         | nan        | nan        |       nan |
| rows_after_dedup  |    1020 | nan         | nan         | nan         | nan         | nan         | nan        | nan        |       nan |
| qc_rules          |       5 | nan         | nan         | nan         | nan         | nan         | nan        | nan        |       nan |