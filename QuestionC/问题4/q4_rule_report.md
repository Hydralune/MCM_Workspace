# Q4 规则+校准概率判定报告

## 概率模型（仅Z系+衍生，Isotonic校准）
AUC: 0.7174, AP: 0.1821

F1最优阈值: 0.0973

        pred_0  pred_1
true_0     395     158
true_1      23      29

              precision    recall  f1-score   support

           0       0.94      0.71      0.81       553
           1       0.16      0.56      0.24        52

    accuracy                           0.70       605
   macro avg       0.55      0.64      0.53       605
weighted avg       0.88      0.70      0.76       605


固定阈值(p=0.1086)

        pred_0  pred_1
true_0     497      56
true_1      39      13

              precision    recall  f1-score   support

           0       0.93      0.90      0.91       553
           1       0.19      0.25      0.21        52

    accuracy                           0.84       605
   macro avg       0.56      0.57      0.56       605
weighted avg       0.86      0.84      0.85       605


### 高召回模式（PR曲线选点）

Recall≈0.70: 阈值=0.0809, Precision≈0.1407, Recall≈0.7115

        pred_0  pred_1
true_0     279     274
true_1       9      43

              precision    recall  f1-score   support

           0       0.97      0.50      0.66       553
           1       0.14      0.83      0.23        52

    accuracy                           0.53       605
   macro avg       0.55      0.67      0.45       605
weighted avg       0.90      0.53      0.63       605


Recall≈0.80: 阈值=0.0795, Precision≈0.1384, Recall≈0.8462

        pred_0  pred_1
true_0     277     276
true_1       8      44

              precision    recall  f1-score   support

           0       0.97      0.50      0.66       553
           1       0.14      0.85      0.24        52

    accuracy                           0.53       605
   macro avg       0.55      0.67      0.45       605
weighted avg       0.90      0.53      0.62       605


Recall≈0.90: 阈值=0.0776, Precision≈0.1259, Recall≈0.9808

        pred_0  pred_1
true_0     198     355
true_1       1      51

              precision    recall  f1-score   support

           0       0.99      0.36      0.53       553
           1       0.13      0.98      0.22        52

    accuracy                           0.41       605
   macro avg       0.56      0.67      0.37       605
weighted avg       0.92      0.41      0.50       605


### Top-K 模式

Top-5% 阈值≈0.1480

        pred_0  pred_1
true_0     528      25
true_1      47       5

              precision    recall  f1-score   support

           0       0.92      0.95      0.94       553
           1       0.17      0.10      0.12        52

    accuracy                           0.88       605
   macro avg       0.54      0.53      0.53       605
weighted avg       0.85      0.88      0.87       605


Top-10% 阈值≈0.1267

        pred_0  pred_1
true_0     505      48
true_1      40      12

              precision    recall  f1-score   support

           0       0.93      0.91      0.92       553
           1       0.20      0.23      0.21        52

    accuracy                           0.85       605
   macro avg       0.56      0.57      0.57       605
weighted avg       0.86      0.85      0.86       605


## 阈值分模式建议

- 高召回模式：采用PR曲线选点（Recall≈0.7/0.8/0.9），用于疾病初筛，需人工复核。
- 平衡模式：F1最优阈值（本次≈0.0973）。
- 高精度模式：可在PR曲线中寻找Precision≥0.30的最大Recall点；若当前不存在，则提示受限。

## 规则直判（Z阈值+孕周、质控标记不剔除）
        pred_0  pred_1
true_0     494      59
true_1      46       6

              precision    recall  f1-score   support

           0       0.91      0.89      0.90       553
           1       0.09      0.12      0.10        52

    accuracy                           0.83       605
   macro avg       0.50      0.50      0.50       605
weighted avg       0.84      0.83      0.84       605


## 质控统计（不剔除，仅提示）
通过质控: 602, 未通过: 3

若需更严格评估，可仅在质控通过子集上统计：
AUC_sub: 0.7176573426573427, AP_sub: 0.18472653332026678


## 阈值扫描：z_max 与 分染色体

z_max: 达到Recall≥0.7的最小阈值 τ=1.50

        pred_0  pred_1
true_0     326     227
true_1      32      20
Precision=0.0810, Recall=0.3846, F1=0.1338

z_max: 达到Recall≥0.8的最小阈值 τ=1.50

        pred_0  pred_1
true_0     326     227
true_1      32      20
Precision=0.0810, Recall=0.3846, F1=0.1338

### 分染色体一对多阈值（合并规则：任一达标即判阳）

目标Recall≥0.7: τ13=3.20, τ18=3.20, τ21=3.20

        pred_0  pred_1
true_0     521      32
true_1      50       2
Precision=0.0588, Recall=0.0385, F1=0.0465

目标Recall≥0.8: τ13=3.20, τ18=3.20, τ21=3.20

        pred_0  pred_1
true_0     521      32
true_1      50       2
Precision=0.0588, Recall=0.0385, F1=0.0465

