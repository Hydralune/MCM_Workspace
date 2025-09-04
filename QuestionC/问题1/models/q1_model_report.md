# 问题1 模型报告

## OLS 回归

```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:               y_target   R-squared:                       0.046
Model:                            OLS   Adj. R-squared:                  0.044
Method:                 Least Squares   F-statistic:                     25.82
Date:                Thu, 04 Sep 2025   Prob (F-statistic):           1.11e-11
Time:                        21:33:14   Log-Likelihood:                -2818.2
No. Observations:                1082   AIC:                             5642.
Df Residuals:                    1079   BIC:                             5657.
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
====================================================================================
                       coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------
Intercept           11.9461      1.116     10.701      0.000       9.756      14.137
gestational_week     0.1255      0.025      5.076      0.000       0.077       0.174
bmi                 -0.1964      0.034     -5.791      0.000      -0.263      -0.130
==============================================================================
Omnibus:                       78.339   Durbin-Watson:                   0.818
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              103.283
Skew:                           0.617   Prob(JB):                     3.73e-23
Kurtosis:                       3.875   Cond. No.                         410.
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```

## 线性混合效应模型 (随机截距: subject_id)

```
           Mixed Linear Model Regression Results
===========================================================
Model:             MixedLM  Dependent Variable:  y_target  
No. Observations:  1082     Method:              ML        
No. Groups:        267      Scale:               2.9254    
Min. group size:   1        Log-Likelihood:      -2448.1632
Max. group size:   8        Converged:           Yes       
Mean group size:   4.1                                     
-----------------------------------------------------------
                 Coef.  Std.Err.   z    P>|z| [0.025 0.975]
-----------------------------------------------------------
Intercept         6.971    1.611  4.328 0.000  3.814 10.127
gestational_week  0.313    0.016 19.447 0.000  0.281  0.344
bmi              -0.138    0.052 -2.660 0.008 -0.239 -0.036
Group Var         8.227    0.529                           
===========================================================

```

## 达标孕周 t_min (预测4%)

### OLS 估计

| bmi_group   |   bmi_repr |   t_min_week |
|:------------|-----------:|-------------:|
| [0,20)      |   nan      |    nan       |
| [20,28)     |    27.6398 |    -20.0501  |
| [28,32)     |    30.3715 |    -15.7759  |
| [32,36)     |    33.4882 |    -10.8995  |
| [36,40)     |    36.9572 |     -5.47193 |
| 40+         |    41.9531 |      2.34485 |

### MixedLM 估计

| bmi_group   |   bmi_repr |   t_min_week |
|:------------|-----------:|-------------:|
| [0,20)      |   nan      |    nan       |
| [20,28)     |    27.6398 |      2.68639 |
| [28,32)     |    30.3715 |      3.89121 |
| [32,36)     |    33.4882 |      5.26581 |
| [36,40)     |    36.9572 |      6.79577 |
| 40+         |    41.9531 |      8.99921 |