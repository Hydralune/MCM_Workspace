# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a mathematical modeling project for NIPT (Non-invasive Prenatal Test) timing optimization and fetal abnormality detection. The project involves analyzing fetal chromosome concentration data to determine optimal testing times based on maternal BMI and other factors.

## Key Data Files

- `附件.csv` / `附件.xlsx`: Raw NIPT detection data with 30+ columns including maternal demographics, chromosome Z-scores, and concentration values
- `问题1/q1_male_cleaned.csv`: Cleaned male fetus data with 1082 records (subject_id, maternal_age, height, weight, bmi, gestational_week, y_concentration, y_zscore)
- `问题1/q1_male_stats.csv`: Statistical summary of male fetus data

## Core Analysis Tasks

1. **Problem 1**: Model relationship between Y-chromosome concentration vs gestational week and BMI
2. **Problem 2**: Optimize BMI grouping and determine best NIPT timing for each group
3. **Problem 3**: Multi-factor optimization including height, weight, age, BMI
4. **Problem 4**: Female fetus abnormality detection using chromosome Z-scores and other indicators

## Data Processing Pipeline

### Key Transformations
- Gestational week conversion: "11w+6" → 11.857 weeks (weeks + days/7)
- BMI calculation: weight(kg) / (height(m))^2
- Missing value handling: Female fetuses have blank Y-chromosome fields
- Repeated measurements: Same subject_id indicates longitudinal data

### Critical Data Fields
- **V列 (Y染色体浓度)**: Target variable for problems 1-3
- **K列 (BMI)**: Primary predictor for grouping
- **J列 (孕周)**: Key temporal variable
- **AB列 (染色体非整倍体)**: Target for problem 4

## Modeling Approaches

### Statistical Models
- Linear mixed-effects models (for repeated measurements)
- Generalized additive models (GAM) for non-linear relationships
- Logistic regression/random forests for classification (problem 4)

### Optimization Strategy
- Risk quantification: Early=1, Mid=5, Late=10 points
- Constraint: Y-chromosome concentration ≥ 4%
- BMI grouping: [20,28), [28,32), [32,36), [36,40), 40+

## Common Development Commands

```bash
# Data exploration
python -c "import pandas as pd; df = pd.read_csv('问题1/q1_male_cleaned.csv'); print(df.describe())"

# Quick visualization
python -c "import matplotlib.pyplot as plt; import pandas as pd; df = pd.read_csv('问题1/q1_male_cleaned.csv'); plt.scatter(df['gestational_week'], df['y_concentration']); plt.xlabel('Gestational Week'); plt.ylabel('Y Concentration'); plt.show()"

# Check data quality
python -c "import pandas as pd; df = pd.read_csv('问题1/q1_male_cleaned.csv'); print(df.isnull().sum())"
```

## File Encoding Notes

- Raw data files use GBK encoding (Chinese characters)
- Processed files use UTF-8 encoding
- Use `encoding='gbk'` when reading original CSV files

## Mathematical Modeling Context

This is a competition-style mathematical modeling project that requires:
- Statistical analysis using Python/R
- Optimization modeling
- Machine learning for classification
- Clear documentation of methodology and results
- Focus on practical clinical applicability