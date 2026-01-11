# Data Quality Report

## Dataset Snapshot
- Rows: 1470
- Columns: 35
- Duplicate rows: 0

## Missing Values
- No missing values detected.

## Target Distribution (Attrition)
Class | Count | Rate
--- | --- | ---
No | 1233 | 0.839
Yes | 237 | 0.161

## Outliers (IQR Method)
Column | Outlier Count
--- | ---
TrainingTimesLastYear | 238
PerformanceRating | 226
MonthlyIncome | 114
YearsSinceLastPromotion | 107
YearsAtCompany | 104
StockOptionLevel | 85
TotalWorkingYears | 63
NumCompaniesWorked | 52
YearsInCurrentRole | 21
YearsWithCurrManager | 14

## Cleaning Decisions
- No missing values were found, so no imputation or row drops were required.
- Duplicate rows were checked and none were removed.
- Outliers were retained to preserve potential signal in employee attrition behavior.
- Outlier indicator flags (IQR-based) are added to the processed dataset for modeling.
- Non-informative columns (EmployeeNumber, Over18, StandardHours, EmployeeCount) are removed in modeling.
- Derived features are created in the modeling pipeline (AgeGroup, TotalSatisfaction).