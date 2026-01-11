# Data Dictionary

This dictionary describes the columns in `data/raw/ibm_hr_attrition.csv`.

Column | Description
--- | ---
Age | Employee age in years.
Attrition | Target label indicating if the employee left (Yes/No).
BusinessTravel | Frequency of business travel.
DailyRate | Daily pay rate.
Department | Employee department.
DistanceFromHome | Distance from home to work (miles).
Education | Education level (ordinal).
EducationField | Field of education.
EmployeeCount | Constant value (all rows = 1).
EmployeeNumber | Unique employee identifier.
EnvironmentSatisfaction | Satisfaction with work environment (ordinal).
Gender | Employee gender.
HourlyRate | Hourly pay rate.
JobInvolvement | Job involvement level (ordinal).
JobLevel | Job level (ordinal).
JobRole | Employee job role.
JobSatisfaction | Job satisfaction level (ordinal).
MaritalStatus | Marital status.
MonthlyIncome | Monthly income.
MonthlyRate | Monthly pay rate.
NumCompaniesWorked | Number of companies previously worked for.
Over18 | Over 18 years old (constant in this dataset).
OverTime | Whether the employee works overtime (Yes/No).
PercentSalaryHike | Percent salary increase.
PerformanceRating | Performance rating (ordinal).
RelationshipSatisfaction | Relationship satisfaction level (ordinal).
StandardHours | Standard hours (constant in this dataset).
StockOptionLevel | Stock option level (ordinal).
TotalWorkingYears | Total years of working experience.
TrainingTimesLastYear | Number of training sessions last year.
WorkLifeBalance | Work-life balance rating (ordinal).
YearsAtCompany | Years at the current company.
YearsInCurrentRole | Years in current role.
YearsSinceLastPromotion | Years since last promotion.
YearsWithCurrManager | Years with current manager.

## Engineered Features (Processed Dataset)
- `AgeGroup`: binned age categories derived from `Age`.
- `TotalSatisfaction`: sum of satisfaction-related columns (EnvironmentSatisfaction, JobSatisfaction, RelationshipSatisfaction, JobInvolvement, WorkLifeBalance).
- `*_outlier`: IQR-based outlier flag for each numeric column (1 = outlier, 0 = typical range).