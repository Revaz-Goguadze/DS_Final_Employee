# Predictive Modeling for Employee Attrition: Identifying Key Drivers of Turnover

## Project Overview
This project analyzes and predicts employee attrition using the IBM HR Analytics dataset. The goal is to identify the factors most associated with turnover and build models that can flag at-risk employees.

## Solo Team
Revaz Goguadze

## Dataset
- Source: IBM HR Analytics Employee Attrition & Performance (Kaggle)
- Local file: `data/raw/ibm_hr_attrition.csv`
- Dataset link (reference):
  ```
  https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
  ```

## Objectives
- Perform data cleaning and preprocessing, including feature engineering.
- Conduct EDA with 5+ visualization types and statistical summaries.
- Train and compare Logistic Regression, Random Forest, and Gradient Boosting models.
- Evaluate models using accuracy, precision, recall, and F1-score.

## Repository Structure
- `data/`:
  - `raw/`: Original dataset
  - `processed/`: Processed dataset output
- `notebooks/`: Jupyter notebooks (exploration, preprocessing, EDA, modeling)
- `reports/`:
  - `*.png`: Static EDA and model visuals
  - `interactive_attrition_department.html`: Interactive EDA plot
  - `data_quality_report.md`: Data cleaning report
  - `data_dictionary.md`: Feature definitions
  - `results/`: Model metric outputs
- `src/`: Python scripts for data processing, EDA, and modeling
- `requirements.txt`: Python dependencies
- `project.md`: Project proposal and updated findings

## Getting Started
```bash
pip install -r requirements.txt
```

## Usage
```bash
# Generate data quality report
python src/data_analysis.py

# Create processed dataset
python src/process_data.py

# Generate EDA plots
python src/eda_visuals.py

# Generate interactive Plotly visualization
python src/interactive_visuals.py

# Train and evaluate models (saves metrics to reports/results)
python src/modeling.py

# Optional: run grid search + ablation study
python src/modeling.py --grid-search --ablation

# Generate confusion matrices + feature importance plots
python src/model_visuals.py
```

## Results Summary (latest run)
All models use SMOTE for class imbalance and an 80/20 stratified split (random_state=42).

- Logistic Regression: accuracy 0.80, attrition recall 0.64
- Random Forest: accuracy 0.85, attrition recall 0.30
- Gradient Boosting: accuracy 0.85, attrition recall 0.26
- Decision Tree: accuracy 0.84, attrition recall 0.38

## Model Selection Summary
| Model | Best CV F1 | 5-fold CV F1 (mean ± std) | Best Params |
| --- | --- | --- | --- |
| Logistic Regression | 0.530 | 0.510 ± 0.030 | `C: 0.1` |
| Random Forest | 0.499 | 0.464 ± 0.085 | `max_depth: None, min_samples_leaf: 2, min_samples_split: 2, n_estimators: 200` |
| Gradient Boosting | 0.538 | 0.530 ± 0.076 | `learning_rate: 0.1, max_depth: 3, n_estimators: 200` |
| Decision Tree | 0.394 | 0.435 ± 0.099 | `max_depth: 5, min_samples_leaf: 4, min_samples_split: 10` |

## Advanced Evaluation
- ROC and Precision-Recall curves saved in `reports/roc_curves.png` and `reports/pr_curves.png`.
- Threshold tradeoff analysis for Logistic Regression saved in `reports/threshold_tradeoff.png` and `reports/results/threshold_analysis.csv`.
- Cost-based threshold curve saved in `reports/threshold_cost_curve.png` and `reports/results/threshold_costs.csv`.
- Grid search results saved in `reports/results/*_gridsearch.csv` (run with `--grid-search`).
- Grid search summary saved in `reports/results/gridsearch_summary.csv` and CV F1 scores in `reports/results/cv_f1_scores.csv`.
- SHAP summary for Random Forest saved in `reports/shap_summary_rf.png`.
- SMOTE ablation results saved in `reports/results/ablation_smote.csv` and `reports/ablation_smote_recall.png`.
- Attrition recall: 0.64 with SMOTE vs 0.36 without SMOTE (Logistic Regression).

## Reports & Notebooks
- Data quality report: `reports/data_quality_report.md`
- Data dictionary: `reports/data_dictionary.md`
- Notebooks: `notebooks/01_data_exploration.ipynb` through `notebooks/04_machine_learning.ipynb`

## Notes
- The dataset has class imbalance (attrition is the minority class). SMOTE is used to improve recall for attrition.
- Non-informative columns (EmployeeNumber, Over18, StandardHours, EmployeeCount) are removed in modeling.
- IQR-based outlier flags (`*_outlier`) are added to the processed dataset to preserve signal without dropping rows.

## AI Assistance Disclosure
AI tools were used for learning, debugging, and testing assistance. All code and analysis were reviewed and validated by the author.

## Submission Checklist
- [ ] Run `python src/data_analysis.py` and confirm `reports/data_quality_report.md` updated
- [ ] Run `python src/process_data.py` to refresh `data/processed/ibm_hr_attrition_processed.csv`
- [ ] Run `python src/eda_visuals.py` and `python src/interactive_visuals.py`
- [ ] Run `python src/modeling.py --grid-search --ablation` and `python src/model_visuals.py`
- [ ] Re-run notebooks (`notebooks/01_*.ipynb` to `notebooks/04_*.ipynb`)
- [ ] Verify `reports/results/` contains metrics, grid search, and ablation files
- [ ] Update README + project.md if any metrics changed
