from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

from modeling import load_and_preprocess

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 6)

def plot_confusion_matrix(y_true, y_pred, title, filename, output_dir: Path) -> None:
    """Save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Attrition', 'Attrition'],
                yticklabels=['No Attrition', 'Attrition'])
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(output_dir / filename)
    plt.close()

def plot_feature_importance(model, feature_names, title, filename, output_dir: Path) -> None:
    """Save a bar chart of feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = 15
    
    plt.figure(figsize=(12, 8))
    plt.title(title)
    plt.barh(range(top_n), importances[indices[:top_n]][::-1], align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()

def plot_logistic_coefficients(coefficients, feature_names, output_dir: Path) -> None:
    """Plot top absolute logistic regression coefficients."""
    abs_coeffs = np.abs(coefficients)
    top_n = 15
    indices = np.argsort(abs_coeffs)[-top_n:]
    sorted_coeffs = coefficients[indices]
    sorted_features = [feature_names[i] for i in indices]
    colors = ["#d62728" if val < 0 else "#1f77b4" for val in sorted_coeffs]

    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), sorted_coeffs, color=colors)
    plt.yticks(range(top_n), sorted_features)
    plt.xlabel("Coefficient Value")
    plt.title("Top Logistic Regression Coefficients (Absolute)")
    plt.tight_layout()
    plt.savefig(output_dir / "logistic_coefficients.png")
    plt.close()


def plot_roc_pr_curves(model_scores, y_true, output_dir: Path) -> None:
    """Plot ROC and Precision-Recall curves for multiple models."""
    plt.figure(figsize=(10, 7))
    for name, scores in model_scores.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curves.png")
    plt.close()

    plt.figure(figsize=(10, 7))
    for name, scores in model_scores.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = average_precision_score(y_true, scores)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_dir / "pr_curves.png")
    plt.close()


def plot_shap_summary(model, sample_data, feature_names, output_dir: Path) -> None:
    """Generate a SHAP summary plot for a tree-based model."""
    try:
        import shap
    except Exception as exc:
        print(f"SHAP not available: {exc}")
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(sample_data)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        shap.summary_plot(
            shap_values,
            sample_data,
            feature_names=feature_names,
            show=False,
            max_display=15,
        )
        plt.tight_layout()
        plt.savefig(output_dir / "shap_summary_rf.png")
        plt.close()
    except Exception as exc:
        print(f"Unable to generate SHAP summary: {exc}")


def threshold_analysis(y_true, scores, output_dir: Path) -> None:
    """Evaluate precision/recall/F1 across thresholds for a single model."""
    thresholds = np.linspace(0.1, 0.9, 17)
    rows = []
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "precision": precision_score(y_true, preds, zero_division=0),
                "recall": recall_score(y_true, preds, zero_division=0),
                "f1_score": f1_score(y_true, preds, zero_division=0),
            }
        )
    results = pd.DataFrame(rows)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "threshold_analysis.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(results["threshold"], results["precision"], label="Precision")
    plt.plot(results["threshold"], results["recall"], label="Recall")
    plt.plot(results["threshold"], results["f1_score"], label="F1 Score")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold Tradeoff (Logistic Regression)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_tradeoff.png")
    plt.close()


def cost_threshold_analysis(
    y_true, scores, output_dir: Path, fn_cost: float = 5.0, fp_cost: float = 1.0
) -> None:
    """Estimate expected cost across thresholds given FN/FP costs."""
    thresholds = np.linspace(0.05, 0.95, 19)
    rows = []
    for threshold in thresholds:
        preds = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        total_cost = fn * fn_cost + fp * fp_cost
        rows.append(
            {
                "threshold": float(threshold),
                "false_negatives": int(fn),
                "false_positives": int(fp),
                "total_cost": float(total_cost),
            }
        )
    results = pd.DataFrame(rows)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "threshold_costs.csv", index=False)

    plt.figure(figsize=(10, 6))
    plt.plot(results["threshold"], results["total_cost"], marker="o")
    best_idx = results["total_cost"].idxmin()
    best_thr = results.loc[best_idx, "threshold"]
    best_cost = results.loc[best_idx, "total_cost"]
    plt.axvline(best_thr, color="#d62728", linestyle="--", label=f"Best threshold={best_thr:.2f}")
    plt.title("Estimated Cost vs Threshold (LR)")
    plt.xlabel("Threshold")
    plt.ylabel("Estimated Cost (FN*5 + FP*1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_cost_curve.png")
    plt.close()


def plot_ablation_results(output_dir: Path) -> None:
    """Plot SMOTE vs no-SMOTE comparison for Logistic Regression."""
    ablation_path = output_dir / "results" / "ablation_smote.csv"
    if not ablation_path.exists():
        return
    ablation = pd.read_csv(ablation_path)
    class_one = ablation[ablation["label"] == "1"].copy()
    if class_one.empty:
        return
    plt.figure(figsize=(8, 6))
    plt.bar(
        class_one["model"],
        class_one["recall"],
        color=["#1f77b4", "#ff7f0e"],
    )
    plt.title("Attrition Recall: SMOTE vs No SMOTE (Logistic Regression)")
    plt.ylabel("Recall")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_smote_recall.png")
    plt.close()


def run_visualizations(data_path: Path, output_dir: Path) -> None:
    try:
        print("Loading and processing data...")
        X, y, categorical_cols, numerical_cols = load_and_preprocess(data_path)
    except Exception as exc:
        print(f"Error loading data: {exc}")
        return
    
    # Preprocessing setup
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Models
    summary_path = output_dir / "results" / "gridsearch_summary.csv"
    best_params: dict[str, dict] = {}
    if summary_path.exists():
        try:
            summary_df = pd.read_csv(summary_path)
            for _, row in summary_df.iterrows():
                model_name = row["model"]
                params = row["best_params"]
                if isinstance(params, str):
                    params = ast.literal_eval(params)
                best_params[model_name] = params
        except Exception as exc:
            print(f"Error reading grid search summary: {exc}")

    print("Training Logistic Regression...")
    lr_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    if "logistic_regression" in best_params:
        lr_pipeline.set_params(**best_params["logistic_regression"])
    try:
        lr_pipeline.fit(X_train, y_train)
        y_pred_lr = lr_pipeline.predict(X_test)
    except Exception as exc:
        print(f"Error training Logistic Regression: {exc}")
        return

    print("Training Random Forest...")
    rf_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    if "random_forest" in best_params:
        rf_pipeline.set_params(**best_params["random_forest"])
    try:
        rf_pipeline.fit(X_train, y_train)
        y_pred_rf = rf_pipeline.predict(X_test)
    except Exception as exc:
        print(f"Error training Random Forest: {exc}")
        return

    print("Training Gradient Boosting...")
    gb_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    if "gradient_boosting" in best_params:
        gb_pipeline.set_params(**best_params["gradient_boosting"])
    try:
        gb_pipeline.fit(X_train, y_train)
        y_pred_gb = gb_pipeline.predict(X_test)
    except Exception as exc:
        print(f"Error training Gradient Boosting: {exc}")
        return

    print("Training Decision Tree...")
    dt_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5))
    ])
    if "decision_tree" in best_params:
        dt_pipeline.set_params(**best_params["decision_tree"])
    try:
        dt_pipeline.fit(X_train, y_train)
        y_pred_dt = dt_pipeline.predict(X_test)
    except Exception as exc:
        print(f"Error training Decision Tree: {exc}")
        return
    
    print("Generating plots...")
    # Confusion Matrices
    plot_confusion_matrix(y_test, y_pred_lr, 'Confusion Matrix - Logistic Regression', 'cm_logistic_regression.png', output_dir)
    plot_confusion_matrix(y_test, y_pred_rf, 'Confusion Matrix - Random Forest', 'cm_random_forest.png', output_dir)
    plot_confusion_matrix(y_test, y_pred_gb, 'Confusion Matrix - Gradient Boosting', 'cm_gradient_boosting.png', output_dir)
    plot_confusion_matrix(y_test, y_pred_dt, 'Confusion Matrix - Decision Tree', 'cm_decision_tree.png', output_dir)
    
    # Feature Importance (Random Forest only)
    # Need to extract feature names from preprocessor
    ohe = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat']
    cat_feature_names = ohe.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numerical_cols + cat_feature_names
    
    rf_model = rf_pipeline.named_steps['classifier']
    plot_feature_importance(rf_model, all_feature_names, 'Top 15 Feature Importances (Random Forest)', 'feature_importance_rf.png', output_dir)

    lr_model = lr_pipeline.named_steps['classifier']
    plot_logistic_coefficients(lr_model.coef_[0], all_feature_names, output_dir)

    # ROC and PR curves
    model_scores = {
        "Logistic Regression": lr_pipeline.predict_proba(X_test)[:, 1],
        "Random Forest": rf_pipeline.predict_proba(X_test)[:, 1],
        "Gradient Boosting": gb_pipeline.predict_proba(X_test)[:, 1],
        "Decision Tree": dt_pipeline.predict_proba(X_test)[:, 1],
    }
    plot_roc_pr_curves(model_scores, y_test, output_dir)

    # Threshold analysis for Logistic Regression
    threshold_analysis(y_test, model_scores["Logistic Regression"], output_dir)
    cost_threshold_analysis(y_test, model_scores["Logistic Regression"], output_dir)

    # Ablation plot for SMOTE vs no-SMOTE (if available)
    plot_ablation_results(output_dir)

    # SHAP summary for Random Forest
    preprocessor_fitted = rf_pipeline.named_steps['preprocessor']
    X_train_transformed = preprocessor_fitted.transform(X_train)
    if hasattr(X_train_transformed, "toarray"):
        X_train_transformed = X_train_transformed.toarray()
    rng = np.random.RandomState(42)
    sample_size = min(200, X_train_transformed.shape[0])
    sample_idx = rng.choice(X_train_transformed.shape[0], size=sample_size, replace=False)
    shap_sample = X_train_transformed[sample_idx]
    plot_shap_summary(rf_model, shap_sample, all_feature_names, output_dir)
    
    print(f"Visualizations saved to {output_dir}")


def _default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "raw" / "ibm_hr_attrition.csv", root / "reports"


def main() -> None:
    default_input, default_output = _default_paths()
    parser = argparse.ArgumentParser(description="Generate model evaluation visuals.")
    parser.add_argument("--input", default=str(default_input), help="Path to raw CSV.")
    parser.add_argument("--output-dir", default=str(default_output), help="Directory for outputs.")
    args = parser.parse_args()

    raw_data_path = Path(args.input)
    report_dir = Path(args.output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    if raw_data_path.exists():
        run_visualizations(raw_data_path, report_dir)
    else:
        print(f"Error: Data not found at {raw_data_path}")


if __name__ == "__main__":
    main()
