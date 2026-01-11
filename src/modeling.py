from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

def load_and_preprocess(file_path: Path) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Load raw data and perform feature engineering for modeling."""
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise RuntimeError(f"Unable to read dataset: {exc}") from exc
    
    # Feature Engineering
    # 1. Age Grouping
    df['AgeGroup'] = pd.cut(df['Age'], bins=[18, 25, 35, 45, 55, 65], labels=['18-25', '26-35', '36-45', '46-55', '56+'])
    
    # 2. Total Satisfaction
    # Summing up satisfaction scores if they exist
    satisfaction_cols = ['EnvironmentSatisfaction', 'JobSatisfaction', 'RelationshipSatisfaction', 'JobInvolvement', 'WorkLifeBalance']
    # Check if cols exist (WorkLifeBalance is in dataset based on head command earlier)
    available_sat_cols = [c for c in satisfaction_cols if c in df.columns]
    if available_sat_cols:
        df['TotalSatisfaction'] = df[available_sat_cols].sum(axis=1)

    # Drop irrelevant columns (EmployeeNumber is just an ID, Over18 is 'Y' for all, StandardHours is 80 for all, EmployeeCount is 1 for all)
    cols_to_drop = ['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount']
    df = df.drop(columns=cols_to_drop)

    # Outlier flags (IQR method) for numeric columns
    numeric_cols_for_flags = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols_for_flags:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        df[f"{col}_outlier"] = ((df[col] < lower) | (df[col] > upper)).astype(int)
    
    # Define Target and Features
    X = df.drop(columns=['Attrition'])
    y = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
    
    # Identify Categorical and Numerical Columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    return X, y, categorical_cols, numerical_cols

def _save_metrics(metrics: dict[str, dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for model_name, report in metrics.items():
        for label in ["0", "1", "macro avg", "weighted avg"]:
            if label not in report:
                continue
            rows.append(
                {
                    "model": model_name,
                    "label": label,
                    "precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1_score": report[label]["f1-score"],
                    "support": report[label]["support"],
                }
            )
    output_path = output_dir / "model_metrics.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return output_path


def build_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    categorical_cols: list[str],
    numerical_cols: list[str],
    output_dir: Path | None = None,
    run_grid_search: bool = False,
    run_ablation: bool = False,
) -> tuple[ImbPipeline, ImbPipeline, ImbPipeline, ImbPipeline, dict[str, dict]]:
    """Train multiple models, evaluate them, and optionally save metrics."""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Define preprocessing steps
    # Note: Using handle_unknown='ignore' for OneHotEncoder to be safe in production scenarios
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])
    
    # 1. Logistic Regression with SMOTE
    lr_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    # 2. Random Forest with SMOTE
    rf_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 3. Gradient Boosting (sklearn) with SMOTE
    gb_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])

    # 4. Decision Tree with SMOTE
    dt_pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('classifier', DecisionTreeClassifier(random_state=42, max_depth=5))
    ])

    if run_grid_search and output_dir is not None:
        results_path = output_dir / "results"
        results_path.mkdir(parents=True, exist_ok=True)

        best_params: dict[str, dict] = {}
        best_scores: dict[str, float] = {}

        search_jobs = [
            (
                "logistic_regression",
                lr_pipeline,
                {
                    "classifier__C": [0.1, 1.0, 10.0],
                },
            ),
            (
                "random_forest",
                rf_pipeline,
                {
                    "classifier__n_estimators": [100, 200],
                    "classifier__max_depth": [None, 10],
                    "classifier__min_samples_split": [2, 5],
                    "classifier__min_samples_leaf": [1, 2],
                },
            ),
            (
                "gradient_boosting",
                gb_pipeline,
                {
                    "classifier__n_estimators": [100, 200],
                    "classifier__learning_rate": [0.05, 0.1],
                    "classifier__max_depth": [2, 3],
                },
            ),
            (
                "decision_tree",
                dt_pipeline,
                {
                    "classifier__max_depth": [3, 5, 7, None],
                    "classifier__min_samples_split": [2, 5, 10],
                    "classifier__min_samples_leaf": [1, 2, 4],
                },
            ),
        ]

        for name, pipeline, param_grid in search_jobs:
            print(f"\n--- {name.replace('_', ' ').title()} Grid Search (SMOTE) ---")
            search = GridSearchCV(
                pipeline,
                param_grid,
                cv=3,
                scoring="f1",
                n_jobs=-1,
            )
            search.fit(X_train, y_train)
            print(f"Best params: {search.best_params_}")
            print(f"Best CV F1: {search.best_score_:.3f}")
            best_params[name] = search.best_params_
            best_scores[name] = float(search.best_score_)
            pd.DataFrame(search.cv_results_).to_csv(
                results_path / f"{name}_gridsearch.csv", index=False
            )

        summary_rows = []
        for name in best_params:
            summary_rows.append(
                {
                    "model": name,
                    "best_cv_f1": best_scores[name],
                    "best_params": best_params[name],
                }
            )
        pd.DataFrame(summary_rows).to_csv(
            results_path / "gridsearch_summary.csv", index=False
        )

        lr_pipeline.set_params(**best_params["logistic_regression"])
        rf_pipeline.set_params(**best_params["random_forest"])
        gb_pipeline.set_params(**best_params["gradient_boosting"])
        dt_pipeline.set_params(**best_params["decision_tree"])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_rows = []
        for name, pipeline in [
            ("logistic_regression", lr_pipeline),
            ("random_forest", rf_pipeline),
            ("gradient_boosting", gb_pipeline),
            ("decision_tree", dt_pipeline),
        ]:
            scores = cross_val_score(
                pipeline, X_train, y_train, scoring="f1", cv=cv, n_jobs=-1
            )
            cv_rows.append(
                {
                    "model": name,
                    "cv_f1_mean": float(scores.mean()),
                    "cv_f1_std": float(scores.std()),
                }
            )
        pd.DataFrame(cv_rows).to_csv(
            results_path / "cv_f1_scores.csv", index=False
        )
    
    # Train and Evaluate Logistic Regression
    print("\n--- Logistic Regression (with SMOTE) ---")
    lr_pipeline.fit(X_train, y_train)
    y_pred_lr = lr_pipeline.predict(X_test)
    lr_report = classification_report(y_test, y_pred_lr, output_dict=True)
    print(classification_report(y_test, y_pred_lr))
    
    # Train and Evaluate Random Forest
    print("\n--- Random Forest (with SMOTE) ---")
    rf_pipeline.fit(X_train, y_train)
    y_pred_rf = rf_pipeline.predict(X_test)
    rf_report = classification_report(y_test, y_pred_rf, output_dict=True)
    print(classification_report(y_test, y_pred_rf))

    # Train and Evaluate Gradient Boosting
    print("\n--- Gradient Boosting (with SMOTE) ---")
    gb_pipeline.fit(X_train, y_train)
    y_pred_gb = gb_pipeline.predict(X_test)
    gb_report = classification_report(y_test, y_pred_gb, output_dict=True)
    print(classification_report(y_test, y_pred_gb))

    print("\n--- Decision Tree (with SMOTE) ---")
    dt_pipeline.fit(X_train, y_train)
    y_pred_dt = dt_pipeline.predict(X_test)
    dt_report = classification_report(y_test, y_pred_dt, output_dict=True)
    print(classification_report(y_test, y_pred_dt))

    metrics = {
        "logistic_regression": lr_report,
        "random_forest": rf_report,
        "gradient_boosting": gb_report,
        "decision_tree": dt_report,
    }

    if run_ablation and output_dir is not None:
        print("\n--- Logistic Regression (no SMOTE) ---")
        lr_params = lr_pipeline.named_steps["classifier"].get_params()
        lr_no_smote = SkPipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42, C=lr_params.get("C", 1.0))),
        ])
        lr_no_smote.fit(X_train, y_train)
        y_pred_lr_no = lr_no_smote.predict(X_test)
        lr_no_report = classification_report(y_test, y_pred_lr_no, output_dict=True)
        print(classification_report(y_test, y_pred_lr_no))

        ablation_rows = []
        for name, report in {
            "logistic_regression_smote": lr_report,
            "logistic_regression_no_smote": lr_no_report,
        }.items():
            for label in ["0", "1", "macro avg", "weighted avg"]:
                if label not in report:
                    continue
                ablation_rows.append(
                    {
                        "model": name,
                        "label": label,
                        "precision": report[label]["precision"],
                        "recall": report[label]["recall"],
                        "f1_score": report[label]["f1-score"],
                        "support": report[label]["support"],
                    }
                )
        ablation_path = output_dir / "results"
        ablation_path.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(ablation_rows).to_csv(
            ablation_path / "ablation_smote.csv", index=False
        )

    if output_dir is not None:
        results_dir = output_dir / "results"
        output_path = _save_metrics(metrics, results_dir)
        print(f"Saved metrics to {output_path}")

    return lr_pipeline, rf_pipeline, gb_pipeline, dt_pipeline, metrics


def _default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "raw" / "ibm_hr_attrition.csv", root / "reports"


def main() -> None:
    default_input, default_output = _default_paths()
    parser = argparse.ArgumentParser(description="Train and evaluate ML models for attrition.")
    parser.add_argument("--input", default=str(default_input), help="Path to raw CSV.")
    parser.add_argument("--output-dir", default=str(default_output), help="Directory for outputs.")
    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Run small GridSearchCV sweeps for all models.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Run SMOTE vs no-SMOTE ablation for Logistic Regression.",
    )
    args = parser.parse_args()

    data_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if data_path.exists():
        try:
            X, y, categorical_cols, numerical_cols = load_and_preprocess(data_path)
            build_and_evaluate(
                X,
                y,
                categorical_cols,
                numerical_cols,
                output_dir=output_dir,
                run_grid_search=args.grid_search,
                run_ablation=args.ablation,
            )
        except Exception as exc:
            print(f"Error during modeling: {exc}")
    else:
        print(f"Error: Data not found at {data_path}")


if __name__ == "__main__":
    main()
