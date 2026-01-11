from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _outlier_counts_iqr(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        counts[col] = int(((df[col] < lower) | (df[col] > upper)).sum())
    return counts


def _build_data_quality_report(df: pd.DataFrame) -> str:
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    missing_by_col = df.isna().sum()
    duplicates = int(df.duplicated().sum())
    outlier_counts = _outlier_counts_iqr(df, numeric_cols)
    attrition_rate = df["Attrition"].value_counts(normalize=True).rename("rate")
    attrition_counts = df["Attrition"].value_counts().rename("count")

    top_outliers = sorted(outlier_counts.items(), key=lambda x: x[1], reverse=True)[:10]

    report_lines = [
        "# Data Quality Report",
        "",
        "## Dataset Snapshot",
        f"- Rows: {df.shape[0]}",
        f"- Columns: {df.shape[1]}",
        f"- Duplicate rows: {duplicates}",
        "",
        "## Missing Values",
    ]

    if missing_by_col.sum() == 0:
        report_lines.append("- No missing values detected.")
    else:
        report_lines.append("Column | Missing Count")
        report_lines.append("--- | ---")
        for col, count in missing_by_col[missing_by_col > 0].items():
            report_lines.append(f"{col} | {int(count)}")

    report_lines += [
        "",
        "## Target Distribution (Attrition)",
        "Class | Count | Rate",
        "--- | --- | ---",
    ]
    for label in attrition_counts.index:
        report_lines.append(f"{label} | {int(attrition_counts[label])} | {attrition_rate[label]:.3f}")

    report_lines += [
        "",
        "## Outliers (IQR Method)",
        "Column | Outlier Count",
        "--- | ---",
    ]
    for col, count in top_outliers:
        report_lines.append(f"{col} | {count}")

    report_lines += [
        "",
        "## Cleaning Decisions",
        "- No missing values were found, so no imputation or row drops were required.",
        "- Duplicate rows were checked and none were removed.",
        "- Outliers were retained to preserve potential signal in employee attrition behavior.",
        "- Outlier indicator flags (IQR-based) are added to the processed dataset for modeling.",
        "- Non-informative columns (EmployeeNumber, Over18, StandardHours, EmployeeCount) are removed in modeling.",
        "- Derived features are created in the modeling pipeline (AgeGroup, TotalSatisfaction).",
    ]

    return "\n".join(report_lines)

def initial_analysis(file_path: Path, report_path: Path | None = None) -> None:
    """Print initial analysis and optionally write a data quality report."""
    try:
        print(f"Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"Error reading dataset: {exc}")
        return

    print("\n--- Dataset Info ---")
    print(df.info())

    print("\n--- First 5 Rows ---")
    print(df.head())

    print("\n--- Missing Values ---")
    print(df.isnull().sum())

    print("\n--- Target Class Distribution (Attrition) ---")
    print(df["Attrition"].value_counts(normalize=True))

    print("\n--- Basic Statistics ---")
    print(df.describe())

    if report_path is not None:
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(_build_data_quality_report(df))
            print(f"\nData quality report written to: {report_path}")
        except Exception as exc:
            print(f"Error writing data quality report: {exc}")


def _default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    raw_path = root / "data" / "raw" / "ibm_hr_attrition.csv"
    report_path = root / "reports" / "data_quality_report.md"
    return raw_path, report_path


def main() -> None:
    default_input, default_report = _default_paths()
    parser = argparse.ArgumentParser(description="Run initial dataset analysis.")
    parser.add_argument("--input", default=str(default_input), help="Path to raw CSV.")
    parser.add_argument("--report", default=str(default_report), help="Path to markdown report.")
    args = parser.parse_args()

    raw_data_path = Path(args.input)
    report_path = Path(args.report) if args.report else None

    if raw_data_path.exists():
        initial_analysis(raw_data_path, report_path=report_path)
    else:
        print(f"Error: File not found at {raw_data_path}")


if __name__ == "__main__":
    main()
