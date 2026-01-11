from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (12, 8)

def run_eda(file_path: Path, output_dir: Path) -> None:
    """Generate EDA visuals and save them to the output directory."""
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"Error reading dataset: {exc}")
        return

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Error creating output directory: {exc}")
        return

    # 1. Attrition by Department (Bar Plot)
    plt.figure()
    sns.countplot(data=df, x='Department', hue='Attrition')
    plt.title('Attrition by Department')
    try:
        plt.savefig(output_dir / 'attrition_department.png')
    except Exception as exc:
        print(f"Error saving attrition_department.png: {exc}")
    plt.close()
    
    # 2. Age Distribution vs Attrition (KDE Plot)
    plt.figure()
    sns.kdeplot(data=df, x='Age', hue='Attrition', fill=True, common_norm=False)
    plt.title('Age Distribution by Attrition')
    try:
        plt.savefig(output_dir / 'age_distribution.png')
    except Exception as exc:
        print(f"Error saving age_distribution.png: {exc}")
    plt.close()
    
    # 3. Monthly Income vs Job Level (Box Plot)
    plt.figure()
    sns.boxplot(data=df, x='JobLevel', y='MonthlyIncome', hue='Attrition')
    plt.title('Monthly Income vs Job Level by Attrition')
    try:
        plt.savefig(output_dir / 'income_joblevel_box.png')
    except Exception as exc:
        print(f"Error saving income_joblevel_box.png: {exc}")
    plt.close()
    
    # 4. Overtime vs Attrition (Count Plot)
    plt.figure()
    sns.countplot(data=df, x='OverTime', hue='Attrition')
    plt.title('Overtime Impact on Attrition')
    try:
        plt.savefig(output_dir / 'overtime_attrition.png')
    except Exception as exc:
        print(f"Error saving overtime_attrition.png: {exc}")
    plt.close()
    
    # 5. Job Satisfaction vs Attrition (Heatmap)
    plt.figure()
    ct = pd.crosstab(df['JobSatisfaction'], df['Attrition'], normalize='index')
    sns.heatmap(ct, annot=True, cmap='YlGnBu')
    plt.title('Job Satisfaction Heatmap (Normalized)')
    try:
        plt.savefig(output_dir / 'satisfaction_heatmap.png')
    except Exception as exc:
        print(f"Error saving satisfaction_heatmap.png: {exc}")
    plt.close()

    # 6. Correlation Heatmap (Numerical Features)
    plt.figure(figsize=(12, 10))
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.tight_layout()
    try:
        plt.savefig(output_dir / "correlation_heatmap.png")
    except Exception as exc:
        print(f"Error saving correlation_heatmap.png: {exc}")
    plt.close()

    # 7. Scatter Plot with Trend Line
    scatter = sns.lmplot(
        data=df,
        x="Age",
        y="MonthlyIncome",
        hue="Attrition",
        height=6,
        aspect=1.3,
        scatter_kws={"alpha": 0.6, "s": 30},
        line_kws={"linewidth": 2},
    )
    scatter.fig.suptitle("Age vs Monthly Income (with Trend)")
    scatter.fig.tight_layout()
    try:
        scatter.fig.savefig(output_dir / "age_income_scatter.png")
    except Exception as exc:
        print(f"Error saving age_income_scatter.png: {exc}")
    plt.close(scatter.fig)
    
    print(f"EDA plots saved to {output_dir}")

def _default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "raw" / "ibm_hr_attrition.csv", root / "reports"


def main() -> None:
    default_input, default_output = _default_paths()
    parser = argparse.ArgumentParser(description="Generate EDA plots for the HR dataset.")
    parser.add_argument("--input", default=str(default_input), help="Path to raw CSV.")
    parser.add_argument("--output-dir", default=str(default_output), help="Directory for plots.")
    args = parser.parse_args()

    raw_data_path = Path(args.input)
    report_dir = Path(args.output_dir)

    if raw_data_path.exists():
        run_eda(raw_data_path, report_dir)
    else:
        print(f"Error: File not found at {raw_data_path}")


if __name__ == "__main__":
    main()
