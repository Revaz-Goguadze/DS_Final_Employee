from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px

def generate_interactive_plot(file_path: Path, output_dir: Path) -> None:
    """Generate an interactive Plotly bar chart for attrition by department."""
    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        print(f"Error reading dataset: {exc}")
        return
    
    # Calculate Attrition rates by Department
    attrition_counts = df.groupby(['Department', 'Attrition']).size().reset_index(name='Count')
    
    fig = px.bar(attrition_counts, 
                 x='Department', 
                 y='Count', 
                 color='Attrition',
                 title='Interactive: Attrition by Department',
                 barmode='group',
                 text='Count',
                 color_discrete_map={'No': '#1f77b4', 'Yes': '#d62728'})
    
    fig.update_layout(template='plotly_white')
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'interactive_attrition_department.html'
        fig.write_html(output_path)
        print(f"Interactive plot saved to {output_path}")
    except Exception as exc:
        print(f"Error saving interactive plot: {exc}")

def _default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "raw" / "ibm_hr_attrition.csv", root / "reports"


def main() -> None:
    default_input, default_output = _default_paths()
    parser = argparse.ArgumentParser(description="Generate interactive EDA plot.")
    parser.add_argument("--input", default=str(default_input), help="Path to raw CSV.")
    parser.add_argument("--output-dir", default=str(default_output), help="Directory for output.")
    args = parser.parse_args()

    raw_data_path = Path(args.input)
    report_dir = Path(args.output_dir)

    if raw_data_path.exists():
        generate_interactive_plot(raw_data_path, report_dir)
    else:
        print(f"Error: Data not found at {raw_data_path}")


if __name__ == "__main__":
    main()
