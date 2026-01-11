from __future__ import annotations

import argparse
from pathlib import Path

from modeling import load_and_preprocess

def save_processed_data(file_path: Path, output_dir: Path) -> Path | None:
    """Load raw data, apply preprocessing, and save a processed CSV."""
    try:
        print(f"Processing data from {file_path}...")
        X, y, _categorical_cols, _numerical_cols = load_and_preprocess(file_path)

        # Combine X and y for the processed file
        processed_df = X.copy()
        processed_df["Attrition"] = y

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "ibm_hr_attrition_processed.csv"
        processed_df.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")
        print(f"Shape: {processed_df.shape}")
        return output_path
    except Exception as exc:
        print(f"Error processing data: {exc}")
        return None

def _default_paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return root / "data" / "raw" / "ibm_hr_attrition.csv", root / "data" / "processed"


def main() -> None:
    default_input, default_output = _default_paths()
    parser = argparse.ArgumentParser(description="Process raw HR data into a clean CSV.")
    parser.add_argument("--input", default=str(default_input), help="Path to raw CSV.")
    parser.add_argument("--output-dir", default=str(default_output), help="Directory for output.")
    args = parser.parse_args()

    raw_data_path = Path(args.input)
    processed_dir = Path(args.output_dir)

    if raw_data_path.exists():
        save_processed_data(raw_data_path, processed_dir)
    else:
        print(f"Error: File not found at {raw_data_path}")


if __name__ == "__main__":
    main()
