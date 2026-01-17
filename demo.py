#!/usr/bin/env python3
"""Run or describe the live demo steps for the attrition project."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_step(command: list[str]) -> None:
    print(f"\n$ {' '.join(command)}")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live demo helper for the project.")
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute demo commands instead of only printing them.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    commands = [
        ["python", str(root / "src" / "eda_visuals.py")],
        ["python", str(root / "src" / "interactive_visuals.py")],
        [
            "python",
            str(root / "src" / "modeling.py"),
            "--grid-search",
            "--ablation",
        ],
        ["python", str(root / "src" / "model_visuals.py")],
    ]

    print("Live demo steps:")
    for command in commands:
        print(f"- {' '.join(command)}")

    if not args.run:
        print("\nRun with --run to execute the commands.")
        return

    for command in commands:
        run_step(command)


if __name__ == "__main__":
    main()
