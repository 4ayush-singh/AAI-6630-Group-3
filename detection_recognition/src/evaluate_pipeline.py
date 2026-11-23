"""Score the pipeline's predictions against the course ground truth.

Usage (from the project root, after src/run_pipeline.py):
    python src/evaluate_pipeline.py --ground_truth pdfs/ground_truth.csv

Reads pipeline_out/predictions.csv and writes, in the same folder:
    evaluation_summary.csv   precision, recall and F1 per paper and overall
    matches.csv              each predicted molecule paired with its best true molecule
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.evaluate import evaluate  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Evaluate pipeline predictions against ground truth.")
    p.add_argument("--ground_truth", required=True, help="CSV with paperID, name and SMILES columns")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = p.parse_args()
    with open(args.config) as f:
        pcfg = yaml.safe_load(f).get("pipeline", {})
    out_dir = ROOT / pcfg.get("output_dir", "pipeline_out")
    summary = evaluate(out_dir / "predictions.csv", args.ground_truth, out_dir, pcfg.get("paper_ids"))
    print(summary.to_string(index=False))
    print(f"\nSaved evaluation_summary.csv and matches.csv to {out_dir}")


if __name__ == "__main__":
    main()
