"""Phase 2 pipeline: PDFs -> molecule crops -> SMILES.

Usage (from the project root):
    python src/run_pipeline.py --pdf_dir pdfs/                      # detection + MolGlyph
    python src/run_pipeline.py --pdf_dir pdfs/ --tool molscribe     # use MolScribe instead
    python src/run_pipeline.py --pdf_dir pdfs/ --detect-only        # crops only, no models
    python src/run_pipeline.py --pdf_dir pdfs/ --detector moldet    # pretrained MolDetV2 detector

Outputs (in pipeline_out/ by default):
    crops/            one PNG per detected molecule
    debug/            page overlays: green = kept, red = rejected (with score)
    crops.csv         detection output (input to recognition)
    predictions.csv   recognition output (input to evaluation)
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.pipeline.detect import detect_molecules  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Extract molecules from PDFs and convert them to SMILES.")
    p.add_argument("--pdf_dir", required=True, help="Folder containing the PDFs")
    p.add_argument("--config", default=str(ROOT / "config.yaml"))
    p.add_argument("--out", default=None, help="Output folder (default from config)")
    p.add_argument("--tool", default=None, choices=["molglyph", "molscribe"])
    p.add_argument("--detector", default=None, choices=["opencv", "moldet"],
                   help="opencv = rule-based (default), moldet = pretrained MolDetV2")
    p.add_argument("--detect-only", action="store_true", help="Stop after detection")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    pcfg = cfg.get("pipeline", {})
    out_dir = ROOT / (args.out or pcfg.get("output_dir", "pipeline_out"))
    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.pdf_dir}")

    # Stage 1: detection and cropping
    detector = args.detector or pcfg.get("detector", "opencv")
    if detector == "moldet":
        from src.pipeline.detect_moldet import detect_molecules_moldet
        mcfg = cfg.get("moldet", {})
        rows = detect_molecules_moldet(pdfs, out_dir, models_dir=ROOT / cfg["benchmark"]["models_dir"],
                                       repo_id=mcfg.get("hf_repo", "UniParser/MolDetv2"),
                                       filename=mcfg.get("hf_file"), dpi=pcfg.get("dpi", 200),
                                       imgsz=mcfg.get("imgsz", 960), conf=mcfg.get("conf", 0.25),
                                       pad=pcfg.get("pad", 10))
    else:
        rows = detect_molecules(pdfs, out_dir, dpi=pcfg.get("dpi", 200),
                                threshold=pcfg.get("threshold", 0.55), pad=pcfg.get("pad", 10))
    print(f"Detection ({detector}): {len(rows)} molecule crops from {len(pdfs)} PDF(s) -> {out_dir / 'crops.csv'}")
    if args.detect_only or not rows:
        return

    # Stage 2: recognition (imports torch/models only when needed)
    from src.pipeline.recognize import recognize_crops
    tool = args.tool or pcfg.get("tool", "molglyph")
    preds = recognize_crops(out_dir / "crops.csv", out_dir / "predictions.csv", tool, cfg)
    print(f"Recognition ({tool}): {int(preds['is_valid'].sum())} valid, "
          f"{int(preds['is_markush'].sum())} Markush, {int(preds['error'].notna().sum())} errors "
          f"-> {out_dir / 'predictions.csv'}")


if __name__ == "__main__":
    main()
