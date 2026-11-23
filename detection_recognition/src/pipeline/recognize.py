"""Stage 2 of the Phase 2 pipeline: convert each molecule crop into SMILES.

Reads crops.csv from detect.py and reuses the tool wrappers from the Phase 1
benchmark (src/tools), so the same MolGlyph / MolScribe code and settings are
used. Each prediction is checked with RDKit and stored in canonical form.

Output contract (read by the evaluation stage), predictions.csv columns:
  crop_id, pdf, page, x, y, w, h, crop_path, tool, pred_smiles,
  canonical_smiles, is_valid, is_markush, confidence, runtime_s, error
"""
from pathlib import Path

import pandas as pd

from src.tools import build_tool

try:
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
except ImportError:  # RDKit is in requirements.txt; this only guards odd installs
    Chem = None


def canonicalize(smiles):
    """RDKit canonical SMILES, or None if the prediction is not a valid molecule."""
    if Chem is None or not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def recognize_crops(crops_csv, out_csv, tool_name, cfg):
    """Run one OCSR tool on every crop listed in crops_csv and save predictions."""
    crops = pd.read_csv(crops_csv)
    tool = build_tool(tool_name, cfg["tools"][tool_name], cfg["benchmark"])
    preds = tool.predict(crops["crop_path"].tolist())
    out = crops[["crop_id", "pdf", "page", "x", "y", "w", "h", "crop_path"]].copy()
    out["tool"] = tool_name
    out["pred_smiles"] = [p.pred_smiles for p in preds]
    out["canonical_smiles"] = out["pred_smiles"].map(canonicalize)
    out["is_valid"] = out["canonical_smiles"].notna()
    out["is_markush"] = out["pred_smiles"].fillna("").str.contains("*", regex=False)
    out["confidence"] = [p.confidence for p in preds]
    out["runtime_s"] = [p.runtime_s for p in preds]
    out["error"] = [p.error for p in preds]
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    return out
