# Phase 2 pipeline: stage interfaces

Each stage reads the previous stage's CSV, so team members can build and test stages separately.

```
PDFs ──► [1] detect.py ──► crops.csv + crops/*.png ──► [2] recognize.py ──► predictions.csv ──► [3] evaluation
```

## crops.csv (output of detection)

| Column | Meaning |
|---|---|
| `crop_id` | Unique ID: `<pdf name>_p<page>_<x>_<y>` |
| `pdf`, `page` | Source PDF file name and page number (1-based) |
| `x`, `y`, `w`, `h` | Bounding box in pixels at the configured DPI (default 200) |
| `score` | Molecule score from the rule-based triage, 0 to 1 |
| `crop_path` | Path to the cropped PNG |
| feature columns | `ink_ratio`, `color_ratio`, `n_segments`, `diag_ratio`, `axis_ratio`, `rings` |

## predictions.csv (output of recognition)

| Column | Meaning |
|---|---|
| `crop_id`, `pdf`, `page`, `x`, `y`, `w`, `h`, `crop_path` | Copied from crops.csv |
| `tool` | `molglyph` or `molscribe` |
| `pred_smiles` | SMILES predicted by the tool (MolGlyph output already converted from E-SMILES) |
| `canonical_smiles` | RDKit canonical form, empty if not a valid single molecule |
| `is_valid`, `is_markush` | RDKit could parse it / contains `*` attachment points |
| `confidence` | MolScribe's confidence score (empty for MolGlyph) |
| `runtime_s`, `error` | Seconds per image; error message if the tool failed |

## For the evaluation stage

Match predictions to ground truth by `pdf` and `page` (and box position, if the ground truth has boxes), then score with `src/metrics.py` (`official_smiles_match`, `exact_match`, `tanimoto`).
