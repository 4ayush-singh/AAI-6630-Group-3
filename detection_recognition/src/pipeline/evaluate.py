"""Stage 3 of the Phase 2 pipeline: score predictions against the ground truth.

The course ground truth lists molecules per paper (paperID, name, SMILES) with
no page or position, so matching is done per paper as two sets of molecules:

  1. For each paper, compare every predicted SMILES with every true SMILES.
  2. Pair them one-to-one, best match first (exact InChIKey match, then exact
     match ignoring stereochemistry, then highest Tanimoto similarity).
  3. Count:
       - true positives  = predictions that exactly match a true molecule
       - precision       = true positives / number of predictions
       - recall          = true positives / number of true molecules
       - F1              = harmonic mean of precision and recall
     The same is reported with stereochemistry ignored.

PDF files are linked to paper IDs by the first number in the file name
(for example "4_paper.pdf" -> paper 4), or by an explicit mapping in
config.yaml under pipeline.paper_ids.
"""
import re
from pathlib import Path

import pandas as pd

from src.metrics import exact_match, exact_match_no_stereo, tanimoto


def paper_id_for(pdf_name, mapping):
    """Paper ID for a PDF file name, from config mapping or the first number in the name."""
    if mapping and pdf_name in mapping:
        return int(mapping[pdf_name])
    m = re.search(r"\d+", Path(pdf_name).stem)
    return int(m.group()) if m else None


def match_paper(preds, truths):
    """One-to-one matching of predicted vs true SMILES for one paper.

    Returns a list of (pred_index, true_index, exact, exact_no_stereo, tanimoto).
    """
    scores = []
    for i, p in enumerate(preds):
        for j, t in enumerate(truths):
            ex = exact_match(p, t)
            ns = ex or exact_match_no_stereo(p, t)
            tan = 1.0 if ex else tanimoto(p, t)
            scores.append((ex, ns, tan, i, j))
    scores.sort(reverse=True)  # exact first, then no-stereo, then Tanimoto
    used_p, used_t, pairs = set(), set(), []
    for ex, ns, tan, i, j in scores:
        if i in used_p or j in used_t or tan <= 0:
            continue
        used_p.add(i); used_t.add(j)
        pairs.append((i, j, ex, ns, tan))
    return pairs


def _prf(tp, n_pred, n_true):
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return round(precision, 3), round(recall, 3), round(f1, 3)


def evaluate(predictions_csv, ground_truth_csv, out_dir, mapping=None):
    """Score predictions per paper; write matches.csv and evaluation_summary.csv."""
    preds = pd.read_csv(predictions_csv)
    # Markush outputs (with '*') are cores for R-group enumeration, not full molecules
    preds = preds[~preds["is_markush"].astype(bool)]
    gt = pd.read_csv(ground_truth_csv)
    gt.columns = [c.strip() for c in gt.columns]
    smiles_col = next(c for c in gt.columns if "smiles" in c.lower())
    name_col = next((c for c in gt.columns if "name" in c.lower()), None)
    preds["paperID"] = preds["pdf"].map(lambda n: paper_id_for(n, mapping))
    unmapped = preds.loc[preds["paperID"].isna(), "pdf"].unique()
    if len(unmapped):
        print(f"Warning: no paper ID found for {list(unmapped)}; add them under pipeline.paper_ids in config.yaml")

    match_rows, summary = [], []
    for pid in sorted(gt["paperID"].unique()):
        truths = gt.loc[gt["paperID"] == pid].reset_index(drop=True)
        p = preds.loc[preds["paperID"] == pid].reset_index(drop=True)
        p_smiles = p["pred_smiles"].fillna("").astype(str).tolist()
        pairs = match_paper(p_smiles, truths[smiles_col].astype(str).tolist())
        for i, j, ex, ns, tan in pairs:
            match_rows.append({"paperID": pid, "crop_id": p.loc[i, "crop_id"], "page": p.loc[i, "page"],
                               "pred_smiles": p_smiles[i], "true_name": truths.loc[j, name_col] if name_col else "",
                               "true_smiles": truths.loc[j, smiles_col], "exact_match": ex,
                               "exact_match_no_stereo": ns, "tanimoto": round(tan, 3)})
        tp = sum(1 for _, _, ex, _, _ in pairs if ex)
        tp_ns = sum(1 for _, _, _, ns, _ in pairs if ns)
        pr, rc, f1 = _prf(tp, len(p), len(truths))
        _, _, f1_ns = _prf(tp_ns, len(p), len(truths))
        summary.append({"paperID": pid, "pdf": ", ".join(p["pdf"].unique()), "n_true": len(truths),
                        "n_predicted": len(p), "exact_matches": tp, "precision": pr, "recall": rc, "f1": f1,
                        "matches_ignoring_stereo": tp_ns, "f1_ignoring_stereo": f1_ns})

    summary = pd.DataFrame(summary)
    tot_tp, tot_p, tot_t = summary.exact_matches.sum(), summary.n_predicted.sum(), summary.n_true.sum()
    pr, rc, f1 = _prf(tot_tp, tot_p, tot_t)
    _, _, f1_ns = _prf(summary.matches_ignoring_stereo.sum(), tot_p, tot_t)
    summary.loc[len(summary)] = {"paperID": "ALL", "pdf": "", "n_true": tot_t, "n_predicted": tot_p,
                                 "exact_matches": tot_tp, "precision": pr, "recall": rc, "f1": f1,
                                 "matches_ignoring_stereo": summary.matches_ignoring_stereo.sum(),
                                 "f1_ignoring_stereo": f1_ns}
    out_dir = Path(out_dir)
    pd.DataFrame(match_rows).to_csv(out_dir / "matches.csv", index=False)
    summary.to_csv(out_dir / "evaluation_summary.csv", index=False)
    return summary
