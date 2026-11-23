"""Stage 1 of the Phase 2 pipeline: find and crop molecule drawings in PDFs.

Classical computer vision with OpenCV, no training data needed:
  1. Render each page to an image (works for raster and vector drawings).
  2. Erase ordinary body text using the PDF's text layer (atom labels such as
     N, OH, CF3 are kept, because they belong to the molecules).
  3. Propose candidate regions: binarise, dilate so the bonds of one molecule
     merge into one blob, take contours, merge nearby fragments.
  4. Triage each region with geometric features (rings, share of diagonal
     bond-like lines, colour, ink density, long axis lines) and a rule score.
  5. Save each accepted region as a crop, plus a crops.csv manifest and a
     debug overlay per page (green = kept, red = rejected).

Output contract (read by recognize.py), crops.csv columns:
  crop_id, pdf, page, x, y, w, h, score, crop_path, plus the feature columns.
"""
import csv
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pdfplumber
import pypdfium2 as pdfium

# Short tokens that are atom or group labels inside a drawing, not body text
ATOM_TOKEN = re.compile(r"^[\(\[]?([A-Z][a-z]?\d*|Me|Et|Ph|Bn|Boc|Ac|n|H)+[+\-−]?[\)\]]?\d*$")


def render_pages(pdf_path, dpi):
    """Yield (page_index, BGR image) for every page of a PDF."""
    pdf = pdfium.PdfDocument(str(pdf_path))
    for i in range(len(pdf)):
        pil = pdf[i].render(scale=dpi / 72).to_pil().convert("RGB")
        yield i, cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def text_boxes(pdf_path, page_index, scale):
    """Pixel boxes of ordinary words to erase before looking for drawings."""
    boxes = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for w in pdf.pages[page_index].extract_words():
            t = w["text"].strip()
            if len(t) <= 4 and ATOM_TOKEN.match(t):
                continue  # keep atom labels such as N, OH, NH2, CF3
            boxes.append((int(w["x0"] * scale) - 2, int(w["top"] * scale) - 2,
                          int(w["x1"] * scale) + 2, int(w["bottom"] * scale) + 2))
    return boxes


def propose_regions(img, erase, dpi):
    """Candidate (x, y, w, h) boxes, plus the binarised page used for features."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for x0, y0, x1, y1 in erase:
        gray[max(y0, 0):y1, max(x0, 0):x1] = 255
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = max(3, int(dpi / 18))  # joins bonds of one molecule, not separate molecules
    merged = cv2.dilate(bw, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
    contours, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H, W = gray.shape
    min_side = dpi * 0.25  # about a quarter inch
    boxes = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w >= min_side and h >= min_side and w * h <= 0.6 * W * H:
            boxes.append([x, y, w, h])
    return _merge_close(boxes, gap=int(dpi * 0.05)), bw


def _merge_close(boxes, gap):
    """Merge boxes closer than `gap` pixels (re-joins fragments of one molecule)."""
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                if (a[0] - gap < b[0] + b[2] and b[0] - gap < a[0] + a[2]
                        and a[1] - gap < b[1] + b[3] and b[1] - gap < a[1] + a[3]):
                    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
                    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
                    boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                    boxes.pop(j)
                    changed = True
                    break
            if changed:
                break
    return [tuple(b) for b in boxes]


def region_features(img, bw, box):
    """Geometric features that separate molecule drawings from charts and photos."""
    x, y, w, h = box
    crop_bw, crop = bw[y:y + h, x:x + w], img[y:y + h, x:x + w]
    ink = crop_bw > 0
    n_ink = int(ink.sum()) or 1
    f = {"ink_ratio": n_ink / (w * h)}
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    f["color_ratio"] = float(((hsv[..., 1] > 80) & ink).sum()) / n_ink

    lines = cv2.createLineSegmentDetector(0).detect(cv2.bitwise_not(crop_bw))[0]
    total = diag = axis = 0.0
    n = 0
    if lines is not None:
        for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
            length = math.hypot(x2 - x1, y2 - y1)
            if length < 4:
                continue
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1))) % 180
            off_axis = min(angle % 90, 90 - angle % 90)
            n += 1
            total += length
            if off_axis > 15:  # bonds at about 30/60 degrees (rings)
                diag += length
            if off_axis < 3 and length > 0.5 * max(w, h):  # plot axes, table rules
                axis += length
    total = total or 1.0
    f.update(n_segments=n, diag_ratio=diag / total, axis_ratio=axis / total)

    cnts, hier = cv2.findContours(crop_bw, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    rings = 0
    if hier is not None:
        for c, hh in zip(cnts, hier[0]):
            if hh[3] == -1:
                continue  # outer contour, not a hole
            area = cv2.contourArea(c)
            if 40 <= area <= 0.25 * w * h:
                approx = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
                if 5 <= len(approx) <= 8 and cv2.isContourConvex(approx):
                    rings += 1  # pentagon/hexagon-like hole
    f["rings"] = rings
    return f


def molecule_score(f):
    """Rule-based molecule score in [0, 1]. To do: replace with a trained classifier."""
    if (f["color_ratio"] > 0.45 or f["ink_ratio"] > 0.35 or f["n_segments"] < 8
            or f["diag_ratio"] < 0.25 or f["axis_ratio"] > 0.3):
        return 0.0
    return round(min(f["rings"], 3) / 3 * 0.40 + min(f["diag_ratio"] / 0.5, 1.0) * 0.35
                 + (1 - min(f["axis_ratio"] / 0.3, 1.0)) * 0.15
                 + (1 - min(f["color_ratio"] / 0.45, 1.0)) * 0.10, 3)


def detect_molecules(pdf_paths, out_dir, dpi=200, threshold=0.55, pad=10):
    """Run detection on PDFs; write crops/, debug/ and crops.csv. Returns the rows."""
    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "debug").mkdir(exist_ok=True)
    rows = []
    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        for page, img in render_pages(pdf_path, dpi):
            regions, bw = propose_regions(img, text_boxes(pdf_path, page, dpi / 72), dpi)
            overlay = img.copy()
            for box in regions:
                feats = region_features(img, bw, box)
                score = molecule_score(feats)
                keep = score >= threshold
                x, y, w, h = box
                color = (0, 170, 0) if keep else (0, 0, 255)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
                cv2.putText(overlay, f"{score:.2f}", (x, max(y - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                if not keep:
                    continue
                crop_id = f"{pdf_path.stem}_p{page + 1}_{x}_{y}"
                crop_path = out_dir / "crops" / f"{crop_id}.png"
                cv2.imwrite(str(crop_path), img[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad])
                rows.append({"crop_id": crop_id, "pdf": pdf_path.name, "page": page + 1,
                             "x": x, "y": y, "w": w, "h": h, "score": score,
                             "crop_path": str(crop_path),
                             **{k: round(float(v), 3) for k, v in feats.items()}})
            cv2.imwrite(str(out_dir / "debug" / f"{pdf_path.stem}_p{page + 1}.jpg"), overlay)
    if rows:
        with open(out_dir / "crops.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows
