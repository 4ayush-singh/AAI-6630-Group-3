"""Alternative stage 1: detect molecules with MolDetV2, a pretrained deep-learning detector.

MolDetV2 is the YOLO11-based molecule detector used inside BioMiner, released
by the DP Technology team on Hugging Face (UniParser/MolDetv2). Unlike the
rule-based OpenCV detector in detect.py, it learned what molecule drawings look
like from labelled document pages. It is used here for inference only.

It writes the same outputs as detect.py (crops/, debug/, crops.csv with the same
columns), so the recognition stage works unchanged. Extra columns: class_name
and confidence; `score` holds the detector's confidence.
"""
import csv
from pathlib import Path

import cv2
from huggingface_hub import hf_hub_download, list_repo_files

from src.pipeline.detect import render_pages


def download_moldet(repo_id, filename, models_dir):
    """Download the MolDetV2 weights (skipped if already present). Returns the local path."""
    if not filename:  # pick the document model from the repo if no file name is configured
        weights = [f for f in list_repo_files(repo_id) if f.endswith(".pt")]
        if not weights:
            raise FileNotFoundError(f"No .pt weights found in {repo_id}")
        filename = next((f for f in weights if "doc" in f), weights[0])
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    return hf_hub_download(repo_id=repo_id, filename=filename, local_dir=str(models_dir))


def detect_molecules_moldet(pdf_paths, out_dir, models_dir, repo_id="UniParser/MolDetv2",
                            filename=None, dpi=200, imgsz=960, conf=0.25, pad=10, device=None):
    """Run MolDetV2 on every page; write crops/, debug/ and crops.csv. Returns the rows."""
    from ultralytics import YOLO  # imported here so the OpenCV detector never needs it

    model = YOLO(download_moldet(repo_id, filename, models_dir))
    out_dir = Path(out_dir)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "debug").mkdir(exist_ok=True)
    rows = []
    for pdf_path in pdf_paths:
        pdf_path = Path(pdf_path)
        for page, img in render_pages(pdf_path, dpi):
            result = model.predict(source=img, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
            boxes = result.boxes
            overlay = img.copy()
            for (x0, y0, x1, y1), score, cls in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(),
                                                    boxes.cls.cpu().numpy()):
                x, y, w, h = int(x0), int(y0), int(x1 - x0), int(y1 - y0)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 170, 0), 3)
                cv2.putText(overlay, f"{score:.2f}", (x, max(y - 6, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 170, 0), 2)
                crop_id = f"{pdf_path.stem}_p{page + 1}_{x}_{y}"
                crop_path = out_dir / "crops" / f"{crop_id}.png"
                cv2.imwrite(str(crop_path), img[max(y - pad, 0):y + h + pad, max(x - pad, 0):x + w + pad])
                rows.append({"crop_id": crop_id, "pdf": pdf_path.name, "page": page + 1,
                             "x": x, "y": y, "w": w, "h": h, "score": round(float(score), 3),
                             "crop_path": str(crop_path), "class_name": result.names[int(cls)],
                             "detector": "moldetv2"})
            cv2.imwrite(str(out_dir / "debug" / f"{pdf_path.stem}_p{page + 1}.jpg"), overlay)
    if rows:
        with open(out_dir / "crops.csv", "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows
