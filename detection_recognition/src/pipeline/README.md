# Phase 2 module: Molecule Detection, Cropping and Recognition

**Owner:** Vaibhavi Bogaraja Raj · AAI 6630 Computer Vision, XN project (Assignment 2.3)

This module is the front half of the group pipeline. It takes research-paper PDFs, finds every drawn molecule on every page, crops each one, and converts each crop into a SMILES string. It also includes a scoring step used to test the module against the course ground truth.

```
PDFs ─► [render pages] ─► [detect molecules] ─► [crop] ─► [recognise: image → SMILES] ─► [validate with RDKit]
            pypdfium2      MolDetV2 (default)     OpenCV      MolGlyph (default)              predictions.csv
                           or OpenCV rules                    or MolScribe
```

## Inputs and outputs

| | What | Format |
|---|---|---|
| **Input** | A folder of PDF files | e.g. `pdfs/1.pdf`, `pdfs/2.pdf` … |
| **Output 1 (detection)** | One PNG per detected molecule, plus a manifest | `pipeline_out/crops/*.png`, `pipeline_out/crops.csv` |
| **Output 2 (recognition)** | One SMILES per crop, with validity checks | `pipeline_out/predictions.csv` |
| **Debug output** | Every page with detection boxes drawn on it | `pipeline_out/debug/*.jpg` |
| **Optional (testing)** | Scores against a ground-truth CSV | `pipeline_out/evaluation_summary.csv`, `pipeline_out/matches.csv` |

The exact column definitions of `crops.csv` and `predictions.csv` (the interface to the group's next stage) are in [`docs/pipeline_interface.md`](../../docs/pipeline_interface.md). In short, `predictions.csv` has one row per crop with `pdf`, `page`, box position (`x`, `y`, `w`, `h`), `pred_smiles`, `canonical_smiles`, `is_valid`, `is_markush`, `confidence`, `runtime_s` and `error`.

## How to run

From the project root, after the setup in the main README (Python 3.10 environment, `uv pip install -r requirements.txt`, Hugging Face token in `.env`):

```bash
# Detection + recognition with the defaults in config.yaml (MolDetV2 + MolGlyph)
python src/run_pipeline.py --pdf_dir pdfs

# Options
python src/run_pipeline.py --pdf_dir pdfs --detector opencv      # rule-based OpenCV detector instead
python src/run_pipeline.py --pdf_dir pdfs --tool molscribe       # MolScribe instead of MolGlyph
python src/run_pipeline.py --pdf_dir pdfs --detect-only          # crops only, no recognition

# Testing against a ground-truth CSV (columns: paperID, name, SMILES)
python src/evaluate_pipeline.py --ground_truth pdfs/ground_truth.csv
```

All settings (DPI, detector, recogniser, thresholds, output folder) are in the `pipeline:` and `moldet:` sections of `config.yaml`; no paths are hard-coded. PDFs and outputs are excluded from Git by `.gitignore`.

## Code guide

| File | Responsibility |
|---|---|
| `src/run_pipeline.py` | Command-line entry point: reads `config.yaml`, runs detection then recognition |
| `src/pipeline/detect_moldet.py` | **Default detector:** pretrained MolDetV2 (YOLO11) on each rendered page, crops with OpenCV |
| `src/pipeline/detect.py` | **Baseline detector:** rule-based OpenCV detection (page rendering, text masking, contour proposals, feature-based triage) |
| `src/pipeline/recognize.py` | Runs the recogniser on every crop, validates and canonicalises SMILES with RDKit |
| `src/pipeline/evaluate.py` | Per-paper matching of predictions to ground truth; precision, recall, F1 |
| `src/evaluate_pipeline.py` | Command-line entry point for evaluation |
| `src/tools/` | Recogniser wrappers shared with the Phase 1 benchmark (MolGlyph, MolScribe) |

## Technical methods

### Pretrained models and transfer learning

The module uses **pretrained models for inference only**. No model is trained or fine-tuned, so there are no training epochs. This is a deliberate transfer-learning choice: both models were trained by their authors on large labelled datasets of chemical documents, which we do not have, and they transfer well to our papers without further training.

| Stage | Model | Why this model |
|---|---|---|
| Detection | **MolDetV2** (YOLO11-nano, `UniParser/MolDetv2`, 5.5 MB), the detector inside BioMiner | Trained on real document pages to box molecule drawings. Compared with my OpenCV baseline on the six course PDFs, it found 40% more correct molecules (63 vs 45). Small and fast enough to run on a laptop CPU. |
| Recognition | **MolGlyph** (BioMiner's OCSR model, Swin Transformer encoder + transformer decoder, `jiaxianustc/MolGlyph`) | Best accuracy in my Phase 1 benchmark on all 8,861 BioVista images: 76.3% vs MolScribe's 46.1%, reproducing the published scores. On the same crops from the six course PDFs it also beat MolScribe (45 vs 31 correct). It can output Markush structures (R-group cores), which the next stage needs. |
| Alternative recogniser | **MolScribe** (Qian et al., 2023) | Kept as a switchable option and possible fallback; it did better than MolGlyph on very large molecules in Phase 1. |
| Baseline detector | **OpenCV rules** (no learning) | Classical CV baseline: renders pages, masks body text using the PDF text layer, merges ink into blobs with morphological dilation, and scores each region on ring count, share of diagonal (bond-like) lines, colour, ink density and axis lines. Fully transparent, needs no model download. |

### Data augmentation

**None is applied.** Data augmentation (rotation, flipping, brightness changes, etc.) is used to make a model more robust *while training it*. This module only runs pretrained models, so augmenting inputs would add no benefit; flipping or rotating a molecule drawing at inference time would also change how the recogniser reads stereochemistry. The pretrained models were trained by their authors with their own augmentations (for example MolScribe was trained on synthetically rendered molecules with heavy appearance augmentation).

### Preprocessing

1. **Format conversion:** each PDF page is rendered to an RGB image with `pypdfium2` at **300 DPI** (configurable). Rendering, rather than extracting embedded images, also captures molecules drawn as vector graphics. Raising the DPI from 200 to 300 improved results slightly (63 → 65 correct).
2. **Detection input:** MolDetV2 receives the full page; Ultralytics resizes it to 960 px (the size the model was trained at) and normalises it internally. Boxes below a confidence of 0.15 are discarded.
3. **Cropping:** each box is cut from the full-resolution page with a 10-pixel white margin so bonds at the edge are not cut off.
4. **Recognition input:** the recogniser's own transform resizes each crop to its training size, converts to greyscale and normalises pixel values (built into the vendored BioMiner code; unchanged so results match the published models).
5. **Post-processing:** MolGlyph's extended-SMILES output is converted to plain SMILES with BioMiner's own converter; every SMILES is validated and canonicalised with RDKit. For evaluation, Markush outputs (containing `*`) are set aside, because they are R-group cores rather than complete molecules.

## Test results (testing evidence)

**Test case:** the six benchmark PDFs provided for the project (paper IDs 1, 2, 4, 6, 16, 19), scored against the course ground truth of 222 molecules. Settings: MolDetV2 (confidence 0.15) + MolGlyph, 300 DPI, CPU (Apple M-series laptop).

Saved outputs from this run are in [`examples/phase2/`](../../examples/phase2/):
- `sample_crops.csv`: example rows of the detection output (input to recognition)
- `sample_predictions.csv`: example recognised molecules with their matched ground-truth molecule, including correct predictions, near misses and Markush cores
- `evaluation_summary.csv`: full per-paper scores

**Sample input → output** (paper 1, page 3):

| Detected crop (box at x, y) | Predicted SMILES | Ground truth | Correct |
|---|---|---|---|
| `1_p3_1716_312` | `COc1cc2c(Oc3ccc(NC(=O)C4(C(=O)Nc5ccc(F)cc5)CC4)cc3F)ccnc2cc1OCCCN1CCOCC1` | Foretinib | ✅ |
| `1_p3_1247_268` | `CCN1CCN(Cc2ccc(NC(=O)Nc3ccc(Oc4cc(NC)ncn4)cc3)cc2C(F)(F)F)CC1` | AST487 | ✅ |
| `1_p3_275_359` | Dasatinib structure (full SMILES in sample file) | Dasatinib | ✅ |
| `1_p3_763_364` | Sorafenib structure (full SMILES in sample file) | Sorafenib | ✅ |

**Overall results:**

| Paper | True molecules | Predicted (full molecules) | Correct | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 1 | 17 | 18 | 17 | 0.944 | 1.000 | **0.971** |
| 2 | 34 | 15 | 12 | 0.800 | 0.353 | 0.490 |
| 4 | 61 | 23 | 14 | 0.609 | 0.230 | 0.333 |
| 6 | 1 | 2 | 0 | 0.000 | 0.000 | 0.000 |
| 16 | 4 | 8 | 2 | 0.250 | 0.500 | 0.333 |
| 19 | 105 | 26 | 20 | 0.769 | 0.190 | 0.305 |
| **All** | **222** | **92** | **65** | **0.707** | **0.293** | **0.414** |

**How the module improved during development** (same six PDFs and ground truth):

| Version | Correct | Precision | Recall | F1 |
|---|---|---|---|---|
| OpenCV detector + MolScribe | 31 | 0.208 | 0.140 | 0.167 |
| OpenCV detector + MolGlyph | 45 | 0.341 | 0.203 | 0.254 |
| OpenCV detector, threshold lowered 0.55 → 0.45 | 45 | 0.302 | 0.203 | 0.243 |
| MolDetV2 + MolGlyph | 63 | 0.244 | 0.284 | 0.262 |
| + Markush cores set aside in scoring | 63 | 0.708 | 0.284 | 0.405 |
| + 300 DPI, MolDetV2 confidence 0.15 (**current**) | **65** | **0.707** | **0.293** | **0.414** |

### What the results show

- **When the module outputs a molecule, it is usually right** (71% precision; 17/17 on paper 1).
- **About 90% of the missing molecules (141 of 157) come from R-group tables** in papers 2, 4 and 19, where one core drawing plus a table represents many molecules. The module already extracts these cores and R-group fragments (157 Markush outputs) but does not yet combine them. This is the planned next stage (R-group enumeration).
- **Only 16 missing molecules are recognition errors**, mostly a single wrong substituent (for example a methoxy group read as ethoxy).

## Limitations and next steps

- R-group enumeration (core + table entries → full molecules) is not implemented yet; it is the largest remaining gain.
- Molecules that appear only as names in text are out of scope for a vision-only module.
- The OpenCV detector's rules were tuned on two papers; MolDetV2 is now the default.
- Recognition runs at about 0.4 s per crop on CPU; a GPU is faster but not required.

## References

- Yan, J., et al. (2026). *BioMiner: A multi-modal system for automated mining of protein-ligand bioactivity data from literature* (arXiv:2604.21508). Source of MolGlyph and the MolDetV2 integration. https://github.com/jiaxianyan/BioMiner
- MolDetV2 model weights: DP Technology, https://huggingface.co/UniParser/MolDetv2
- Qian, Y., et al. (2023). MolScribe: Robust molecular structure recognition with image-to-graph generation. *Journal of Chemical Information and Modeling, 63*(7), 1925–1934.
- Ultralytics YOLO: https://docs.ultralytics.com
