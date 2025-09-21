# Claw Segmentation

This repository contains an automated pipeline that detects the inner and outer claws of cattle hooves, measures their relative area, and generates visual analytics for longitudinal monitoring. A pre-trained YOLOv8 segmentation model drives the detections, and the surrounding Python code takes care of data preparation, metric computation, persistence, and plotting.

## Repository Layout

- `claw_segmentation/inference_pipeline.py` – reusable orchestration code that loads the YOLO model, runs inference, extracts metrics, and writes all artefacts.
- `claw_segmentation/run_inference.py` – command-line entry point for inference; accepts paths and thresholds and delegates to the pipeline module.
- `claw_segmentation/analysis.py` – optional post-processing script that enriches the results with birth dates, weights, and linear scores.
- `claw_segmentation/data/` – expected input structure. Store one folder per recording date (e.g. `18_04_2023/`) containing images. The `ergebnisse/` sub-folder is reserved for generated outputs.
- `claw_segmentation/models/` – location for YOLO weight files.

## Environment Setup

1. Install the dependencies (Python 3.10+):

   ```bash
   poetry install
   ```

2. Ensure the data directory follows the expected structure:

   ```text
   claw_segmentation/data/
   ├── 18_04_2023/
   │   ├── <images ending in L*.jpg>
   │   └── <images ending in R*.jpg>
   ├── 28_01_2025/
   │   └── ...
   ├── Klauenerziehung_Urliste.xlsx   # optional, used for colour-group metadata
   └── ergebnisse/                     # automatically maintained by the pipeline
   ```

3. Place the trained YOLO weights (defaults to `claw_segmentation/models/yolov8_claw_segmentation_v0.pt`).

## Running Inference (`run_inference.py`)

The `run_inference.py` module is dedicated to inference. Run it via Python or Poetry:

```bash
poetry run python -m claw_segmentation.run_inference \
  --data-dir claw_segmentation/data \
  --result-dir claw_segmentation/data/ergebnisse \
  --model-path claw_segmentation/models/yolov8_claw_segmentation_v0.pt
```

If you prefer to call the script directly (for example inside an already
activated virtual environment), the module now also supports plain execution:

```bash
python claw_segmentation/run_inference.py \
  --data-dir claw_segmentation/data \
  --result-dir claw_segmentation/data/ergebnisse \
  --model-path claw_segmentation/models/yolov8_claw_segmentation_v0.pt
```

Available flags:

- `--conf` / `--iou` / `--max-det` – forwarded to the YOLO predictor.
- `--excel-filename` – relative Excel file used to map ear tags to colour groups (defaults to `Klauenerziehung_Urliste.xlsx`).

If you omit the arguments, the defaults above are used.

## What the Pipeline Does

1. **Load previous results** from `ergebnisse/ergebnisse.csv` (if it exists) and record the dates that have already been processed.
2. **Collect new date folders** in `data/` (ignoring the `ergebnisse/` directory) and only keep the ones that have not yet been processed.
3. **Run YOLOv8 segmentation** on every image of those folders with the configured thresholds.
4. **Extract claw metrics** per detection:
   - figure out whether the image represents a left or right hoof using the filename suffix (`L`/`R`),
   - compute inner/outer claw areas, their centers, and the relative area ratio (outer / inner).
5. **Render per-image overlays** that annotate the detections and save them under `ergebnisse/grafiken/<YYYY_MM_DD>/`.
6. **Aggregate results** into a dataframe with the columns:
   - `Ohrmarkennummer`
   - `Datum` (normalised to `YYYY_MM_DD`)
   - `Klaue` (`links` / `rechts`)
   - `Relative Klauenfläche (=Aussen/Innen)`
   - `Mess Nr.` (per-animal counting index)
7. **Join metadata** from `Klauenerziehung_Urliste.xlsx` if present and append a `Gruppe` column.
8. **Write artefacts**:
   - CSV: `ergebnisse/ergebnisse.csv`
   - Box, line, and histogram plots for both `Klaue` and `Gruppe`
   - Joint plot comparing left vs. right claw ratios.

All steps are idempotent: rerunning the script skips folders whose dates already exist in the CSV.

## Outputs

After a successful run you will find:

- Updated CSV with cumulative measurements: `claw_segmentation/data/ergebnisse/ergebnisse.csv`
- Analytical visualisations in `claw_segmentation/data/ergebnisse/*.png`
- Annotated prediction images in `claw_segmentation/data/ergebnisse/grafiken/<date>/`
