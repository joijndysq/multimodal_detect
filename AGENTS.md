# Repository Guidelines

## Project Structure & Module Organization
This repository centers on the nested `TwoStream_Yolov8/` project, where most code changes should land. Core library code lives in `TwoStream_Yolov8/ultralytics/`. Task-specific scripts are grouped by purpose: `dataProcess/` for dataset conversion, `detect/` for HBB/OBB inference, `export/` for ONNX and engine export, `deploy/` for realtime camera pipelines, `plot/` for visualization, and `int8Calib/` for TensorRT calibration. Custom model definitions belong in `TwoStream_Yolov8/yaml/`. Root-level helpers such as `convert_llvip_to_yolo.py` and `xml_to_yolo.py` are utility scripts, while `exp_record.md` stores experiment notes. Keep generated artifacts such as `runs/`, `engine/`, weights, datasets, videos, and screenshots out of commits.

## Build, Test, and Development Commands
Run commands from `TwoStream_Yolov8/` unless noted otherwise.

- `pip install -e .` installs the local Ultralytics fork in editable mode.
- `python train.py` starts the default training flow.
- `python train_for_windows.py` uses the Windows-safe training entry point.
- `python test.py` runs the current validation script against a configured checkpoint.
- `python info.py` prints model structure and parameter information.
- `python export/exportOnnx.py` exports a trained model to ONNX.
- `pytest tests/` is the upstream test command referenced by CI, but this fork currently does not include a local `tests/` directory.

## Coding Style & Naming Conventions
Follow Python 4-space indentation and keep lines within the `ruff`/`yapf` limit of 120 characters defined in `pyproject.toml`. Prefer `snake_case` for functions, variables, and filenames; keep model YAML names descriptive, for example `VIF-YOLO.yaml` or `PC2f_MPF_yolov8n.yaml`. Put reusable logic in `ultralytics/` and keep one-off experiments in dedicated scripts. Avoid new machine-specific absolute paths; prefer configurable dataset and weight locations.

## Testing Guidelines
There is no maintained local test suite yet, so contributors should validate changes with the smallest relevant script: training edits with `train.py`, inference edits with scripts in `detect/` or `deploy/`, and export changes with `export/exportOnnx.py`. Record notable experiment outcomes in `exp_record.md` when they affect model quality or deployment behavior. If you add automated tests, place them in `TwoStream_Yolov8/tests/` and name files `test_*.py` so they match the existing `pytest` configuration.

## Commit & Pull Request Guidelines
Recent history mixes conventional commits (`feat: add EMA attention...`, `chore: initialize local repository`) with short ad hoc messages. Prefer `type: summary` format such as `feat: add llvip training config` or `fix: correct OBB postprocess path`. Keep pull requests focused, describe the dataset/model YAML used for verification, link related issues, and include screenshots or sample outputs for detection, plotting, or deployment changes.
