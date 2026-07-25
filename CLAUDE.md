# CLAUDE.md

## Workspace Focus
- Main development happens in `TwoStream_Yolov8/`.
- Root-level scripts such as `convert_llvip_to_yolo.py`, `xml_to_yolo.py`, and `testir.py` are utility entry points, not the core library.
- Keep large binaries, videos, screenshots, weights, datasets, and generated `runs/` outputs out of normal code commits.

## Code Placement
- Put reusable model, training, and inference logic in `TwoStream_Yolov8/ultralytics/`.
- Put experiment-specific helpers in the closest task folder: `detect/`, `export/`, `dataProcess/`, `deploy/`, `plot/`, or `int8Calib/`.
- Store custom architecture YAMLs in `TwoStream_Yolov8/yaml/`.

## Working Conventions
- Run Python commands from `TwoStream_Yolov8/` unless the target script is at the repository root.
- Prefer small, reviewable changes over broad refactors.
- Do not hardcode machine-specific paths when touching scripts; replace them with configurable paths where practical.
- Preserve existing Chinese comments or notes when they carry project-specific context.

## Validation
- For training changes, verify with `python train.py` or the narrowest affected training entry point.
- For inference changes, validate with the relevant script in `detect/` or `deploy/`.
- For export changes, validate with `python export/exportOnnx.py`.
- There is no maintained local `tests/` directory yet, so script-level verification is the default.

## Documentation
- Keep contributor-facing instructions in `AGENTS.md`.
- Use `exp_record.md` for experiment notes and outcome tracking when the change affects model quality or deployment behavior.
