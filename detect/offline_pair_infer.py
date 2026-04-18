import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Offline pair inference for two-stream YOLO weights")
    parser.add_argument("--weights", type=str, required=True, help="Path to .pt weights")
    parser.add_argument("--rgb-dir", type=str, required=True, help="RGB image directory")
    parser.add_argument("--ir-dir", type=str, required=True, help="IR image directory")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--max-images", type=int, default=100, help="Max number of pairs to process")
    return parser.parse_args()


def find_images(folder):
    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    files = []
    for ext in exts:
        files.extend(glob.glob(str(Path(folder) / ext)))
    return sorted(files)


def ensure_bgr(img):
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def draw_hbb(img, result):
    out = img.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return out

    xywh = boxes.xywh.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(int)
    conf = boxes.conf.cpu().numpy()
    names = result.names

    for b, c, s in zip(xywh, cls, conf):
        x, y, w, h = b
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        color = (0, 255, 0)
        label = f"{names.get(c, str(c))} {s:.2f}" if isinstance(names, dict) else f"{c} {s:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def main():
    args = parse_args()

    out_rgb = Path(args.out_dir) / "rgb"
    out_ir = Path(args.out_dir) / "ir"
    out_rgb.mkdir(parents=True, exist_ok=True)
    out_ir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    rgb_files = find_images(args.rgb_dir)
    if not rgb_files:
        raise FileNotFoundError(f"No images found in {args.rgb_dir}")

    done = 0
    skipped = 0

    for rgb_path in rgb_files[: args.max_images]:
        name = Path(rgb_path).name
        stem = Path(name).stem

        ir_candidates = [Path(args.ir_dir) / name, Path(args.ir_dir) / f"{stem}.png", Path(args.ir_dir) / f"{stem}.jpg"]
        ir_path = next((p for p in ir_candidates if p.exists()), None)
        if ir_path is None:
            skipped += 1
            continue

        rgb = cv2.imread(rgb_path)
        ir = cv2.imread(str(ir_path), cv2.IMREAD_UNCHANGED)
        rgb = ensure_bgr(rgb)
        ir = ensure_bgr(ir)
        if rgb is None or ir is None:
            skipped += 1
            continue

        if ir.shape[:2] != rgb.shape[:2]:
            ir = cv2.resize(ir, (rgb.shape[1], rgb.shape[0]))

        fused = np.concatenate((rgb, ir), axis=2)
        result = model.predict(fused, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]

        rgb_draw = draw_hbb(rgb, result)
        ir_draw = draw_hbb(ir, result)

        cv2.imwrite(str(out_rgb / name), rgb_draw)
        cv2.imwrite(str(out_ir / name), ir_draw)
        done += 1

    print(f"done={done} skipped={skipped} out={args.out_dir}")


if __name__ == "__main__":
    main()
