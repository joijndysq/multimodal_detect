import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

# Ensure custom ultralytics in this repo is used for custom model modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics import YOLO

from camera_init import MultiCameraSystem, build_parser as build_camera_parser


def build_parser() -> argparse.ArgumentParser:
    parser = build_camera_parser()
    parser.description = "实时融合检测：Lepton 红外 + RealSense RGB"
    parser.add_argument("--weights", type=str, default=str(Path(__file__).resolve().parent / "best.pt"), help="模型权重路径")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--obb", action="store_true", help="使用旋转框输出")
    parser.add_argument("--fuse", action="store_true", help="启动时执行模型 fuse（可能较慢）")
    return parser


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _fuse_rgb_ir(rgb: np.ndarray, ir: np.ndarray, imgsz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    ir_resized = cv2.resize(ir, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    ir_resized = _ensure_bgr(ir_resized)
    fused = np.concatenate((rgb_resized, ir_resized), axis=2)
    return fused, rgb_resized, ir_resized


def _draw_hbb(img: np.ndarray, result, names: dict[int, str]) -> np.ndarray:
    out = img.copy()
    if result.boxes is None or len(result.boxes) == 0:
        return out

    xywh = result.boxes.xywh.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    conf = result.boxes.conf.cpu().numpy()

    for box, c, s in zip(xywh, cls, conf):
        x, y, w, h = box
        x1, y1 = int(x - w / 2), int(y - h / 2)
        x2, y2 = int(x + w / 2), int(y + h / 2)
        color = (0, 255, 0)
        label = f"{names.get(c, str(c))} {s:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def _draw_obb(img: np.ndarray, result, names: dict[int, str]) -> np.ndarray:
    out = img.copy()
    if result.obb is None or len(result.obb) == 0:
        return out

    xywhr = result.obb.xywhr.cpu().numpy()
    cls = result.obb.cls.cpu().numpy().astype(int)
    conf = result.obb.conf.cpu().numpy()

    for box, c, s in zip(xywhr, cls, conf):
        cx, cy, w, h, angle = box
        ang = np.deg2rad(angle)
        cos_a, sin_a = np.cos(ang), np.sin(ang)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        corners = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
        pts = (corners @ rot.T + np.array([cx, cy])).astype(np.int32).reshape((-1, 1, 2))

        color = (0, 255, 255)
        label = f"{names.get(c, str(c))} {s:.2f}"
        cv2.polylines(out, [pts], True, color, 2)
        p0 = pts[0][0]
        cv2.putText(out, label, (int(p0[0]), max(0, int(p0[1]) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def main() -> int:
    args = build_parser().parse_args()

    if not Path(args.weights).exists():
        print(f"[错误] 权重文件不存在: {args.weights}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[信息] Using device: {device}")

    model = YOLO(args.weights).to(device)
    if args.fuse:
        print("[信息] 正在执行模型 fuse，这一步可能较慢...")
        model.fuse()
    else:
        print("[信息] 已跳过 fuse（如需可加 --fuse）")

    camera = MultiCameraSystem(args, enable_depth=False, enable_color=True)

    try:
        calib = camera.load_calibration()
        print(f"[信息] 标定已加载，外参单位统一到 {calib['target_unit']}")
        camera.start()
    except Exception as error:
        print(f"[错误] 初始化失败: {error}")
        camera.stop()
        return 1

    print("[信息] 实时检测已启动，按 q 或 ESC 退出。")

    names = model.names if isinstance(model.names, dict) else {}

    try:
        while True:
            data = camera.read_frames()
            if data is None:
                print("[警告] 帧读取失败，退出。")
                break

            rgb_frame = data["color"]
            ir_frame = data["lwir"]

            fused, rgb_net, ir_net = _fuse_rgb_ir(rgb_frame, ir_frame, args.imgsz)
            results = model.predict(fused, conf=args.conf, imgsz=args.imgsz, verbose=False, obb=args.obb)
            result = results[0]

            if args.obb:
                rgb_vis = _draw_obb(rgb_net, result, names)
                ir_vis = _draw_obb(ir_net, result, names)
            else:
                rgb_vis = _draw_hbb(rgb_net, result, names)
                ir_vis = _draw_hbb(ir_net, result, names)

            cv2.imshow("RGB Detection", rgb_vis)
            cv2.imshow("IR Detection", ir_vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
