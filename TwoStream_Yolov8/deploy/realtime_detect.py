import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch

# Ensure custom ultralytics in this repo is used for custom model modules.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEPLOY_DIR = Path(__file__).resolve().parent
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from ultralytics import YOLO

from camera_init import MultiCameraSystem, build_parser as build_camera_parser

try:
    from multicam_sync_py import MultiCamSync as CppMultiCamSync
except ImportError:
    CppMultiCamSync = None


def build_parser() -> argparse.ArgumentParser:
    parser = build_camera_parser()
    parser.description = "实时融合检测：Lepton 红外 + RealSense RGB"
    parser.add_argument("--camera-backend", type=str, default="auto", choices=["auto", "python", "cpp"], help="相机后端")
    parser.add_argument("--weights", type=str, default=str(Path(__file__).resolve().parent / "best.pt"), help="模型权重路径")
    parser.add_argument("--imgsz", type=int, default=640, help="推理输入尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--obb", action="store_true", help="使用旋转框输出")
    parser.add_argument("--fuse", action="store_true", help="启动时执行模型 fuse（可能较慢）")
    parser.add_argument("--max-frame-miss", type=int, default=60, help="允许连续丢帧次数，超限后退出")
    parser.add_argument("--draw-xyz", action=argparse.BooleanOptionalAction, default=True, help="在检测结果上绘制目标采样点及三维坐标")
    parser.add_argument("--point-offset-x", type=float, default=0.0, help="采样点在框内的 x 偏移比例（相对宽度）")
    parser.add_argument("--point-offset-y", type=float, default=0.8, help="采样点在框内的 y 偏移比例（相对高度）")
    parser.add_argument("--ir-render-mode", type=str, default="whitehot", choices=["blackhot", "whitehot", "jet"], help="红外显示模式")
    return parser


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    img = np.ascontiguousarray(img)
    if img.dtype != np.uint8:
        if np.issubdtype(img.dtype, np.floating):
            img = np.clip(img, 0, 255).astype(np.uint8)
        elif img.dtype == np.uint16:
            img = cv2.convertScaleAbs(img, alpha=1.0 / 256.0)
        else:
            img = img.astype(np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _normalize_lwir_raw16(lwir_raw16: np.ndarray, invert: bool) -> np.ndarray:
    raw = np.asarray(lwir_raw16)
    if raw.ndim == 3:
        raw = raw.squeeze()
    if raw.shape[0] > 120:
        raw = raw[:120, :]
    raw = raw.astype(np.float32, copy=False)
    min_val = float(np.percentile(raw, 2))
    max_val = float(np.percentile(raw, 98))
    scale = max(max_val - min_val, 1.0)
    img = np.clip((raw - min_val) / scale * 255.0, 0, 255).astype(np.uint8)
    if invert:
        img = 255 - img
    img = cv2.flip(img, -1)
    return cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)


def _render_ir_for_display(data: dict, mode: str) -> np.ndarray:
    lwir_raw16 = data.get("lwir_raw16")
    if lwir_raw16 is not None:
        if mode == "whitehot":
            gray = _normalize_lwir_raw16(lwir_raw16, invert=False)
            return gray
        if mode == "blackhot":
            gray = _normalize_lwir_raw16(lwir_raw16, invert=True)
            return gray
        gray = _normalize_lwir_raw16(lwir_raw16, invert=False)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    lwir = data["lwir"]
    if mode == "jet":
        gray = lwir if lwir.ndim == 2 else cv2.cvtColor(lwir, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    return lwir


def _prepare_ir_for_model(data: dict) -> np.ndarray:
    lwir_raw16 = data.get("lwir_raw16")
    if lwir_raw16 is not None:
        return _normalize_lwir_raw16(lwir_raw16, invert=True)
    lwir = data["lwir"]
    if lwir.ndim == 3:
        return cv2.cvtColor(lwir, cv2.COLOR_BGR2GRAY)
    return lwir


def _fuse_rgb_ir(rgb: np.ndarray, ir: np.ndarray, imgsz: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb_resized = cv2.resize(rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    ir_resized = cv2.resize(ir, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    ir_resized = _ensure_bgr(ir_resized)
    fused = np.concatenate((rgb_resized, ir_resized), axis=2)
    return fused, rgb_resized, ir_resized


def _rgb_nonzero_ratio(img: np.ndarray | None) -> float:
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return 0.0
    if img.ndim == 2:
        valid = img > 0
    else:
        valid = np.any(img > 0, axis=2)
    return float(np.count_nonzero(valid)) / float(valid.size)


def _select_rgb_for_model(data: dict) -> tuple[np.ndarray, str]:
    color = data["color"]
    rgb_aligned = data.get("rgb_aligned")
    coverage = _rgb_nonzero_ratio(rgb_aligned)
    if isinstance(rgb_aligned, np.ndarray) and rgb_aligned.size > 0 and coverage >= 0.15:
        return rgb_aligned, f"aligned({coverage:.2f})"
    return color, f"raw({coverage:.2f})"


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


def _draw_xyz_points(
    img: np.ndarray,
    boxes_xywh: np.ndarray,
    depth_frame_obj,
    depth_raw,
    depth_scale: float,
    depth_intrin,
    color_shape: tuple[int, int],
    crop_left: int,
    crop_bottom: int,
    point_offset_x: float,
    point_offset_y: float,
) -> np.ndarray:
    """在检测框采样点绘制3D坐标，兼容 HBB 与 OBB。

    boxes_xywh: [N, >=4] 每行前四列为 (cx, cy, w, h)。
    坐标变换: RealSense (x→右, y→下, z→前) → (X=前, Y=左, Z=上)，单位 m。
    color_shape: 裁减后 RGB 的尺寸 (h, w)，用于从网络输出映射到裁减图。
    crop_left / crop_bottom: RGB 裁减量，用于从裁减图坐标恢复到原始深度图坐标。
    """
    out = img.copy()
    if len(boxes_xywh) == 0 or depth_intrin is None:
        return out

    net_h, net_w = out.shape[:2]
    color_h, color_w = color_shape

    sx = float(color_w) / float(net_w)
    sy = float(color_h) / float(net_h)

    for box in boxes_xywh:
        cx, cy, w, h = box[0], box[1], box[2], box[3]
        ux = int(round(cx + point_offset_x * w))
        uy = int(round(cy + point_offset_y * h))
        if not (0 <= ux < net_w and 0 <= uy < net_h):
            continue

        # 映射到裁减后 RGB 坐标，再加上偏移恢复到原始深度图坐标
        ux_cropped = int(round(ux * sx))
        uy_cropped = int(round(uy * sy))
        ux_depth = ux_cropped + crop_left
        uy_depth = uy_cropped  # 裁 bottom，上边不动

        img_h_uncropped = 480  # 深度图高度
        img_w_uncropped = 640  # 深度图宽度
        ux_depth = min(max(ux_depth, 0), img_w_uncropped - 1)
        uy_depth = min(max(uy_depth, 0), img_h_uncropped - 1)

        dis = 0.0
        if depth_frame_obj is not None:
            dis = float(depth_frame_obj.get_distance(ux_depth, uy_depth))
        elif depth_raw is not None:
            raw_value = float(depth_raw[uy_depth, ux_depth])
            dis = raw_value * float(depth_scale)
        if dis <= 0.0:
            continue

        if isinstance(depth_intrin, dict):
            fx = float(depth_intrin["fx"])
            fy = float(depth_intrin["fy"])
            cxi = float(depth_intrin["cx"])
            cyi = float(depth_intrin["cy"])
            xyz = np.array(
                [
                    (ux_depth - cxi) * dis / fx,
                    (uy_depth - cyi) * dis / fy,
                    dis,
                ],
                dtype=np.float32,
            )
        else:
            xyz = rs.rs2_deproject_pixel_to_point(depth_intrin, [ux_depth, uy_depth], dis)
            xyz = np.asarray(xyz, dtype=np.float32)
        xyz = np.round(xyz, 3)
        # 坐标变换：RealSense (x→右, y→下, z→前) → 更直观的 (X=前, Y=左, Z=上)
        rx, ry, rz = float(xyz[0]), float(xyz[1]), float(xyz[2])
        xyz = np.array([rz, -rx, -ry], dtype=np.float32)
        xyz = np.round(xyz, 3)

        cv2.circle(out, (ux, uy), 4, (255, 255, 255), -1)
        cv2.putText(
            out,
            f"[{xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}]m",
            (max(0, ux - 80), max(16, uy - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (225, 255, 255),
            1,
            cv2.LINE_AA,
        )

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

    use_cpp_backend = args.camera_backend == "cpp" or (args.camera_backend == "auto" and CppMultiCamSync is not None)
    camera = None
    cpp_camera = None

    try:
        if use_cpp_backend:
            if CppMultiCamSync is None:
                raise RuntimeError("未找到 multicam_sync_py 模块，请先编译 C++ Python 扩展")
            cpp_camera = CppMultiCamSync(
                serial=args.serial,
                lwir_index=args.lwir_index,
                lwir_width=args.lwir_width,
                lwir_height=args.lwir_height,
                lwir_fps=args.lwir_fps,
                depth_width=args.depth_width,
                depth_height=args.depth_height,
                depth_fps=args.depth_fps,
                color_width=args.color_width,
                color_height=args.color_height,
                color_fps=args.color_fps,
                sync_max_delta_ms=args.sync_max_delta_ms,
                sync_buffer_size=args.sync_buffer_size,
                white_hot=(args.ir_render_mode != "blackhot"),
                d435_calib=args.d435_calib,
                lepton_calib=args.lepton_calib,
                stereo_extrinsics=args.stereo_extrinsics,
                align_depth_to_lwir=args.align_depth_to_lwir,
                align_rgb_to_lwir=args.align_rgb_to_lwir,
                align_skip_frames=args.align_skip_frames,
            )
            cpp_camera.start()
            print("[信息] 已启用 C++ 相机后端")
        else:
            camera = MultiCameraSystem(args)
            calib = camera.load_calibration()
            print(f"[信息] 标定已加载，外参单位统一到 {calib['target_unit']}")
            camera.start()
            print("[信息] 已启用 Python 相机后端")
    except Exception as error:
        print(f"[错误] 初始化失败: {error}")
        if camera is not None:
            camera.stop()
        if cpp_camera is not None:
            cpp_camera.stop()
        return 1

    print("[信息] 实时检测已启动，按 q 或 ESC 退出。")
    print("[信息] 红外模式切换: 1=黑热, 2=白热, 3=伪彩")

    names = model.names if isinstance(model.names, dict) else {}
    miss_count = 0
    ir_render_mode = args.ir_render_mode
    last_stats_print = 0
    last_rgb_source = ""

    try:
        while True:
            data = cpp_camera.read(1000) if cpp_camera is not None else camera.read_frames()
            if not data:
                miss_count += 1
                if miss_count == 1 or miss_count % 10 == 0:
                    print(f"[警告] 帧读取失败，已连续 {miss_count} 次。")
                if miss_count >= max(1, args.max_frame_miss):
                    print("[错误] 连续丢帧过多，退出。")
                    break
                continue

            if cpp_camera is not None and not data.get("synced", True):
                miss_count += 1
                rgb_preview = data.get("color")
                ir_preview = _render_ir_for_display(data, ir_render_mode) if data.get("lwir") is not None else None
                if rgb_preview is not None and isinstance(rgb_preview, np.ndarray) and rgb_preview.size > 0:
                    cv2.imshow("RGB Input", rgb_preview)
                if ir_preview is not None and isinstance(ir_preview, np.ndarray) and ir_preview.size > 0:
                    cv2.imshow("IR Input", ir_preview)

                if miss_count == 1 or miss_count % 10 == 0:
                    stats = data.get("stats") or {}
                    print(
                        "[警告] 尚未同步到成对帧: "
                        f"lwir={stats.get('lwir_frames', 0)} "
                        f"rs={stats.get('rs_frames', 0)} "
                        f"matched={stats.get('matched_frames', 0)} "
                        f"timeouts={stats.get('sync_timeouts', 0)}"
                    )
                key = cv2.waitKey(1) & 0xFF
                if key == ord("1"):
                    ir_render_mode = "blackhot"
                elif key == ord("2"):
                    ir_render_mode = "whitehot"
                elif key == ord("3"):
                    ir_render_mode = "jet"
                if key in (ord("q"), 27):
                    break
                if miss_count >= max(1, args.max_frame_miss):
                    print("[错误] 连续未同步过多，退出。")
                    break
                continue

            if "color" not in data or "lwir" not in data:
                miss_count += 1
                if miss_count == 1 or miss_count % 10 == 0:
                    print(f"[警告] 帧内容不完整，已连续 {miss_count} 次。")
                if miss_count >= max(1, args.max_frame_miss):
                    print("[错误] 连续丢帧过多，退出。")
                    break
                continue

            miss_count = 0

            rgb_preview = data["color"]
            rgb_frame, rgb_source = _select_rgb_for_model(data)
            ir_frame = _prepare_ir_for_model(data)
            ir_display = _render_ir_for_display(data, ir_render_mode)
            depth_frame_obj = data.get("depth_frame_obj")
            depth_raw = data.get("depth_raw")
            depth_scale = float(data.get("depth_scale", 0.001))
            depth_intrin = data.get("depth_intrin")

            if rgb_source != last_rgb_source:
                print(f"[信息] 当前 RGB 来源: {rgb_source}")
                last_rgb_source = rgb_source

            fused, rgb_net, ir_net = _fuse_rgb_ir(rgb_frame, ir_frame, args.imgsz)
            ir_display_net = cv2.resize(_ensure_bgr(ir_display), (args.imgsz, args.imgsz), interpolation=cv2.INTER_LINEAR)
            results = model.predict(fused, conf=args.conf, imgsz=args.imgsz, verbose=False, obb=args.obb)
            result = results[0]

            if args.obb:
                rgb_vis = _draw_obb(rgb_net, result, names)
                ir_vis = _draw_obb(ir_display_net, result, names)
            else:
                rgb_vis = _draw_hbb(rgb_net, result, names)
                ir_vis = _draw_hbb(ir_display_net, result, names)

            if args.draw_xyz:
                # 提取检测框：OBB 和 HBB 前四列均为 (cx, cy, w, h)
                if args.obb and result.obb is not None and len(result.obb) > 0:
                    boxes_xywh = result.obb.xywhr.cpu().numpy()
                elif result.boxes is not None and len(result.boxes) > 0:
                    boxes_xywh = result.boxes.xywh.cpu().numpy()
                else:
                    boxes_xywh = np.array([], dtype=np.float32)

                color_shape = (int(rgb_frame.shape[0]), int(rgb_frame.shape[1]))
                crop_left = int(getattr(args, "crop_rgb_left", 0)) if getattr(args, "crop_align", False) else 0
                crop_bottom = int(getattr(args, "crop_rgb_bottom", 0)) if getattr(args, "crop_align", False) else 0
                rgb_vis = _draw_xyz_points(
                    rgb_vis,
                    boxes_xywh,
                    depth_frame_obj,
                    depth_raw,
                    depth_scale,
                    depth_intrin,
                    color_shape,
                    crop_left,
                    crop_bottom,
                    args.point_offset_x,
                    args.point_offset_y,
                )

            cv2.imshow("RGB Input", rgb_preview)
            cv2.imshow("IR Input", ir_display)
            cv2.imshow("RGB Detection", rgb_vis)
            cv2.imshow("IR Detection", ir_vis)

            stats = data.get("stats") or {}
            if stats and stats.get("matched_frames", 0) >= last_stats_print + 30:
                last_stats_print = stats.get("matched_frames", 0)
                print(
                    "[信息] 同步统计: "
                    f"lwir={stats.get('lwir_frames', 0)} "
                    f"rs={stats.get('rs_frames', 0)} "
                    f"matched={stats.get('matched_frames', 0)} "
                    f"timeouts={stats.get('sync_timeouts', 0)} "
                    f"delta={float(data.get('sync_delta_ms', 0.0)):.1f}ms"
                )

            key = cv2.waitKey(1) & 0xFF
            if key == ord("1"):
                ir_render_mode = "blackhot"
            elif key == ord("2"):
                ir_render_mode = "whitehot"
            elif key == ord("3"):
                ir_render_mode = "jet"
            if key in (ord("q"), 27):
                break
    finally:
        if camera is not None:
            camera.stop()
        if cpp_camera is not None:
            cpp_camera.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
