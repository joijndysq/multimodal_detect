#!/usr/bin/env python3
"""Lepton 红外相机内参标定：棋盘格 12x9 25mm，自动保存样本。

核心策略：160x120 分辨率太小，角点间距只有~13像素，难以稳定检测。
解决方案：采集后立即 4x 放大到 640x480，在放大图像上检测角点（间距~52像素），
然后将检测到的角点坐标除以 4 映射回原始分辨率用于标定。
"""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import cv2
import yaml
from types import SimpleNamespace

__all__ = [
    "detect_board_corners",
    "create_board_world_points",
    "build_parser",
    "main",
]

LOG_PATH = Path(__file__).parent / "calib_lepton_intrinsics_gui.log"
BURST_FRAMES = 5  # 增加突发帧数，提高稳定性
INVERT_IR = False  # 改为白热模式
ROTATE_IR_DISPLAY = False  # Lepton 通常不需要旋转
_MAX_DRAW_FAILS = 5
DISPLAY_SCALE = 4  # 160x120 -> 640x480
MIN_AREA_RATIO = 0.3  # 棋盘格最小占比


def _log(msg: str) -> None:
    try:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


class _DrawState:
    def __init__(self, max_fails: int = _MAX_DRAW_FAILS) -> None:
        self.ir_logged: bool = False
        self.ir_fails: int = 0
        self.max_fails: int = max_fails
        self.draw_enabled: bool = True

    def record_fail(self, error: Exception) -> None:
        if not self.ir_logged:
            _log(f"[WARN] 绘制 IR 棋盘角点失败: {error}")
            self.ir_logged = True
        self.ir_fails += 1
        if self.ir_fails >= self.max_fails:
            self.draw_enabled = False
            _log("[WARN] 已禁用棋盘角点绘制，连续失败达到阈值")
            self.ir_fails = 0

    def record_success(self) -> None:
        self.ir_fails = 0


class _CalibSession:
    def __init__(self, min_samples: int) -> None:
        self.min_samples = min_samples
        self.object_points: List[np.ndarray] = []
        self.image_points: List[np.ndarray] = []
        self.reproj_errors: List[float] = []
        self.detected_count: int = 0
        self.rejected_count: int = 0
        self.last_error: float = 0.0
        self.draw_state = _DrawState()


class _UIState:
    done: bool = False
    min_reached: bool = False
    button: Tuple[int, int, int, int] = (0, 0, 0, 0)


class LeptonCamera:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.cap: Optional[cv2.VideoCapture] = None
        self.started = False

    def start(self) -> None:
        self.cap = cv2.VideoCapture(self.args.lepton_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开 Lepton 相机 video{self.args.lepton_index}")
        if self.args.lepton_width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.lepton_width)
        if self.args.lepton_height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.lepton_height)
        if self.args.lepton_fps > 0:
            self.cap.set(cv2.CAP_PROP_FPS, self.args.lepton_fps)
        self.started = True

    def read_frame(self) -> Optional[np.ndarray]:
        if not self.started:
            raise RuntimeError("相机尚未启动")
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        return frame

    def stop(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()
        self.started = False


def _preprocess_ir(ir_frame: np.ndarray, clahe: Any) -> np.ndarray:
    if ir_frame.ndim == 3:
        ir_gray = cv2.cvtColor(ir_frame, cv2.COLOR_BGR2GRAY)
    else:
        ir_gray = ir_frame.copy()

    if ir_gray.dtype != np.uint8:
        if ir_gray.dtype in (np.float32, np.float64):
            ir_gray = np.clip(ir_gray, 0, 255).astype(np.uint8)
        else:
            ir_gray = ir_gray.astype(np.uint8)

    # 白热模式优化：增强对比度
    ir_gray = clahe.apply(ir_gray)

    # 自动阈值调整，确保棋盘格有足够的对比度
    if INVERT_IR:
        # 白热模式：背景暗，棋盘格亮
        _, ir_gray = cv2.threshold(ir_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return ir_gray


def detect_board_corners(
    gray: np.ndarray,
    board_cols: int,
    board_rows: int,
    criteria=None,
) -> Tuple[bool, Optional[np.ndarray]]:
    if criteria is None:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if gray.dtype != np.uint8:
        if gray.dtype in (np.float32, np.float64):
            gray = np.clip(gray, 0, 255).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)

    ret, corners = cv2.findChessboardCorners(gray, (board_cols, board_rows), None)
    if not ret or corners is None or corners.size == 0:
        return False, None

    try:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    except Exception as e:
        _log(f"[WARN] cornerSubPix 失败: {e}")
        return False, None

    return True, corners


def create_board_world_points(
    board_cols: int, board_rows: int, square_size_mm: float
) -> np.ndarray:
    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2)
    objp *= square_size_mm / 1000.0
    return objp


def _to_points2d(corners_input: Any) -> Optional[np.ndarray]:
    if corners_input is None:
        return None
    try:
        arr = np.asarray(corners_input)
    except Exception:
        return None
    if arr.size == 0:
        return None
    try:
        if arr.ndim == 3 and arr.shape[1] == 1:
            arr = arr.reshape(-1, 2)
        elif arr.ndim == 2:
            arr = arr.reshape(-1, 2)
        else:
            return None
    except Exception:
        return None
    if arr.shape[1] != 2 or arr.shape[0] == 0:
        return None
    return arr.astype(np.float32)  # 改为 float32，calibrateCamera 需要


def _safe_draw_corners(
    img: np.ndarray,
    corners: np.ndarray,
    board_cols: int,
    board_rows: int,
    pattern_was_found: bool,
) -> Optional[np.ndarray]:
    try:
        expected = board_cols * board_rows
        if corners is None or corners.size == 0:
            return None
        arr = np.asarray(corners)
        if arr.ndim == 2:
            arr = arr.reshape(-1, 1, 2)
        elif arr.ndim == 3 and arr.shape[1] != 1:
            arr = arr.reshape(-1, 1, 2)
        if arr.shape[0] != expected:
            _log(f"[WARN] 角点数量不匹配: 期望 {expected}, 实际 {arr.shape[0]}")
            return None
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32)

        img_copy = img.copy()
        if img_copy.ndim == 2:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
        elif img_copy.shape[2] == 1:
            img_copy = cv2.cvtColor(img_copy, cv2.COLOR_GRAY2BGR)
        if img_copy.dtype != np.uint8:
            img_copy = np.clip(img_copy, 0, 255).astype(np.uint8)

        return cv2.drawChessboardCorners(
            img_copy, (board_cols, board_rows), arr, pattern_was_found
        )
    except Exception as e:
        _log(f"[WARN] drawChessboardCorners 失败: {e}")
        return None


def _compute_reproj_error(
    obj_pts: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    img_pts: np.ndarray,
) -> float:
    try:
        proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - img_pts, axis=1)
        return float(np.mean(err)) if err.size > 0 else 0.0
    except Exception:
        return float("inf")


def _save_intrinsics_yaml(
    out_path: str,
    K: np.ndarray,
    D: np.ndarray,
    avg_error: float,
    board_cols: int,
    board_rows: int,
    square_size_mm: float,
) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "thermal_camera": {
            "fx": float(K[0, 0]),
            "fy": float(K[1, 1]),
            "cx": float(K[0, 2]),
            "cy": float(K[1, 2]),
            "distortion_coefficients": D.reshape(-1).tolist(),
        },
        "calibration_info": {
            "board_cols": board_cols,
            "board_rows": board_rows,
            "square_size_mm": square_size_mm,
            "avg_reproj_error": round(avg_error, 4),
            "timestamp": int(time.time()),
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    _log(f"[INFO] 内参已保存到: {out_path}")


EMBED_CONFIG = True


def _build_default_config() -> dict:
    base = Path(__file__).resolve().parent
    return {
        "lepton_index": 1,
        "lepton_width": 160,
        "lepton_height": 120,
        "lepton_fps": 25,
        "board_cols": 9,
        "board_rows": 6,
        "square_size_mm": 25.0,
        "min_samples": 25,  # 增加最少样本数
        "output": str(base / "lepton35_calib.yaml"),
        "print_only": False,
    }


DEFAULT_CONFIG = _build_default_config()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lepton 红外相机内参标定（棋盘格 12x9 25mm）")
    parser.add_argument("--lepton-index", type=int, default=1, help="Lepton 相机设备编号")
    parser.add_argument("--lepton-width", type=int, default=160, help="Lepton 采集宽度")
    parser.add_argument("--lepton-height", type=int, default=120, help="Lepton 采集高度")
    parser.add_argument("--lepton-fps", type=int, default=25, help="Lepton 采集帧率")
    parser.add_argument("--board-cols", type=int, default=9, help="棋盘格角点列数（内点）")
    parser.add_argument("--board-rows", type=int, default=6, help="棋盘格角点行数（内点）")
    parser.add_argument("--square-size-mm", type=float, default=25.0, help="棋盘格边长 (mm)")
    parser.add_argument("--min-samples", type=int, default=25, help="最小样本数（建议>=25）")
    parser.add_argument("--output", type=str, default="lepton35_calib.yaml", help="内参输出 YAML 路径")
    parser.add_argument("--print-only", action="store_true", help="仅输出参数，不启动相机")
    return parser


def _make_on_mouse(ui_state: _UIState):
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            btn = param.get("button")
            if btn is None:
                return
            x1, y1, x2, y2 = btn
            if x1 <= x <= x2 and y1 <= y <= y2 and param.get("min_reached", False):
                param["done"] = True
    return on_mouse


def main() -> int:
    parser = build_parser()
    if EMBED_CONFIG:
        args = SimpleNamespace(**DEFAULT_CONFIG)
    else:
        args = parser.parse_args()

    if args.print_only:
        print("[INFO] 仅输出模式，不启动相机。")
        _log("[INFO] Print-only mode: not starting camera calibration.")
        return 0

    camera = LeptonCamera(args)
    try:
        camera.start()
    except Exception as e:
        print(f"[ERROR] 启动相机失败: {e}")
        camera.stop()
        return 1

    board_cols = int(args.board_cols)
    board_rows = int(args.board_rows)
    sq_mm = float(args.square_size_mm)
    min_samples = int(args.min_samples)
    obj_pts_template = create_board_world_points(board_cols, board_rows, sq_mm)

    session = _CalibSession(min_samples)
    ui_state = _UIState()
    clahe = cv2.createCLAHE(2.0, (8, 8))  # 适度增强对比度，白热模式下不需要太强

    burst_window: deque[np.ndarray] = deque(maxlen=BURST_FRAMES)

    window_name = "Lepton Calibration - 9x6 25mm"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 640, 480)

    on_mouse = _make_on_mouse(ui_state)

    print(f"[INFO] 标定模式：移动棋盘格，确保角点清晰。需要 {min_samples} 个样本。")
    print(f"[INFO] 策略：160x120 采集后 {DISPLAY_SCALE}x 放大到 {160*DISPLAY_SCALE}x{120*DISPLAY_SCALE} 检测")
    print(f"[INFO] 已改为白热模式（黑背景白棋盘格），使用9x6棋盘格")
    _log(f"[INFO] Calibration started. Board: {board_cols}x{board_rows}, Square: {sq_mm}mm")

    while True:
        try:
            frame = camera.read_frame()
        except Exception as e:
            _log(f"[ERROR] 读取帧失败: {e}")
            camera.stop()
            return 1

        if frame is None:
            cv2.waitKey(100)
            continue

        try:
            # 预处理：灰度 + CLAHE
            if frame.ndim == 3:
                ir_gray_small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                ir_gray_small = frame.copy()

            ir_gray_small = _preprocess_ir(ir_gray_small, clahe)

            # 关键：4x 放大到 640x480 后再检测
            ir_gray_big = cv2.resize(
                ir_gray_small,
                None,
                fx=DISPLAY_SCALE,
                fy=DISPLAY_SCALE,
                interpolation=cv2.INTER_LINEAR,
            )

        except Exception as e:
            _log(f"[WARN] 预处理失败: {e}")
            continue

        # 在放大图像上检测角点
        found, corners_raw = detect_board_corners(ir_gray_big, board_cols, board_rows)
        corners_big = _to_points2d(corners_raw) if found else None
        if corners_big is None:
            found = False

        # 突发采样：连续检测到 BURST_FRAMES 次后平均
        if found and corners_big is not None:
            session.detected_count += 1
            burst_window.append(corners_big.copy())

            if len(burst_window) == BURST_FRAMES:
                try:
                    shapes = [f.shape for f in burst_window]
                    if len(set(shapes)) == 1:
                        avg_corners_big = np.mean(np.array(burst_window), axis=0)
                        burst_window.clear()

                        # 映射回原始 160x120 分辨率用于标定
                        avg_corners_small = avg_corners_big / DISPLAY_SCALE

                        session.object_points.append(obj_pts_template.copy())
                        # imagePoints 需要是 vector<vector<Point2f>>
                        session.image_points.append(avg_corners_small.reshape(-1, 2).astype(np.float32))
                        session.reproj_errors.append(0.0)
                        n_accepted = len(session.object_points)
                        print(f"[INFO] 样本已接受: {n_accepted}/{min_samples}")
                        _log(f"[INFO] Sample accepted: {n_accepted}/{min_samples}")
                    else:
                        _log("[WARN] 突发帧形状不一致，跳过")
                except Exception as e:
                    _log(f"[WARN] 突发采样异常: {e}")
                    session.rejected_count += 1
                burst_window.clear()

        # 显示：使用放大后的图像
        try:
            disp = cv2.cvtColor(ir_gray_big, cv2.COLOR_GRAY2BGR)

            if session.draw_state.draw_enabled and found and corners_big is not None:
                result = _safe_draw_corners(disp, corners_big, board_cols, board_rows, True)
                if result is not None:
                    disp = result
                    session.draw_state.record_success()
                else:
                    session.draw_state.record_fail(Exception("绘制返回空"))

            if ROTATE_IR_DISPLAY:
                disp = cv2.rotate(disp, cv2.ROTATE_180)
            if INVERT_IR:
                disp = cv2.bitwise_not(disp)

        except Exception as e:
            _log(f"[WARN] 绘制显示失败: {e}")
            continue

        h, w = disp.shape[:2]
        btn_w, btn_h = 200, 60
        btn_x1 = w - btn_w - 30
        btn_y1 = h - btn_h - 30
        btn_x2 = w - 30
        btn_y2 = h - 30
        n_accepted = len(session.object_points)
        color_btn = (0, 255, 0) if n_accepted >= min_samples else (160, 160, 160)
        cv2.rectangle(disp, (btn_x1, btn_y1), (btn_x2, btn_y2), color_btn, -1)
        cv2.rectangle(disp, (btn_x1, btn_y1), (btn_x2, btn_y2), (255, 255, 255), 2)
        cv2.putText(
            disp,
            "DONE",
            (btn_x1 + 50, btn_y1 + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 0) if n_accepted >= min_samples else (200, 200, 200),
            2,
        )

        status_lines = [
            f"Samples: {n_accepted}/{min_samples}",
            f"Detected: {session.detected_count} | Rejected: {session.rejected_count}",
        ]
        for i, line in enumerate(status_lines):
            cv2.putText(
                disp,
                line,
                (20, 30 + i * 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

        guide_y = 30 + len(status_lines) * 30 + 20
        cv2.putText(
            disp,
            "ESC/q: quit",
            (20, guide_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        try:
            cv2.imshow(window_name, disp)
        except Exception as e:
            _log(f"[ERROR] cv2.imshow 失败: {e}")
            continue

        ui_state.button = (btn_x1, btn_y1, btn_x2, btn_y2)
        ui_state.min_reached = n_accepted >= min_samples
        callback_dict = {
            "done": False,
            "min_reached": ui_state.min_reached,
            "button": ui_state.button,
        }
        cv2.setMouseCallback(window_name, on_mouse, callback_dict)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if callback_dict.get("done", False):
            ui_state.done = True
            break

    n_accepted = len(session.object_points)
    if n_accepted < min_samples:
        print(f"[ERROR] 样本不足：{n_accepted} < {min_samples}，未保存内参。")
        camera.stop()
        return 0

    print("[INFO] 开始计算内参...")
    _log(f"[INFO] Running calibrateCamera with {n_accepted} samples")

    try:
        # 数据格式验证
        for i in range(n_accepted):
            print(f"[DEBUG] Sample {i}: obj_pts shape={session.object_points[i].shape}, img_pts shape={session.image_points[i].shape}")
            if session.image_points[i].shape[1] != 2:
                print(f"[ERROR] Sample {i} image points has wrong shape: {session.image_points[i].shape}")
                return 1

        obj_pts_list = session.object_points
        img_pts_list = session.image_points
        img_size = (int(args.lepton_width), int(args.lepton_height))

        # 使用基本的标定参数
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            obj_pts_list, img_pts_list, img_size, None, None,
            flags=cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5
        )
        if not ret:
            print("[ERROR] calibrateCamera 失败")
            camera.stop()
            return 1

        # 计算重投影误差
        total_error = 0.0
        for i in range(n_accepted):
            err = _compute_reproj_error(
                obj_pts_list[i], rvecs[i], tvecs[i], K, D, img_pts_list[i]
            )
            total_error += err
            session.reproj_errors[i] = err
        avg_error = total_error / n_accepted

        session.reproj_errors = [float(e) for e in session.reproj_errors[:n_accepted]]

    except Exception as e:
        print(f"[ERROR] 标定计算失败: {e}")
        _log(f"[ERROR] Calibration failed: {e}")
        camera.stop()
        return 1

    print(f"[自检] 内参矩阵 K:\n{K}")
    print(f"[自检] 畸变系数 D: {D.reshape(-1).tolist()}")

    # 输出每个样本的重投影误差
    print("\n[自检] 各样本重投影误差 (px):")
    for i in range(n_accepted):
        print(f"  样本 {i+1:2d}: {session.reproj_errors[i]:.3f}")

    print(f"\n[自检] 平均重投影误差: {avg_error:.3f} px")
    if session.reproj_errors:
        max_err = float(np.max(session.reproj_errors))
        min_err = float(np.min(session.reproj_errors))
        print(f"[自检] 最大重投影误差: {max_err:.3f} px")
        print(f"[自检] 最小重投影误差: {min_err:.3f} px")
        print(f"[自检] 重投影误差标准差: {np.std(session.reproj_errors):.3f} px")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    _save_intrinsics_yaml(
        str(out_path), K, D, avg_error, board_cols, board_rows, sq_mm
    )

    camera.stop()
    return 0


def _safe_main() -> int:
    try:
        return main()
    except Exception as e:
        _log(f"[FATAL] 未处理异常: {e}")
        print(f"Fatal: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(_safe_main())
