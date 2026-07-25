#!/usr/bin/env python3
"""外参标定：LWIR 与 RealSense RGB 的棋盘格外参标定，自动保存样本并支持通过点击确定完成。

- 捕获 LWIR 与 RGB 的棋盘格角点，当两边均检测到角点时自动保存为一个样本。
- 关键改进：IR 图像 4x 放大到 640×480 后检测，提高角点检测稳定性。
- 当样本数量达到最小阈值时，显示一个"确定"按钮，点击即可结束标定并保存外参。
- 计算 RGB-to-IR 的外参：T_rgb_to_ir = T_rgb_board * inv(T_ir_board)，对多组样本取平均后输出。
- 输出 YAML 结构：{ target_unit, extrinsics_rgb_to_ir: { R, t, source_unit } }
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

LOG_PATH = Path(__file__).parent / "calib_extrinsics_gui.log"
REPROJ_ERR_THRESH = 2.0  # 外参标定重投影误差阈值
BURST_FRAMES = 3
INVERT_IR = True
ROTATE_IR_DISPLAY = True
CORNER_DRAW_ENABLED = True
_MAX_DRAW_FAILS = 5
DISPLAY_SCALE = 4  # IR 放大倍数


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
        self.rgb_logged: bool = False
        self.ir_logged: bool = False
        self.rgb_fails: int = 0
        self.ir_fails: int = 0
        self.max_fails: int = max_fails
        self.draw_enabled: bool = True

    def record_fail(self, channel: str, error: Exception) -> None:
        if channel == "rgb":
            if not self.rgb_logged:
                _log(f"[WARN] 绘制 RGB 棋盘角点失败: {error}")
                self.rgb_logged = True
            self.rgb_fails += 1
            if self.rgb_fails >= self.max_fails:
                self.draw_enabled = False
                _log("[WARN] 已禁用棋盘角点绘制，连续失败达到阈值")
                self.rgb_fails = 0
        else:
            if not self.ir_logged:
                _log(f"[WARN] 绘制 IR 棋盘角点失败: {error}")
                self.ir_logged = True
            self.ir_fails += 1
            if self.ir_fails >= self.max_fails:
                self.draw_enabled = False
                _log("[WARN] 已禁用棋盘角点绘制，连续失败达到阈值")
                self.ir_fails = 0

    def record_success(self, channel: str) -> None:
        if channel == "rgb":
            self.rgb_fails = 0
        else:
            self.ir_fails = 0


class _CalibSession:
    def __init__(self, min_samples: int) -> None:
        self.min_samples = min_samples
        self.samples_rgb: List[np.ndarray] = []
        self.samples_ir: List[np.ndarray] = []
        self.samples_obj: List[np.ndarray] = []
        self.reproj_rgb_errors: List[float] = []
        self.reproj_ir_errors: List[float] = []
        self.detected_count: int = 0
        self.rejected_count: int = 0
        self.last_error: float = 0.0
        self.draw_state = _DrawState()


class _UIState:
    done: bool = False
    min_reached: bool = False
    button: Tuple[int, int, int, int] = (0, 0, 0, 0)


class StereoCameraPair:
    """Manages LWIR (OpenCV VideoCapture) + RealSense RGB pipeline."""
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.lwir_cap: Optional[cv2.VideoCapture] = None
        self.pipeline: Any = None
        self.started = False

    def start(self) -> None:
        self.lwir_cap = cv2.VideoCapture(self.args.lwir_index)
        if not self.lwir_cap.isOpened():
            raise RuntimeError(f"无法打开 LWIR 摄像头 video{self.args.lwir_index}")
        if self.args.lwir_width > 0:
            self.lwir_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.lwir_width)
        if self.args.lwir_height > 0:
            self.lwir_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.lwir_height)
        if self.args.lwir_fps > 0:
            self.lwir_cap.set(cv2.CAP_PROP_FPS, self.args.lwir_fps)

        import pyrealsense2 as rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.args.serial:
            config.enable_device(self.args.serial)
        config.enable_stream(
            rs.stream.color,
            self.args.color_width,
            self.args.color_height,
            rs.format.bgr8,
            self.args.color_fps,
        )
        self.pipeline.start(config)
        self.started = True

    def read_frames(self) -> Optional[dict[str, np.ndarray]]:
        if not self.started:
            raise RuntimeError("相机尚未启动")

        ret, lwir_frame = self.lwir_cap.read()
        if not ret or lwir_frame is None:
            return None

        import pyrealsense2 as rs
        try:
            frames = self.pipeline.wait_for_frames(
                timeout_ms=max(1, int(self.args.rs_timeout_ms))
            )
        except RuntimeError:
            return None

        color_frame = frames.get_color_frame()
        if not color_frame:
            return None

        rgb_arr = np.asarray(color_frame.get_data())
        return {"lwir": lwir_frame, "color": rgb_arr}

    def stop(self) -> None:
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None
        if self.lwir_cap is not None:
            self.lwir_cap.release()
            self.lwir_cap = None
        cv2.destroyAllWindows()
        self.started = False


def _preprocess_for_detection(
    rgb_frame: np.ndarray,
    lwir_frame: np.ndarray,
    clahe: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    if rgb_frame.ndim == 3:
        rgb_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)
    else:
        rgb_gray = rgb_frame

    if lwir_frame.ndim == 3:
        lwir_gray = cv2.cvtColor(lwir_frame, cv2.COLOR_BGR2GRAY)
    else:
        lwir_gray = lwir_frame

    if lwir_gray.dtype != np.uint8:
        if lwir_gray.dtype in (np.float32, np.float64):
            lwir_gray = np.clip(lwir_gray, 0, 255).astype(np.uint8)
        else:
            lwir_gray = lwir_gray.astype(np.uint8)

    # 关键：立即放大 IR 4x 到 640×480 再检测
    lwir_gray_big = cv2.resize(
        lwir_gray, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE,
        interpolation=cv2.INTER_LINEAR
    )

    rgb_gray = clahe.apply(rgb_gray)
    lwir_gray_big = clahe.apply(lwir_gray_big)

    return rgb_gray, lwir_gray_big


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


def _to_points2d(corners: Any) -> Optional[np.ndarray]:
    if corners is None:
        return None
    try:
        arr = np.asarray(corners)
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
    return arr.astype(np.float64)


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
    Xb: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    pts2: np.ndarray,
) -> float:
    try:
        proj, _ = cv2.projectPoints(Xb, rvec, tvec, K, D)
        proj = proj.reshape(-1, 2)
        err = np.linalg.norm(proj - pts2, axis=1)
        return float(np.mean(err)) if err.size > 0 else 0.0
    except Exception:
        return float("inf")


def _select_best_ir_order(
    Xb: np.ndarray,
    x_rgb: np.ndarray,
    x_ir: np.ndarray,
    K_rgb: np.ndarray,
    D_rgb: np.ndarray,
    K_ir: np.ndarray,
    D_ir: np.ndarray,
    reproj_rgb_errors: List[float],
    reproj_ir_errors: List[float],
) -> Optional[Tuple]:
    if Xb is None or x_rgb is None or x_ir is None:
        return None
    if Xb.size == 0 or x_rgb.size == 0 or x_ir.size == 0:
        return None
    n = min(Xb.shape[0], x_rgb.shape[0], x_ir.shape[0])
    if n <= 0:
        return None
    Xb = Xb[:n]
    x_rgb = x_rgb[:n]
    x_ir = x_ir[:n]

    best = None
    best_err = float("inf")
    best_rgb_err = 0.0
    best_ir_err = 0.0

    try:
        # 使用 SOLVEPNP_ITERATIVE 代替 RANSAC，更稳定
        ret_rgb, rvec_rgb, tvec_rgb = cv2.solvePnP(
            Xb, x_rgb, K_rgb, D_rgb, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if ret_rgb:
            ret_ir, rvec_ir, tvec_ir = cv2.solvePnP(
                Xb, x_ir, K_ir, D_ir, flags=cv2.SOLVEPNP_ITERATIVE
            )
            if ret_ir:
                mean_err_rgb = _compute_reproj_error(Xb, rvec_rgb, tvec_rgb, K_rgb, D_rgb, x_rgb)
                mean_err_ir = _compute_reproj_error(Xb, rvec_ir, tvec_ir, K_ir, D_ir, x_ir)
                best = (rvec_rgb, tvec_rgb, rvec_ir, tvec_ir, x_ir)
                best_err = mean_err_rgb + mean_err_ir
                best_rgb_err = mean_err_rgb
                best_ir_err = mean_err_ir

                # 尝试反转 IR 角点顺序
                x_ir_rev = x_ir[::-1].copy()
                ret_ir_rev, rvec_ir_rev, tvec_ir_rev = cv2.solvePnP(
                    Xb, x_ir_rev, K_ir, D_ir, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if ret_ir_rev:
                    mean_err_ir_rev = _compute_reproj_error(Xb, rvec_ir_rev, tvec_ir_rev, K_ir, D_ir, x_ir_rev)
                    if (mean_err_rgb + mean_err_ir_rev) < best_err:
                        best = (rvec_rgb, tvec_rgb, rvec_ir_rev, tvec_ir_rev, x_ir_rev)
                        best_err = mean_err_rgb + mean_err_ir_rev
                        best_rgb_err = mean_err_rgb
                        best_ir_err = mean_err_ir_rev
    except Exception as e:
        _log(f"[WARN] PnP 计算失败: {e}")
        best = None

    if best is None:
        return None

    reproj_rgb_errors.append(best_rgb_err)
    reproj_ir_errors.append(best_ir_err)
    return best


def _homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    tr = float(R[0, 0] + R[1, 1] + R[2, 2])
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def _rot_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def _average_rotations(mats: List[np.ndarray]) -> np.ndarray:
    if len(mats) == 0:
        raise ValueError("no rotations to average")
    if len(mats) == 1:
        return mats[0].copy()

    base_q = _quat_from_R(mats[0])
    quats: list[np.ndarray] = []
    for R in mats:
        q = _quat_from_R(R)
        if np.dot(q, base_q) < 0:
            q = -q
        quats.append(q)

    Q_stack = np.array(quats)
    N = Q_stack.T @ Q_stack

    eigenvalues, eigenvectors = np.linalg.eigh(N)

    q_avg = eigenvectors[:, -1]
    q_avg = q_avg / np.linalg.norm(q_avg)
    return _rot_from_quat(q_avg)


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _to_camera_matrix(cam: dict[str, Any]) -> np.ndarray:
    return np.array(
        [
            [float(cam["fx"]), 0.0, float(cam["cx"])],
            [0.0, float(cam["fy"]), float(cam["cy"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _to_dist_coeffs(cam: dict[str, Any]) -> np.ndarray:
    coeffs = cam.get("distortion_coefficients", [0.0, 0.0, 0.0, 0.0, 0.0])
    return np.asarray(coeffs, dtype=np.float64).reshape(-1)


def _extract_KD_from_dict(
    cam_dict: dict,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    if not isinstance(cam_dict, dict):
        return None, None

    if all(k in cam_dict for k in ("fx", "fy", "cx", "cy")):
        return _to_camera_matrix(cam_dict), _to_dist_coeffs(cam_dict)

    if "camera_matrix" in cam_dict:
        cm = cam_dict["camera_matrix"]
        try:
            cmat = np.array(cm, dtype=float)
            if cmat.shape == (3, 3):
                K = cmat
            elif cmat.size == 9:
                K = cmat.reshape(3, 3)
            else:
                K = None
        except Exception:
            K = None
        D = cam_dict.get("distortion_coefficients", cam_dict.get("distCoeffs", None))
        if K is not None:
            if D is not None:
                D = _to_dist_coeffs(D) if not isinstance(D, np.ndarray) else D
            else:
                D = np.zeros(5, dtype=np.float64)
            return K, np.asarray(D, dtype=np.float64).reshape(-1)

    if "intrinsics" in cam_dict and isinstance(cam_dict["intrinsics"], dict):
        intr = cam_dict["intrinsics"]
        if "K" in intr:
            K = np.asarray(intr["K"], dtype=np.float64).reshape(3, 3)
            D = np.asarray(
                intr.get("D", intr.get("distortion_coefficients", [0, 0, 0, 0, 0])),
                dtype=np.float64,
            ).reshape(-1)
            return K, D

    return None, None


def _load_rgb_ir_intrinsics(
    d435_yaml: str, lepton_yaml: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d435 = _load_yaml(d435_yaml)
    lepton = _load_yaml(lepton_yaml)
    try:
        rgb_cam = d435["color_camera"]
        ir_cam = lepton["thermal_camera"]
        K_rgb = _to_camera_matrix(rgb_cam)
        D_rgb = _to_dist_coeffs(rgb_cam)
        K_ir = _to_camera_matrix(ir_cam)
        D_ir = _to_dist_coeffs(ir_cam)
        return K_rgb, D_rgb, K_ir, D_ir
    except Exception:
        pass
    K_rgb, D_rgb = _extract_KD_from_dict(d435)
    K_ir, D_ir = _extract_KD_from_dict(lepton)
    if K_rgb is None or K_ir is None or D_rgb is None or D_ir is None:
        raise RuntimeError("无法从 YAML 提取内参。请提供有效的内参 YAML。")
    return K_rgb, D_rgb, K_ir, D_ir


def _save_extrinsics_yaml(out_path: str, R: np.ndarray, t: np.ndarray, unit: str) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "target_unit": unit,
        "extrinsics_rgb_to_ir": {
            "R": R.tolist(),
            "t": t.tolist(),
            "source_unit": unit,
        },
        "timestamp": int(time.time()),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    _log(f"[INFO] 外参已保存到: {out_path}")


EMBED_CONFIG = True


def _build_default_config() -> dict:
    base = Path(__file__).resolve().parent
    return {
        "d435_calib": str(base / "d435if_calib.yaml"),
        "lepton_calib": str(base / "lepton35_calib.yaml"),
        "output": str(base / "calib_extrinsics.yaml"),
        "lwir_index": 1,
        "lwir_width": 160,
        "lwir_height": 120,
        "lwir_fps": 25,
        "rs_timeout_ms": 5000,
        "board_cols": 9,
        "board_rows": 6,
        "square_size_mm": 25.0,
        "min_samples": 12,
        "print_only": False,
        "serial": "",
        "color_width": 640,
        "color_height": 480,
        "color_fps": 30,
    }


DEFAULT_CONFIG = _build_default_config()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LWIR + RGB 外参标定（棋盘格）")
    parser.add_argument("--d435-calib", type=str, required=True, help="RealSense D435 的标定 YAML 路径")
    parser.add_argument("--lepton-calib", type=str, required=True, help="Lepton 的标定 YAML 路径")
    parser.add_argument(
        "--stereo-extrinsics", type=str, default="",
        help="两相机外参参考 YAML 路径（可选）",
    )
    parser.add_argument("--output", type=str, default="calib_extrinsics.yaml", help="外参输出 YAML 路径")
    parser.add_argument("--lwir-index", type=int, default=1, help="LWIR 相机设备编号")
    parser.add_argument("--lwir-width", type=int, default=160, help="LWIR 采集宽度")
    parser.add_argument("--lwir-height", type=int, default=120, help="LWIR 采集高度")
    parser.add_argument("--lwir-fps", type=int, default=25, help="LWIR 采集帧率")
    parser.add_argument("--rs-timeout-ms", type=int, default=5000, help="RealSense 超时（毫秒）")
    parser.add_argument("--board-cols", type=int, default=9, help="棋盘格角点列数")
    parser.add_argument("--board-rows", type=int, default=6, help="棋盘格角点行数")
    parser.add_argument("--square-size-mm", type=float, default=25.0, help="棋盘格边长 (mm)")
    parser.add_argument("--min-samples", type=int, default=12, help="最小样本数")
    parser.add_argument("--print-only", action="store_true", help="仅输出内参，不启动相机")
    parser.add_argument("--serial", type=str, default="", help="RealSense 设备序列号")
    parser.add_argument("--color-width", type=int, default=640, help="RGB 流宽度")
    parser.add_argument("--color-height", type=int, default=480, help="RGB 流高度")
    parser.add_argument("--color-fps", type=int, default=30, help="RGB 流帧率")
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

    try:
        K_rgb, D_rgb, K_ir, D_ir = _load_rgb_ir_intrinsics(args.d435_calib, args.lepton_calib)
    except Exception as e:
        print(f"[ERROR] 加载内参失败: {e}")
        return 1

    board_cols = int(args.board_cols)
    board_rows = int(args.board_rows)
    sq_mm = float(args.square_size_mm)
    min_samples = int(args.min_samples)
    obj_pts = create_board_world_points(board_cols, board_rows, sq_mm)

    camera = StereoCameraPair(args)
    try:
        camera.start()
    except Exception as e:
        print(f"[ERROR] 启动相机失败: {e}")
        camera.stop()
        return 1

    session = _CalibSession(min_samples)
    ui_state = _UIState()
    clahe = cv2.createCLAHE(2.0, (8, 8))

    burst_window_rgb: deque[np.ndarray] = deque(maxlen=BURST_FRAMES)
    burst_window_ir: deque[np.ndarray] = deque(maxlen=BURST_FRAMES)

    window_name = "Extrinsic Calib"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 480)  # 640+640

    on_mouse = _make_on_mouse(ui_state)

    print(f"[INFO] 进入标定模式：请移动棋盘格，确保角点清晰。达到 {min_samples} 个样本后可点击「确定」停止标定。")
    print(f"[INFO] 策略：IR 160x120 采集后 {DISPLAY_SCALE}x 放大到 {160*DISPLAY_SCALE}x{120*DISPLAY_SCALE} 检测")
    _log(f"[INFO] Calibration mode started. Board: {board_cols}x{board_rows}, Square: {sq_mm}mm")

    while True:
        try:
            frame = camera.read_frames()
        except Exception as e:
            _log(f"[ERROR] 读取帧失败: {e}")
            camera.stop()
            return 1

        if frame is None:
            cv2.waitKey(100)
            continue

        try:
            lwir = frame["lwir"]
            rgb = frame["color"]
            if lwir is None or rgb is None:
                _log("[WARN] 帧数据为空")
                continue
        except KeyError as e:
            _log(f"[WARN] 帧缺少键: {e}")
            continue

        try:
            rgb_gray, lwir_gray_big = _preprocess_for_detection(rgb, lwir, clahe)
        except Exception as e:
            _log(f"[WARN] 预处理失败: {e}")
            continue

        # 在放大后的图像上检测角点（更稳定）
        found_rgb, corners_rgb_raw = detect_board_corners(rgb_gray, board_cols, board_rows)
        corners_rgb = _to_points2d(corners_rgb_raw) if found_rgb else None
        if corners_rgb is None:
            found_rgb = False

        found_ir = False
        corners_ir_big: Optional[np.ndarray] = None

        if found_rgb and corners_rgb is not None:
            try:
                pts_rgb = corners_rgb.reshape(-1, 2)
                xs, ys = pts_rgb[:, 0], pts_rgb[:, 1]
                minx, maxx = int(xs.min()), int(xs.max())
                miny, maxy = int(ys.min()), int(ys.max())

                # 在 IR 放大图像上定义 ROI
                minx_ir = int(minx * (160.0 / 640.0))  # 映射回 IR 原始坐标再放大
                maxx_ir = int(maxx * (160.0 / 640.0))
                miny_ir = int(miny * (120.0 / 480.0))
                maxy_ir = int(maxy * (120.0 / 480.0))

                width_ir = max(1, maxx_ir - minx_ir)
                height_ir = max(1, maxy_ir - miny_ir)
                pad_x = max(20, int(0.2 * width_ir))
                pad_y = max(20, int(0.2 * height_ir))

                x0 = max(0, minx_ir - pad_x)
                y0 = max(0, miny_ir - pad_y)
                x1 = min(lwir_gray_big.shape[1], maxx_ir + pad_x)
                y1 = min(lwir_gray_big.shape[0], maxy_ir + pad_y)

                if y1 > y0 and x1 > x0:
                    lwir_roi = lwir_gray_big[y0:y1, x0:x1]
                    found_ir_roi, corners_ir_roi_raw = detect_board_corners(
                        lwir_roi, board_cols, board_rows
                    )
                    if found_ir_roi and corners_ir_roi_raw is not None:
                        roi_arr = np.asarray(corners_ir_roi_raw)
                        if roi_arr.ndim in (2, 3) and roi_arr.size == board_cols * board_rows * 2:
                            corners_ir_offset = np.array([x0, y0], dtype=np.float64)
                            corners_ir_big = roi_arr.reshape(-1, 2).astype(np.float64) + corners_ir_offset
                            found_ir = True
            except Exception as e:
                _log(f"[WARN] ROI 检测异常: {e}")

        if not found_ir:
            found_ir_raw, corners_ir_raw = detect_board_corners(
                lwir_gray_big, board_cols, board_rows
            )
            corners_ir_big = _to_points2d(corners_ir_raw) if found_ir_raw else None
            if corners_ir_big is not None:
                found_ir = True

        # 突发采样
        if found_rgb and found_ir and corners_rgb is not None and corners_ir_big is not None:
            session.detected_count += 1
            burst_window_rgb.append(corners_rgb.copy())
            burst_window_ir.append(corners_ir_big.copy())

            if len(burst_window_rgb) == BURST_FRAMES:
                try:
                    rgb_shapes = [f.shape for f in burst_window_rgb]
                    ir_shapes = [f.shape for f in burst_window_ir]
                    if len(set(rgb_shapes)) == 1 and len(set(ir_shapes)) == 1:
                        x_rgb_avg = np.mean(np.array(burst_window_rgb), axis=0)
                        x_ir_avg_big = np.mean(np.array(burst_window_ir), axis=0)

                        # 将 IR 角点映射回原始 160x120 分辨率用于标定
                        x_ir_avg_small = x_ir_avg_big / DISPLAY_SCALE

                        best = _select_best_ir_order(
                            obj_pts, x_rgb_avg, x_ir_avg_small,
                            K_rgb, D_rgb, K_ir, D_ir,
                            session.reproj_rgb_errors,
                            session.reproj_ir_errors,
                        )
                        if best is not None:
                            total_err = session.reproj_rgb_errors[-1] + session.reproj_ir_errors[-1]
                            session.last_error = total_err

                            if total_err <= REPROJ_ERR_THRESH:
                                session.samples_rgb.append(x_rgb_avg)
                                session.samples_ir.append(x_ir_avg_small)
                                session.samples_obj.append(obj_pts)
                                n_accepted = len(session.samples_rgb)
                                print(f"[INFO] 样本已接受: {n_accepted}/{min_samples} (误差: {total_err:.2f}px)")
                                _log(f"[INFO] Sample accepted: {n_accepted}/{min_samples} (err: {total_err:.2f}px)")
                            else:
                                session.rejected_count += 1
                                session.reproj_rgb_errors.pop()
                                session.reproj_ir_errors.pop()
                                print(f"[WARN] 样本被拒绝: 误差 {total_err:.2f} > 阈值 {REPROJ_ERR_THRESH}")
                                _log(f"[WARN] Sample rejected: err {total_err:.2f} > thresh {REPROJ_ERR_THRESH}")
                        else:
                            session.rejected_count += 1
                            _log("[WARN] 样本被拒绝: PnP 计算失败")
                    else:
                        _log("[WARN] Burst 帧形状不一致，使用最新帧")
                        x_rgb_avg = burst_window_rgb[-1]
                        x_ir_avg_small = burst_window_ir[-1] / DISPLAY_SCALE
                        session.samples_rgb.append(x_rgb_avg)
                        session.samples_ir.append(x_ir_avg_small)
                        session.samples_obj.append(obj_pts)
                        n_accepted = len(session.samples_rgb)
                        print(f"[INFO] 样本已接受(单帧): {n_accepted}/{min_samples}")

                except Exception as e:
                    _log(f"[WARN] Burst 采样异常: {e}")
                    session.rejected_count += 1

                burst_window_rgb.clear()
                burst_window_ir.clear()

        # 绘制显示
        try:
            if rgb.ndim == 2:
                rgb_disp = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            else:
                rgb_disp = rgb.copy()

            if lwir.ndim == 2:
                ir_disp = cv2.cvtColor(lwir_gray_big, cv2.COLOR_GRAY2BGR)
            else:
                ir_disp = lwir.copy()
                ir_disp = cv2.cvtColor(ir_disp, cv2.COLOR_BGR2GRAY)
                ir_disp = cv2.resize(ir_disp, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_LINEAR)
                ir_disp = cv2.cvtColor(ir_disp, cv2.COLOR_GRAY2BGR)

            if session.draw_state.draw_enabled and found_rgb and corners_rgb is not None:
                result = _safe_draw_corners(rgb_disp, corners_rgb, board_cols, board_rows, True)
                if result is not None:
                    rgb_disp = result
                    session.draw_state.record_success("rgb")
                else:
                    session.draw_state.record_fail("rgb", Exception("绘制返回空"))

            if session.draw_state.draw_enabled and found_ir and corners_ir_big is not None:
                result = _safe_draw_corners(ir_disp, corners_ir_big, board_cols, board_rows, True)
                if result is not None:
                    ir_disp = result
                    session.draw_state.record_success("ir")
                else:
                    session.draw_state.record_fail("ir", Exception("绘制返回空"))

            if ROTATE_IR_DISPLAY:
                ir_disp = cv2.rotate(ir_disp, cv2.ROTATE_180)
            if INVERT_IR:
                ir_disp = cv2.bitwise_not(ir_disp)
                ir_disp = cv2.bitwise_not(ir_disp)  # 双重反转

            combined = np.hstack((rgb_disp, ir_disp))

        except Exception as e:
            _log(f"[WARN] 绘制显示失败: {e}")
            continue

        h, w = combined.shape[:2]
        btn_w, btn_h = 180, 50
        btn_x1 = max(0, w - btn_w - 20)
        btn_y1 = h - btn_h - 20
        btn_x2 = w - 20
        btn_y2 = h - 20
        n_accepted = len(session.samples_rgb)
        color_btn = (0, 255, 0) if n_accepted >= min_samples else (160, 160, 160)
        cv2.rectangle(combined, (btn_x1, btn_y1), (btn_x2, btn_y2), color_btn, -1)
        cv2.putText(combined, "OK", (btn_x1 + 60, btn_y1 + 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 2)

        status_lines = [
            f"Accepted: {n_accepted}/{min_samples}",
            f"Detected: {session.detected_count} | Rejected: {session.rejected_count}",
        ]
        if session.reproj_rgb_errors:
            avg_err = float(np.mean(session.reproj_rgb_errors[-10:]))
            status_lines.append(f"Avg RGB err (last 10): {avg_err:.2f}px")
        if session.reproj_ir_errors:
            avg_err = float(np.mean(session.reproj_ir_errors[-10:]))
            status_lines.append(f"Avg IR  err (last 10): {avg_err:.2f}px")

        for i, line in enumerate(status_lines):
            cv2.putText(combined, line, (10, 20 + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        try:
            cv2.imshow(window_name, combined)
        except Exception as e:
            _log(f"[ERROR] cv2.imshow 失败: {e}")
            continue

        ui_state.button = (btn_x1, btn_y1, btn_x2, btn_y2)
        ui_state.min_reached = n_accepted >= min_samples
        callback_dict = {"done": False, "min_reached": ui_state.min_reached, "button": ui_state.button}
        cv2.setMouseCallback(window_name, on_mouse, callback_dict)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if callback_dict.get("done", False):
            ui_state.done = True
            break

    n_accepted = len(session.samples_rgb)
    if n_accepted < min_samples:
        print(f"[ERROR] 样本不足：{n_accepted} < {min_samples}，未保存外参。")
        camera.stop()
        return 0

    if session.reproj_rgb_errors:
        mean_rgb = float(np.mean(session.reproj_rgb_errors))
        max_rgb = float(np.max(session.reproj_rgb_errors))
        print(f"[自检] RGB 重投影误差(平均/最大): {mean_rgb:.3f} / {max_rgb:.3f} px")
    if session.reproj_ir_errors:
        mean_ir = float(np.mean(session.reproj_ir_errors))
        max_ir = float(np.max(session.reproj_ir_errors))
        print(f"[自检] IR  重投影误差(平均/最大): {mean_ir:.3f} / {max_ir:.3f} px")

    Rs: List[np.ndarray] = []
    ts: List[np.ndarray] = []

    for i in range(len(session.samples_rgb)):
        x_rgb = session.samples_rgb[i]
        x_ir = session.samples_ir[i]
        Xb = session.samples_obj[i]

        best = _select_best_ir_order(
            Xb, x_rgb, x_ir,
            K_rgb, D_rgb, K_ir, D_ir,
            [],
            [],
        )
        if best is None:
            _log(f"[WARN] PnP 失败于样本 {i+1}，跳过")
            continue

        rvec_rgb, tvec_rgb, rvec_ir, tvec_ir, _ = best
        R_rgb, _ = cv2.Rodrigues(rvec_rgb)
        t_rgb = tvec_rgb.reshape(3)
        T_rgb_board = _homogeneous(R_rgb, t_rgb)

        R_ir, _ = cv2.Rodrigues(rvec_ir)
        t_ir = tvec_ir.reshape(3)
        T_ir_board = _homogeneous(R_ir, t_ir)

        T_rgb_to_ir_i = T_rgb_board @ np.linalg.inv(T_ir_board)
        Rs.append(T_rgb_to_ir_i[:3, :3])
        ts.append(T_rgb_to_ir_i[:3, 3].reshape(3))

    if len(Rs) == 0:
        print("[ERROR] 所有样本 PnP 计算失败，无法生成外参。")
        camera.stop()
        return 1

    R_avg = _average_rotations(Rs)
    t_avg = np.mean(np.array(ts), axis=0)

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = Path(__file__).resolve().parent / out_path
    _save_extrinsics_yaml(str(out_path), R_avg, t_avg, "m")

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
