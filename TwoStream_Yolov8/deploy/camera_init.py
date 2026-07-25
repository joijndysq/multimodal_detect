import argparse
import subprocess
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


def _list_video_devices() -> list[tuple[int, str]]:
    """列出系统中的 V4L2 视频设备。优先使用 sysfs，避免 v4l2-ctl 因单个坏设备整体失败。"""
    devices: list[tuple[int, str]] = []
    sysfs_root = Path("/sys/class/video4linux")
    if sysfs_root.exists():
        for node in sorted(sysfs_root.glob("video*")):
            try:
                index = int(node.name.replace("video", ""))
            except ValueError:
                continue
            name_file = node / "name"
            name = name_file.read_text(encoding="utf-8", errors="ignore").strip() if name_file.exists() else ""
            devices.append((index, name))
        if devices:
            return devices

    try:
        out = subprocess.check_output(["v4l2-ctl", "--list-devices"], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return devices

    current_name = ""
    for raw_line in out.splitlines():
        line = raw_line.strip()
        if not line:
            current_name = ""
            continue
        if raw_line[0].isspace() and line.startswith("/dev/video"):
            try:
                devices.append((int(line.replace("/dev/video", "")), current_name))
            except ValueError:
                continue
        else:
            current_name = line.rstrip(":")
    return devices


def _find_purethermal_index() -> int | None:
    """根据设备名自动查找 PureThermal /dev/video 编号"""
    keywords = ("purethermal", "lepton", "flir")
    for index, name in _list_video_devices():
        if any(keyword in name.lower() for keyword in keywords):
            return index
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lepton 3.5 + RealSense D435i 多模态相机系统 (修复版)")
    parser.add_argument("--serial", type=str, default="", help="RealSense 设备序列号")
    parser.add_argument("--lwir-index", type=int, default=-1, help="Lepton 相机设备号 (-1=自动查找)")
    parser.add_argument("--lwir-width", type=int, default=160, help="Lepton 采集宽度")
    parser.add_argument("--lwir-height", type=int, default=120, help="Lepton 采集高度")
    parser.add_argument("--lwir-fps", type=int, default=25, help="Lepton 采集帧率")
    parser.add_argument("--lwir-display-scale", type=int, default=4, help="Lepton 显示放大倍数")
    parser.add_argument("--depth-width", type=int, default=640, help="RealSense 深度采集宽度")
    parser.add_argument("--depth-height", type=int, default=480, help="RealSense 深度采集高度")
    parser.add_argument("--depth-fps", type=int, default=30, help="RealSense 深度采集帧率")
    parser.add_argument("--color-width", type=int, default=640, help="RealSense 彩色采集宽度")
    parser.add_argument("--color-height", type=int, default=480, help="RealSense 彩色采集高度")
    parser.add_argument("--color-fps", type=int, default=30, help="RealSense 彩色采集帧率")
    parser.add_argument("--rs-timeout-ms", type=int, default=5000, help="RealSense 取帧超时(毫秒)")
    parser.add_argument("--d435-calib", type=str, default="d435if_calib.yaml", help="D435i 标定文件")
    parser.add_argument("--lepton-calib", type=str, default="lepton35_calib(1).yaml", help="Lepton 标定文件")
    parser.add_argument("--stereo-extrinsics", type=str, default="calib_extrinsics(2).yaml", help="外参标定文件")
    parser.add_argument("--target-unit", type=str, default="m", choices=["m", "mm"], help="外参平移单位")
    parser.add_argument("--print-calib-only", action="store_true", help="仅打印标定参数")
    parser.add_argument("--align-depth-to-lwir", action="store_true",default=True, help="将深度图对齐到红外坐标系")
    parser.add_argument("--align-skip-frames", type=int, default=10, help="深度对齐跳帧数(越大越流畅)")
    parser.add_argument("--white-hot", action=argparse.BooleanOptionalAction, default=False, help="红外白热模式(热目标=白色)")
    parser.add_argument("--align-rgb-to-lwir", action=argparse.BooleanOptionalAction, default=True, help="将RGB对齐到红外坐标系")
    parser.add_argument("--lwir-filter", type=str, default="median", choices=["none", "gaussian", "median", "bilateral"], help="红外滤波方式")
    parser.add_argument("--show-overlay", action="store_true", default=False, help="叠加IR和RGB检查对齐")
    parser.add_argument("--enable-timestamp-sync", action=argparse.BooleanOptionalAction, default=True, help="按接收时间戳软同步 LWIR 和 RealSense")
    parser.add_argument("--sync-max-delta-ms", type=float, default=40.0, help="允许配对的最大时间差(毫秒)")
    parser.add_argument("--sync-buffer-size", type=int, default=8, help="每路缓冲的最大帧数")
    return parser


@dataclass
class LwirFramePacket:
    recv_ts_ns: int
    frame: np.ndarray


@dataclass
class RsFramePacket:
    recv_ts_ns: int
    depth_frame_obj: rs.depth_frame | None
    depth_intrin: rs.intrinsics | None
    depth_raw: np.ndarray | None
    color: np.ndarray | None


@dataclass
class SyncedFrameBundle:
    lwir_raw: np.ndarray | None
    depth_raw: np.ndarray | None
    color: np.ndarray | None
    lwir_ts_ns: int | None = None
    rs_ts_ns: int | None = None
    depth_frame_obj: rs.depth_frame | None = None
    depth_intrin: rs.intrinsics | None = None


def _load_yaml(path: str) -> dict[str, Any]:
    resolved = _resolve_input_path(path)
    with resolved.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_input_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    search_paths = [candidate]
    if not candidate.is_absolute():
        search_paths.append(SCRIPT_DIR / candidate)

    for item in search_paths:
        resolved = item.resolve()
        if resolved.exists():
            return resolved

    searched = ", ".join(str(item) for item in search_paths)
    raise FileNotFoundError(f"标定文件不存在: {path} (已检查: {searched})")


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


def _infer_translation_unit(translation: np.ndarray) -> str:
    return "mm" if float(np.max(np.abs(translation))) > 1.0 else "m"


def load_and_unify_calibration(
    d435_yaml: str,
    lepton_yaml: str,
    stereo_yaml: str,
    target_unit: str = "m",
) -> dict[str, Any]:
    d435 = _load_yaml(d435_yaml)
    lepton = _load_yaml(lepton_yaml)
    stereo = _load_yaml(stereo_yaml)

    d435_depth = d435["depth_camera"]
    d435_color = d435["color_camera"]
    lepton_cam = lepton["thermal_camera"]

    # 兼容两种外参文件格式：
    #   新格式: 顶层 extrinsics_rgb_to_ir {R, t, source_unit}
    #   旧格式: 顶层 extrinsic_parameters {rotation_matrix, translation_vector/array, unit}
    if "extrinsics_rgb_to_ir" in stereo:
        extrinsics = stereo["extrinsics_rgb_to_ir"]
        rot = np.asarray(extrinsics["R"], dtype=np.float64)
        t_raw = np.asarray(extrinsics["t"], dtype=np.float64).reshape(3)
        src_unit = extrinsics.get("source_unit", "m").strip().lower()
        if src_unit.startswith("millimeter"):
            src_unit = "mm"
        elif src_unit.startswith("meter"):
            src_unit = "m"
    elif "extrinsic_parameters" in stereo:
        extrinsics = stereo["extrinsic_parameters"]
        rot = np.asarray(extrinsics["rotation_matrix"], dtype=np.float64)
        t_raw = extrinsics.get("translation_array")
        if t_raw is None:
            t_raw = [
                extrinsics["translation_vector"]["x"],
                extrinsics["translation_vector"]["y"],
                extrinsics["translation_vector"]["z"],
            ]
        t_raw = np.asarray(t_raw, dtype=np.float64).reshape(3)
        raw_unit = extrinsics.get("unit", "").strip().lower()
        if raw_unit.startswith("millimeter"):
            src_unit = "mm"
        elif raw_unit.startswith("meter"):
            src_unit = "m"
        else:
            src_unit = _infer_translation_unit(t_raw)
    else:
        raise KeyError("外参文件缺少 extrinsics_rgb_to_ir 或 extrinsic_parameters 字段")

    if src_unit == "mm" and target_unit == "m":
        t_target = t_raw.astype(np.float64) / 1000.0
    elif src_unit == "m" and target_unit == "mm":
        t_target = t_raw.astype(np.float64) * 1000.0
    else:
        t_target = t_raw.astype(np.float64)

    return {
        "target_unit": target_unit,
        "depth": {
            "K": _to_camera_matrix(d435_depth),
            "D": _to_dist_coeffs(d435_depth),
            "size": (int(d435_depth["width"]), int(d435_depth["height"])),
        },
        "rgb": {
            "K": _to_camera_matrix(d435_color),
            "D": _to_dist_coeffs(d435_color),
            "size": (int(d435_color["width"]), int(d435_color["height"])),
        },
        "ir": {
            "K": _to_camera_matrix(lepton_cam),
            "D": _to_dist_coeffs(lepton_cam),
            "size": (int(lepton_cam["width"]), int(lepton_cam["height"])),
        },
        "extrinsics_rgb_to_ir": {"R": rot, "t": t_target, "source_unit": src_unit},
        "extrinsics_depth_to_rgb": {
            "R": np.asarray(d435["extrinsics"]["depth_to_color"]["rotation"], dtype=np.float64),
            "t": np.asarray(d435["extrinsics"]["depth_to_color"]["translation"], dtype=np.float64).reshape(3),
            "unit": "m",
        },
    }


def print_unified_calibration(calib: dict[str, Any]) -> None:
    print("=" * 60)
    print("[标定参数加载成功]")
    print(f"  目标单位: {calib['target_unit']}")
    print("-" * 60)
    print("[RGB 内参矩阵]")
    print(np.round(calib["rgb"]["K"], 4))
    print("[RGB 畸变系数]")
    print(np.round(calib["rgb"]["D"], 6))
    print("-" * 60)
    print("[IR 内参矩阵]")
    print(np.round(calib["ir"]["K"], 4))
    print("[IR 畸变系数]")
    print(np.round(calib["ir"]["D"], 6))
    print("-" * 60)
    print("[外参 RGB -> IR 旋转矩阵]")
    print(np.round(calib["extrinsics_rgb_to_ir"]["R"], 6))
    print(f"[外参 RGB -> IR 平移向量 ({calib['target_unit']})]")
    print(np.round(calib["extrinsics_rgb_to_ir"]["t"], 6))
    print("=" * 60)


class MultiCameraSystem:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.calib: dict[str, Any] | None = None

        self.lwir_cap: cv2.VideoCapture | None = None
        self.pipeline: rs.pipeline | None = None
        self.align_to_color: rs.align | None = None
        self.depth_scale: float = 0.001
        self.depth_intrin: rs.intrinsics | None = None

        self._frame_count = 0
        self._cached_depth_aligned: np.ndarray | None = None
        self._cached_rgb_aligned: np.ndarray | None = None
        self._lwir_buffer: deque[LwirFramePacket] = deque(maxlen=max(1, args.sync_buffer_size))
        self._rs_buffer: deque[RsFramePacket] = deque(maxlen=max(1, args.sync_buffer_size))

    def _align_depth_to_lwir_fast(self, depth_raw: np.ndarray) -> np.ndarray:
        """将 RealSense 深度图对齐到 Lepton IR 坐标系（修复版）。

        核心修复：
        1. 使用真正的外参 R/t（从YAML加载）做坐标变换
        2. 使用正确的旋转向量（从R矩阵转换）做投影
        3. 正确处理不同分辨率的坐标映射
        """
        if self.calib is None:
            raise RuntimeError("标定参数未加载")

        ir_w, ir_h = self.calib["ir"]["size"]
        depth_h, depth_w = depth_raw.shape

        # 1. 转换深度值为米（子采样加速：每2个像素取1个）
        stride = 2
        z_m = depth_raw.astype(np.float32)[::stride, ::stride] * self.depth_scale
        valid = z_m > 0.1

        if not np.any(valid):
            return np.zeros((ir_h, ir_w), dtype=np.float32)

        # 2. 提取有效点的像素坐标（子采样后坐标系需还原）
        v, u = np.where(valid)
        u = (u * stride).astype(np.int32)
        v = (v * stride).astype(np.int32)
        z_valid = z_m[valid]
        N = len(u)

        # 3. 构建像素坐标并去畸变（深度相机内参）
        pts_pixel = np.stack([u, v], axis=1).reshape(-1, 1, 2).astype(np.float32)
        K_depth = self.calib["depth"]["K"].astype(np.float32)
        D_depth = self.calib["depth"]["D"].astype(np.float32)

        pts_undist = cv2.undistortPoints(pts_pixel, K_depth, D_depth)
        if pts_undist is None:
            return np.zeros((ir_h, ir_w), dtype=np.float32)
        pts_undist = pts_undist.reshape(-1, 2)

        # 4. 反投影到3D（深度相机坐标系）
        x = pts_undist[:, 0] * z_valid
        y = pts_undist[:, 1] * z_valid
        pts_3d_depth = np.stack([x, y, z_valid], axis=1)

        # 5. 坐标变换（Depth -> RGB -> IR）
        # 5.1 Depth 到 RGB
        R_d2r = self.calib["extrinsics_depth_to_rgb"]["R"].astype(np.float64)
        t_d2r = self.calib["extrinsics_depth_to_rgb"]["t"].astype(np.float64)
        pts_3d_rgb = (pts_3d_depth @ R_d2r.T) + t_d2r

        # 5.2 RGB 到 IR（使用真正的外参！）
        R_r2i = self.calib["extrinsics_rgb_to_ir"]["R"].astype(np.float64)
        t_r2i = self.calib["extrinsics_rgb_to_ir"]["t"].astype(np.float64)
        pts_3d_ir = (pts_3d_rgb @ R_r2i.T) + t_r2i

        # 6. 过滤有效点（Z > 0）
        positive_z = pts_3d_ir[:, 2] > 0.1
        if not np.any(positive_z):
            return np.zeros((ir_h, ir_w), dtype=np.float32)

        pts_3d_ir = pts_3d_ir[positive_z]
        z_ir = pts_3d_ir[:, 2]

        # 7. 投影到IR图像平面（使用IR内参 + 真正的外参R/t）
        K_ir = self.calib["ir"]["K"].astype(np.float64)
        D_ir = self.calib["ir"]["D"].astype(np.float64)

        # 关键修复：使用正确的旋转向量（从R矩阵转换）
        rvec_ir, _ = cv2.Rodrigues(R_r2i.astype(np.float64))
        tvec_ir = t_r2i.astype(np.float64)

        img_pts, _ = cv2.projectPoints(
            pts_3d_ir.reshape(-1, 1, 3).astype(np.float64),
            rvec_ir,
            tvec_ir,
            K_ir,
            D_ir,
        )
        img_pts = img_pts.reshape(-1, 2)

        u_ir = np.round(img_pts[:, 0]).astype(np.int32)
        v_ir = np.round(img_pts[:, 1]).astype(np.int32)

        # 8. 边界检查（IR图像范围 160×120）
        in_bounds = (u_ir >= 0) & (u_ir < ir_w) & (v_ir >= 0) & (v_ir < ir_h)
        if not np.any(in_bounds):
            return np.zeros((ir_h, ir_w), dtype=np.float32)

        u_ir = u_ir[in_bounds]
        v_ir = v_ir[in_bounds]
        z_ir_final = z_ir[in_bounds]

        # 9. Z-buffer（最近深度优先）
        depth_ir = np.full((ir_h, ir_w), np.inf, dtype=np.float32)
        indices = v_ir * ir_w + u_ir

        # 使用 np.minimum.at 进行原子操作
        np.minimum.at(depth_ir.ravel(), indices, z_ir_final)

        # 10. 处理inf
        depth_ir[np.isinf(depth_ir)] = 0.0

        # 11. 中值滤波去噪
        depth_ir = cv2.medianBlur(depth_ir, 3)
        return depth_ir

    def _align_rgb_to_lwir(self, rgb: np.ndarray, depth_aligned_m: np.ndarray) -> np.ndarray:
        """将 RGB 图像对齐到 IR 坐标系，输出 640x480 RGB（与 LWIR 逐像素对齐）"""
        if self.calib is None:
            raise RuntimeError("标定参数未加载")

        ir_w, ir_h = self.calib["ir"]["size"]  # 原生 160x120
        rgb_h, rgb_w = rgb.shape[:2]

        # 缩放到显示分辨率 (640x480)
        scale_w = 640 / ir_w
        scale_h = 480 / ir_h
        K_ir = self.calib["ir"]["K"].astype(np.float64).copy()
        K_ir[0, 0] *= scale_w
        K_ir[1, 1] *= scale_h
        K_ir[0, 2] *= scale_w
        K_ir[1, 2] *= scale_h

        depth_resized = cv2.resize(depth_aligned_m, (640, 480), interpolation=cv2.INTER_NEAREST)
        rgb_out = np.zeros((480, 640, 3), dtype=np.uint8)

        # 1. 有效深度点 (640x480, 子采样加速)
        stride = 2
        depth_sub = depth_resized[::stride, ::stride]
        valid = depth_sub > 0.1
        if not np.any(valid):
            return rgb_out

        v_ir, u_ir = np.where(valid)
        u_ir = (u_ir * stride).astype(np.int32)
        v_ir = (v_ir * stride).astype(np.int32)
        z = depth_sub[valid].astype(np.float64)

        # 2. 在 IR 坐标系反投影到 3D (使用缩放后内参)
        fx_ir, fy_ir = K_ir[0, 0], K_ir[1, 1]
        cx_ir, cy_ir = K_ir[0, 2], K_ir[1, 2]
        x_ir = (u_ir - cx_ir) * z / fx_ir
        y_ir = (v_ir - cy_ir) * z / fy_ir
        pts_ir = np.stack([x_ir, y_ir, z], axis=1)

        # 3. IR → RGB 坐标变换（外参的逆）
        R_r2i = self.calib["extrinsics_rgb_to_ir"]["R"].astype(np.float64)
        t_r2i = self.calib["extrinsics_rgb_to_ir"]["t"].astype(np.float64)
        R_i2r = R_r2i.T
        t_i2r = -R_r2i.T @ t_r2i
        pts_rgb = (pts_ir @ R_i2r.T) + t_i2r

        z_rgb = pts_rgb[:, 2]
        positive = z_rgb > 0.1
        if not np.any(positive):
            return rgb_out

        pts_rgb = pts_rgb[positive]
        z_rgb = z_rgb[positive]
        u_ir_f = u_ir[positive]
        v_ir_f = v_ir[positive]

        # 4. 投影到 RGB 图像平面
        K_rgb = self.calib["rgb"]["K"].astype(np.float64)
        u_rgb = (pts_rgb[:, 0] * K_rgb[0, 0] / z_rgb + K_rgb[0, 2])
        v_rgb = (pts_rgb[:, 1] * K_rgb[1, 1] / z_rgb + K_rgb[1, 2])

        u_rgb_i = np.round(u_rgb).astype(np.int32)
        v_rgb_i = np.round(v_rgb).astype(np.int32)

        in_rgb = (u_rgb_i >= 0) & (u_rgb_i < rgb_w) & (v_rgb_i >= 0) & (v_rgb_i < rgb_h)
        if not np.any(in_rgb):
            return rgb_out

        u_rgb_i = u_rgb_i[in_rgb]
        v_rgb_i = v_rgb_i[in_rgb]
        u_ir_f = u_ir_f[in_rgb]
        v_ir_f = v_ir_f[in_rgb]

        # 5. 采样 RGB 颜色
        rgb_out[v_ir_f, u_ir_f] = rgb[v_rgb_i, u_rgb_i]
        return rgb_out

    def load_calibration(self) -> dict[str, Any]:
        d435_path = _resolve_input_path(self.args.d435_calib)
        lepton_path = _resolve_input_path(self.args.lepton_calib)
        stereo_path = _resolve_input_path(self.args.stereo_extrinsics)

        self.calib = load_and_unify_calibration(
            str(d435_path),
            str(lepton_path),
            str(stereo_path),
            target_unit=self.args.target_unit,
        )
        return self.calib

    def start(self) -> None:
        print("[信息] 正在初始化 RealSense D435i...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.args.serial:
            config.enable_device(self.args.serial)

        config.enable_stream(rs.stream.depth, self.args.depth_width, self.args.depth_height, rs.format.z16, self.args.depth_fps)
        config.enable_stream(rs.stream.color, self.args.color_width, self.args.color_height, rs.format.bgr8, self.args.color_fps)

        profile = self.pipeline.start(config)
        self.align_to_color = rs.align(rs.stream.color)
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = float(depth_sensor.get_depth_scale())
        depth_profile = profile.get_stream(rs.stream.depth)
        self.depth_intrin = depth_profile.as_video_stream_profile().get_intrinsics()
        print(f"[信息] RealSense 初始化成功，深度刻度: {self.depth_scale:.8f} m/unit")

        print("[信息] 正在初始化 Lepton 3.5 红外相机...")
        lwir_idx = self.args.lwir_index
        if lwir_idx < 0:
            auto_idx = _find_purethermal_index()
            if auto_idx is None:
                devices = _list_video_devices()
                details = ", ".join(f"/dev/video{idx}={name or 'unknown'}" for idx, name in devices) or "无可见 V4L2 设备"
                raise RuntimeError(f"未找到 PureThermal 设备，请用 --lwir-index 手动指定。当前视频设备: {details}")
            lwir_idx = auto_idx
            print(f"[信息] 自动检测到 PureThermal: /dev/video{lwir_idx}")

        self.lwir_cap = cv2.VideoCapture(lwir_idx, cv2.CAP_V4L2)
        if not self.lwir_cap.isOpened():
            print("[警告] V4L2后端失败，尝试默认后端...")
            self.lwir_cap = cv2.VideoCapture(lwir_idx)

        if not self.lwir_cap.isOpened():
            raise RuntimeError(f"无法打开Lepton相机: video{lwir_idx}")

        self.lwir_cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        self.lwir_cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"Y16 "))
        self.lwir_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.lwir_width)
        self.lwir_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.lwir_height)
        self.lwir_cap.set(cv2.CAP_PROP_FPS, self.args.lwir_fps)

        for _ in range(5):
            self.lwir_cap.read()

        print(f"[信息] Lepton 初始化成功，分辨率: {self.args.lwir_width}x{self.args.lwir_height} @ {self.args.lwir_fps}fps")
        print(f"[性能] 深度对齐每 {self.args.align_skip_frames} 帧计算一次")

    def read_frames(self) -> dict[str, np.ndarray]:
        if self.calib is None:
            raise RuntimeError("标定参数未加载")

        self._frame_count += 1

        lwir_packet = self._read_lwir_packet()
        if lwir_packet is not None:
            self._lwir_buffer.append(lwir_packet)

        rs_packet = self._read_rs_packet()
        if rs_packet is not None:
            self._rs_buffer.append(rs_packet)

        if self.args.enable_timestamp_sync:
            pair = self._pop_synced_pair()
            if pair is None:
                return {}
            lwir_packet, rs_packet = pair
        else:
            lwir_packet = self._lwir_buffer.pop() if self._lwir_buffer else None
            rs_packet = self._rs_buffer.pop() if self._rs_buffer else None
            self._lwir_buffer.clear()
            self._rs_buffer.clear()

        return self._build_output(lwir_packet, rs_packet)

    def _read_lwir_packet(self) -> LwirFramePacket | None:
        if self.lwir_cap is None:
            return None
        lwir_ok, lwir_raw = self.lwir_cap.read()
        if not lwir_ok or lwir_raw is None:
            return None
        recv_ts_ns = time.monotonic_ns()
        return LwirFramePacket(recv_ts_ns=recv_ts_ns, frame=lwir_raw)

    def _read_rs_packet(self) -> RsFramePacket | None:
        if self.pipeline is None or self.align_to_color is None:
            return None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=self.args.rs_timeout_ms)
            recv_ts_ns = time.monotonic_ns()
            frames = self.align_to_color.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            depth_raw = np.asarray(depth_frame.get_data()) if depth_frame else None
            color = np.asarray(color_frame.get_data()) if color_frame else None
            return RsFramePacket(
                recv_ts_ns=recv_ts_ns,
                depth_frame_obj=depth_frame,
                depth_intrin=self.depth_intrin,
                depth_raw=depth_raw,
                color=color,
            )
        except RuntimeError:
            return None

    def _pop_synced_pair(self) -> tuple[LwirFramePacket | None, RsFramePacket | None] | None:
        if not self._lwir_buffer or not self._rs_buffer:
            return None

        max_delta_ns = int(self.args.sync_max_delta_ms * 1e6)
        best_pair: tuple[int, int] | None = None
        best_delta_ns: int | None = None

        for lwir_idx, lwir_packet in enumerate(self._lwir_buffer):
            for rs_idx, rs_packet in enumerate(self._rs_buffer):
                delta_ns = abs(lwir_packet.recv_ts_ns - rs_packet.recv_ts_ns)
                if delta_ns > max_delta_ns:
                    continue
                if best_delta_ns is None or delta_ns < best_delta_ns:
                    best_delta_ns = delta_ns
                    best_pair = (lwir_idx, rs_idx)

        if best_pair is None:
            self._drop_stale_packets(max_delta_ns)
            return None

        lwir_idx, rs_idx = best_pair
        lwir_packet = self._remove_from_deque(self._lwir_buffer, lwir_idx)
        rs_packet = self._remove_from_deque(self._rs_buffer, rs_idx)
        return lwir_packet, rs_packet

    @staticmethod
    def _remove_from_deque(buffer: deque, index: int):
        items = list(buffer)
        selected = items.pop(index)
        buffer.clear()
        buffer.extend(items)
        return selected

    def _drop_stale_packets(self, max_delta_ns: int) -> None:
        if not self._lwir_buffer or not self._rs_buffer:
            return
        newest_lwir_ts = self._lwir_buffer[-1].recv_ts_ns
        newest_rs_ts = self._rs_buffer[-1].recv_ts_ns
        while self._lwir_buffer and self._lwir_buffer[0].recv_ts_ns < newest_rs_ts - max_delta_ns:
            self._lwir_buffer.popleft()
        while self._rs_buffer and self._rs_buffer[0].recv_ts_ns < newest_lwir_ts - max_delta_ns:
            self._rs_buffer.popleft()

    def _build_output(
        self,
        lwir_packet: LwirFramePacket | None,
        rs_packet: RsFramePacket | None,
    ) -> dict[str, np.ndarray]:
        out: dict[str, Any] = {}

        if lwir_packet is not None:
            lwir_raw = lwir_packet.frame
            if lwir_raw.dtype == np.uint16:
                lwir_raw = lwir_raw.squeeze()
                if lwir_raw.shape[0] > self.args.lwir_height:
                    lwir_raw = lwir_raw[:self.args.lwir_height, :]
                min_val = np.percentile(lwir_raw, 2)
                max_val = np.percentile(lwir_raw, 98)
                scale = max(max_val - min_val, 1.0)
                lwir_norm = np.clip((lwir_raw - min_val) / scale * 255, 0, 255).astype(np.uint8)
                if not self.args.white_hot:
                    lwir_norm = 255 - lwir_norm
            else:
                lwir_norm = cv2.cvtColor(lwir_raw, cv2.COLOR_BGR2GRAY) if lwir_raw.ndim == 3 else lwir_raw

            lwir_norm = cv2.flip(lwir_norm, -1)
            ft = self.args.lwir_filter
            if ft == "gaussian":
                lwir_norm = cv2.GaussianBlur(lwir_norm, (3, 3), 0)
            elif ft == "median":
                lwir_norm = cv2.medianBlur(lwir_norm, 3)
            elif ft == "bilateral":
                lwir_norm = cv2.bilateralFilter(lwir_norm, 5, 50, 50)
            out["lwir"] = cv2.resize(lwir_norm, (640, 480), interpolation=cv2.INTER_LINEAR)
            out["lwir_ts_ns"] = lwir_packet.recv_ts_ns

        if rs_packet is not None:
            out["rs_ts_ns"] = rs_packet.recv_ts_ns
            if rs_packet.depth_frame_obj is not None:
                out["depth_frame_obj"] = rs_packet.depth_frame_obj
                out["depth_intrin"] = rs_packet.depth_intrin
            elif rs_packet.depth_intrin is not None:
                out["depth_intrin"] = rs_packet.depth_intrin
            out["depth_scale"] = self.depth_scale
            if rs_packet.depth_raw is not None:
                depth_raw = rs_packet.depth_raw
                out["depth_raw"] = depth_raw
                out["depth_colormap"] = cv2.applyColorMap(cv2.convertScaleAbs(depth_raw, alpha=0.03), cv2.COLORMAP_JET)

                if self.args.align_depth_to_lwir and self.calib is not None:
                    if self._frame_count % self.args.align_skip_frames == 0 or self._cached_depth_aligned is None:
                        self._cached_depth_aligned = self._align_depth_to_lwir_fast(depth_raw)

                    out["depth_aligned_m"] = self._cached_depth_aligned
                    out["depth_aligned_colormap"] = cv2.applyColorMap(
                        cv2.convertScaleAbs(self._cached_depth_aligned, alpha=25.5), cv2.COLORMAP_JET
                    )

            if rs_packet.color is not None:
                rgb = rs_packet.color
                out["color_uncropped"] = rgb
                out["color"] = rgb

                if self.args.align_rgb_to_lwir and self.calib is not None and self._cached_depth_aligned is not None:
                    try:
                        if self._frame_count % self.args.align_skip_frames == 0 or self._cached_rgb_aligned is None:
                            self._cached_rgb_aligned = self._align_rgb_to_lwir(rgb, self._cached_depth_aligned)
                        out["rgb_aligned"] = self._cached_rgb_aligned
                    except Exception as e:
                        print(f"[警告] RGB对齐失败: {e}")

        if lwir_packet is not None and rs_packet is not None:
            sync_delta_ms = abs(lwir_packet.recv_ts_ns - rs_packet.recv_ts_ns) / 1e6
            out["sync_delta_ms"] = sync_delta_ms

        return out

    def process_synced_bundle(self, bundle: SyncedFrameBundle) -> dict[str, Any]:
        """处理外部同步好的帧。

        适用于:
        1. Python 侧已有同步器
        2. C/C++ 侧完成采集+同步，再把原始帧交给当前类做标定和几何对齐
        """
        lwir_packet = None
        rs_packet = None

        if bundle.lwir_raw is not None:
            lwir_packet = LwirFramePacket(
                recv_ts_ns=bundle.lwir_ts_ns if bundle.lwir_ts_ns is not None else time.monotonic_ns(),
                frame=bundle.lwir_raw,
            )

        if bundle.depth_raw is not None or bundle.color is not None:
            rs_packet = RsFramePacket(
                recv_ts_ns=bundle.rs_ts_ns if bundle.rs_ts_ns is not None else time.monotonic_ns(),
                depth_frame_obj=bundle.depth_frame_obj,
                depth_intrin=bundle.depth_intrin if bundle.depth_intrin is not None else self.depth_intrin,
                depth_raw=bundle.depth_raw,
                color=bundle.color,
            )

        self._frame_count += 1
        return self._build_output(lwir_packet, rs_packet)

    def stop(self) -> None:
        print("[信息] 正在释放相机资源...")
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
        print("[信息] 相机资源已释放")


def main() -> int:
    args = build_parser().parse_args()
    camera = MultiCameraSystem(args)

    try:
        calib = camera.load_calibration()
        print_unified_calibration(calib)
    except Exception as e:
        print(f"[错误] 标定文件加载失败: {e}")
        return 1

    if args.print_calib_only:
        return 0

    try:
        camera.start()
    except Exception as e:
        print(f"[错误] 相机启动失败: {e}")
        camera.stop()
        return 1

    print("\n[信息] 所有相机已启动。按 q 或 ESC 退出。")
    print(f"[信息] LWIR 显示: {'白热' if args.white_hot else '黑热'}模式")
    print(f"[信息] 深度对齐: {'启用' if args.align_depth_to_lwir else '禁用'}")
    print(f"[信息] RGB->IR对齐: {'启用' if args.align_rgb_to_lwir else '禁用'}")

    import time as _time
    fps_timer = _time.time()
    fps_count = 0
    _fps = 0.0

    try:
        while True:
            data = camera.read_frames()
            if not data:
                continue

            fps_count += 1
            now = _time.time()
            elapsed = now - fps_timer
            if elapsed >= 1.0:
                _fps = fps_count / elapsed
                fps_count = 0
                fps_timer = now

            if "lwir" in data:
                lwir_show = data["lwir"].copy()
                cv2.putText(lwir_show, f"FPS: {_fps:.1f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("LWIR", lwir_show)
            if "color" in data:
                cv2.imshow("RealSense RGB", data["color"])
            if args.show_overlay and "lwir" in data and "color" in data:
                lwir_bgr = cv2.cvtColor(data["lwir"], cv2.COLOR_GRAY2BGR)
                overlay = cv2.addWeighted(data["color"], 0.5, lwir_bgr, 0.5, 0)
                cv2.imshow("Overlay IR+RGB [50%]", overlay)
            if "rgb_aligned" in data:
                cv2.imshow("Aligned RGB to IR", data["rgb_aligned"])
            if "depth_aligned_colormap" in data:
                cv2.imshow("Aligned Depth to IR", data["depth_aligned_colormap"])

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        camera.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
