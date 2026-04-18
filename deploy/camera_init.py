import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyrealsense2 as rs
import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="实时显示长波红外相机与 RealSense 深度相机")
    parser.add_argument("--serial", type=str, default="", help="可选：指定 RealSense 设备序列号")
    parser.add_argument("--lwir-index", type=int, default=2, help="长波红外相机设备编号，例如 /dev/video0 -> 0")
    parser.add_argument("--lwir-width", type=int, default=640, help="长波红外采集宽度")
    parser.add_argument("--lwir-height", type=int, default=480, help="长波红外采集高度")
    parser.add_argument("--lwir-fps", type=int, default=25, help="长波红外采集帧率")
    parser.add_argument("--lwir-display-width", type=int, default=640, help="长波红外显示宽度")
    parser.add_argument("--lwir-display-height", type=int, default=480, help="长波红外显示高度")
    parser.add_argument("--depth-width", type=int, default=640, help="RealSense Depth 采集宽度")
    parser.add_argument("--depth-height", type=int, default=480, help="RealSense Depth 采集高度")
    parser.add_argument("--depth-fps", type=int, default=30, help="RealSense Depth 采集帧率")
    parser.add_argument("--color-width", type=int, default=640, help="RealSense Color 采集宽度")
    parser.add_argument("--color-height", type=int, default=480, help="RealSense Color 采集高度")
    parser.add_argument("--color-fps", type=int, default=30, help="RealSense Color 采集帧率")
    parser.add_argument("--d435-calib", type=str, default="/home/lyric/ir_rgb/TwoStream_Yolov8/deploy/d435if_calib.yaml")
    parser.add_argument("--lepton-calib", type=str, default="/home/lyric/ir_rgb/TwoStream_Yolov8/deploy/lepton35_calib.yaml")
    parser.add_argument("--stereo-extrinsics", type=str, default="/home/lyric/ir_rgb/TwoStream_Yolov8/deploy/stereo_extrinsics_detail.yaml")
    parser.add_argument(
        "--target-unit",
        type=str,
        default="m",
        choices=["m", "mm"],
        help="统一输出外参平移单位: m 或 mm",
    )
    parser.add_argument("--print-calib-only", action="store_true", help="仅打印统一后的标定参数，不打开相机")
    return parser


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


def _infer_translation_unit(translation: np.ndarray) -> str:
    # 若平移量大于 1，通常是毫米量级；否则按米处理。
    return "mm" if float(np.max(np.abs(translation))) > 1.0 else "m"


def _convert_translation_unit(translation: np.ndarray, src_unit: str, dst_unit: str) -> np.ndarray:
    if src_unit == dst_unit:
        return translation.astype(np.float64)
    if src_unit == "mm" and dst_unit == "m":
        return translation.astype(np.float64) / 1000.0
    if src_unit == "m" and dst_unit == "mm":
        return translation.astype(np.float64) * 1000.0
    raise ValueError(f"不支持的单位转换: {src_unit} -> {dst_unit}")


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

    rgb_to_ir = stereo["extrinsic_parameters"]
    rot = np.asarray(rgb_to_ir["rotation_matrix"], dtype=np.float64)
    t_raw = rgb_to_ir.get("translation_array")
    if t_raw is None:
        t_raw = [
            rgb_to_ir["translation_vector"]["x"],
            rgb_to_ir["translation_vector"]["y"],
            rgb_to_ir["translation_vector"]["z"],
        ]
    t_raw = np.asarray(t_raw, dtype=np.float64).reshape(3)

    src_unit = rgb_to_ir.get("unit", _infer_translation_unit(t_raw)).strip().lower()
    if src_unit.startswith("millimeter"):
        src_unit = "mm"
    elif src_unit.startswith("meter"):
        src_unit = "m"
    if src_unit not in {"m", "mm"}:
        src_unit = _infer_translation_unit(t_raw)

    t_target = _convert_translation_unit(t_raw, src_unit, target_unit)

    return {
        "target_unit": target_unit,
        "rgb": {
            "K": _to_camera_matrix(d435_color),
            "D": _to_dist_coeffs(d435_color),
            "distortion_model": d435_color.get("distortion", "unknown"),
            "size": (int(d435_color["width"]), int(d435_color["height"])),
        },
        "ir": {
            "K": _to_camera_matrix(lepton_cam),
            "D": _to_dist_coeffs(lepton_cam),
            "distortion_model": lepton_cam.get("distortion", "unknown"),
            "size": (int(lepton_cam["width"]), int(lepton_cam["height"])),
        },
        # 变换方向：RGB -> IR
        "extrinsics_rgb_to_ir": {
            "R": rot,
            "t": t_target,
            "source_unit": src_unit,
        },
    }


def print_unified_calibration(calib: dict[str, Any]) -> None:
    print("[标定] 单位统一完成")
    print(f"  目标单位: {calib['target_unit']}")

    print("[RGB 内参]")
    print(calib["rgb"]["K"])
    print("[RGB 畸变]")
    print(calib["rgb"]["D"])

    print("[IR 内参]")
    print(calib["ir"]["K"])
    print("[IR 畸变]")
    print(calib["ir"]["D"])

    print("[外参 RGB -> IR] R=")
    print(calib["extrinsics_rgb_to_ir"]["R"])
    print(f"[外参 RGB -> IR] t ({calib['target_unit']})=")
    print(calib["extrinsics_rgb_to_ir"]["t"])


class MultiCameraSystem:
    """封装 PureThermal 与 RealSense 的初始化、取帧与资源释放。"""

    def __init__(self, args: argparse.Namespace, enable_depth: bool = True, enable_color: bool = False):
        self.args = args
        self.enable_depth = enable_depth
        self.enable_color = enable_color
        self.calib: dict[str, Any] | None = None

        self.lwir_cap: cv2.VideoCapture | None = None
        self.pipeline: rs.pipeline | None = None
        self.started = False

    def load_calibration(self) -> dict[str, Any]:
        for p in [self.args.d435_calib, self.args.lepton_calib, self.args.stereo_extrinsics]:
            if not Path(p).exists():
                raise FileNotFoundError(f"标定文件不存在: {p}")

        self.calib = load_and_unify_calibration(
            self.args.d435_calib,
            self.args.lepton_calib,
            self.args.stereo_extrinsics,
            target_unit=self.args.target_unit,
        )
        return self.calib

    def start(self) -> None:
        self.lwir_cap = open_lwir_camera(
            self.args.lwir_index,
            self.args.lwir_width,
            self.args.lwir_height,
            self.args.lwir_fps,
        )
        if not self.lwir_cap.isOpened():
            raise RuntimeError(f"无法打开长波红外相机: /dev/video{self.args.lwir_index}")

        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.args.serial:
            config.enable_device(self.args.serial)

        if self.enable_depth:
            config.enable_stream(
                rs.stream.depth,
                self.args.depth_width,
                self.args.depth_height,
                rs.format.z16,
                self.args.depth_fps,
            )

        if self.enable_color:
            config.enable_stream(
                rs.stream.color,
                self.args.color_width,
                self.args.color_height,
                rs.format.bgr8,
                self.args.color_fps,
            )

        if not self.enable_depth and not self.enable_color:
            raise RuntimeError("RealSense 至少需要启用 depth 或 color 其中一路")

        self.pipeline.start(config)
        self.started = True

    def read_frames(self) -> dict[str, np.ndarray] | None:
        if not self.started or self.lwir_cap is None or self.pipeline is None:
            raise RuntimeError("相机尚未启动，请先调用 start()")

        lwir_ok, lwir_frame = self.lwir_cap.read()
        if not lwir_ok or lwir_frame is None:
            return None

        frames = self.pipeline.wait_for_frames()
        out: dict[str, np.ndarray] = {}

        lwir_show = cv2.flip(to_displayable_lwir(lwir_frame), -1)
        if self.args.lwir_display_width > 0 and self.args.lwir_display_height > 0:
            if lwir_show.shape[1] != self.args.lwir_display_width or lwir_show.shape[0] != self.args.lwir_display_height:
                lwir_show = cv2.resize(
                    lwir_show,
                    (self.args.lwir_display_width, self.args.lwir_display_height),
                    interpolation=cv2.INTER_LINEAR,
                )
        out["lwir"] = lwir_show

        if self.enable_depth:
            depth_frame = frames.get_depth_frame()
            if not depth_frame:
                return None
            depth_raw = np.asanyarray(depth_frame.get_data())
            depth_norm = cv2.convertScaleAbs(depth_raw, alpha=0.03)
            out["depth_raw"] = depth_raw
            out["depth_colormap"] = cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)

        if self.enable_color:
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            out["color"] = np.asanyarray(color_frame.get_data())

        return out

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


def open_lwir_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    lwir_cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not lwir_cap.isOpened():
        return lwir_cap

    if width > 0:
        lwir_cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        lwir_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        lwir_cap.set(cv2.CAP_PROP_FPS, fps)
    return lwir_cap


def to_displayable_lwir(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if frame.dtype == np.uint16:
        return cv2.convertScaleAbs(frame, alpha=255.0 / 65535.0)

    if frame.dtype != np.uint8:
        normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)

    return frame


def main() -> int:
    args = build_parser().parse_args()
    camera = MultiCameraSystem(args, enable_depth=True, enable_color=False)

    try:
        calib = camera.load_calibration()
        print_unified_calibration(calib)
    except Exception as error:
        print(f"[错误] 标定文件加载失败: {error}")
        return 1

    if args.print_calib_only:
        return 0

    try:
        camera.start()
    except Exception as error:
        print(f"[错误] 相机启动失败: {error}")
        print("请确认设备已连接，并检查序列号、设备号和权限设置。")
        camera.stop()
        return 1

    print("[信息] 长波红外相机 + RealSense Depth 已打开。按 q 或 ESC 退出。")

    try:
        while True:
            data = camera.read_frames()
            if data is None:
                print("[警告] 帧读取失败，退出。")
                break

            cv2.imshow("LWIR", data["lwir"])
            cv2.imshow("RealSense Depth", data["depth_colormap"])

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        camera.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
