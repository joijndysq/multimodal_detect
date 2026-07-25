import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO


def build_parser():
    parser = argparse.ArgumentParser(description="TwoStream YOLO 实时检测 - 红外与可见光融合")
    parser.add_argument("--weights", type=str, default="/home/lyric/ir_rgb/TwoStream_Yolov8/pth/baseline.pt", 
                        help="模型权重路径")
    parser.add_argument("--serial", type=str, default="", help="RealSense 设备序列号")
    parser.add_argument("--lwir-index", type=int, default=2, help="红外相机设备编号")
    parser.add_argument("--lwir-width", type=int, default=640)
    parser.add_argument("--lwir-height", type=int, default=480)
    parser.add_argument("--lwir-fps", type=int, default=25)
    parser.add_argument("--rgb-width", type=int, default=640)
    parser.add_argument("--rgb-height", type=int, default=480)
    parser.add_argument("--rgb-fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640, help="推理图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--obb", action="store_true", help="使用旋转框检测")
    return parser


def open_lwir_camera(index, width, height, fps):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def to_displayable_lwir(frame):
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame.dtype == np.uint16:
        return cv2.convertScaleAbs(frame, alpha=255.0 / 65535.0)
    if frame.dtype != np.uint8:
        normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
        return normalized.astype(np.uint8)
    return frame


def load_model(weights_path, device):
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, using CPU")
        device = "cpu"
    model = YOLO(weights_path).to(device)
    model.fuse()
    return model


def preprocess_fusion(rgb_img, ir_img, target_size=640):
    """将RGB和红外图像在通道维度上融合"""
    rgb_resized = cv2.resize(rgb_img, (target_size, target_size))
    ir_resized = cv2.resize(ir_img, (target_size, target_size))
    
    if ir_resized.ndim == 2:
        ir_resized = cv2.cvtColor(ir_resized, cv2.COLOR_GRAY2BGR)
    
    fused = np.concatenate((rgb_resized, ir_resized), axis=2)
    return fused, rgb_resized, ir_resized


def draw_boxes(img, results, names, colors, obb=False):
    if obb:
        confs = results[0].obb.conf
        classes = results[0].obb.cls
        xywhr = results[0].obb.xywhr
    else:
        confs = results[0].boxes.conf
        classes = results[0].boxes.cls
        xywhr = results[0].boxes.xywh

    confs = confs.cpu().numpy()
    classes = classes.cpu().numpy()
    xywhr = xywhr.cpu().numpy()

    for i, (cls, conf, box) in enumerate(zip(classes, confs, xywhr)):
        cls = int(cls)
        color = colors[cls % len(colors)]
        
        if obb:
            cx, cy, w, h, angle = box
            angle_rad = np.deg2rad(angle)
            cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
            R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            corners = np.array([[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]])
            rotated = corners @ R.T + [cx, cy]
            pts = rotated.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(img, [pts], True, color, 2)
            label = f"{names[cls]} {conf:.2f}"
        else:
            x, y, w, h = box
            x1, y1 = int(x - w/2), int(y - h/2)
            x2, y2 = int(x + w/2), int(y + h/2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{names[cls]} {conf:.2f}"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return img


def main():
    args = build_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(args.weights, device)

    lwir_cap = open_lwir_camera(args.lwir_index, args.lwir_width, args.lwir_height, args.lwir_fps)
    if lwir_cap is None or not lwir_cap.isOpened():
        print(f"Error: Cannot open LWIR camera /dev/video{args.lwir_index}")
        return 1

    pipeline = rs.pipeline()
    config = rs.config()
    if args.serial:
        config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.rgb_width, args.rgb_height, rs.format.bgr8, args.rgb_fps)

    try:
        pipeline.start(config)
    except Exception as e:
        print(f"Error starting RealSense: {e}")
        lwir_cap.release()
        return 1

    class_names = ["car", "truck", "bus", "van", "freight"]
    colors = [(165, 0, 255), (0, 255, 0), (102, 255, 255), (255, 165, 0), (255, 255, 0)]

    print("Press 'q' or ESC to exit")

    try:
        while True:
            lwir_ok, lwir_frame = lwir_cap.read()
            if not lwir_ok:
                break

            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            rgb_frame = np.asanyarray(color_frame.get_data())
            lwir_frame = to_displayable_lwir(lwir_frame)
            lwir_frame = cv2.flip(lwir_frame, -1)

            fused_img, rgb_resized, ir_resized = preprocess_fusion(rgb_frame, lwir_frame, args.imgsz)

            results = model.predict(fused_img, conf=args.conf, imgsz=args.imgsz, verbose=False, obb=args.obb)

            rgb_display = draw_boxes(rgb_resized.copy(), results, class_names, colors, args.obb)
            ir_display = draw_boxes(ir_resized.copy(), results, class_names, colors, args.obb)

            cv2.imshow("RGB Detection", rgb_display)
            cv2.imshow("IR Detection", ir_display)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
    finally:
        pipeline.stop()
        lwir_cap.release()
        cv2.destroyAllWindows()
    
    return 0


if __name__ == "__main__":
    exit(main())