# 将 .pt 模型导出为 .onnx 格式（支持双流 6 通道输入）
import argparse
import os
import torch
import onnx
from ultralytics import YOLO
from ultralytics.nn.tasks import DetectionModel

# 允许加载自定义模型结构
torch.serialization.add_safe_globals([DetectionModel])


def export_onnx(
    pt_path: str,
    output_path: str = None,
    img_size: int = 640,
    batch_size: int = 1,
    opset: int = 17,
    half: bool = False,
    dynamic: bool = False,
    simplify: bool = True,
    in_channels: int = 6,
):
    """
    将 .pt 模型导出为 .onnx 文件。

    Args:
        pt_path:     输入的 .pt 文件路径
        output_path: 输出的 .onnx 文件路径（默认与 pt 同目录同名）
        img_size:    输入图像尺寸
        batch_size:  batch 大小（dynamic=True 时忽略）
        opset:       ONNX opset 版本
        half:        是否导出 FP16
        dynamic:     是否开启动态 batch/shape
        simplify:    是否使用 onnx-simplifier 简化模型
        in_channels: 输入通道数（单流=3，双流 RGB+IR=6）
    """
    assert os.path.exists(pt_path), f"文件不存在: {pt_path}"

    if output_path is None:
        output_path = os.path.splitext(pt_path)[0] + ".onnx"

    print(f"[INFO] 加载模型: {pt_path}")
    model = YOLO(pt_path)

    device = "cuda" if torch.cuda.is_available() and not half else "cpu"
    if half and not torch.cuda.is_available():
        print("[WARN] 未检测到 CUDA，half 模式已回退到 CPU FP32")
        half = False

    print(f"[INFO] 导出参数: imgsz={img_size}, batch={batch_size}, opset={opset}, "
          f"half={half}, dynamic={dynamic}, simplify={simplify}, in_channels={in_channels}")

    # 使用 ultralytics 内置导出接口
    exported = model.export(
        format="onnx",
        imgsz=img_size,
        batch=batch_size,
        opset=opset,
        half=half,
        dynamic=dynamic,
        simplify=simplify,
    )

    # ultralytics 默认将 onnx 保存在与 pt 同目录，重命名/移动到指定路径
    default_onnx = os.path.splitext(pt_path)[0] + ".onnx"
    if exported and str(exported) != output_path and os.path.exists(default_onnx):
        os.replace(default_onnx, output_path)

    # 验证导出结果
    if os.path.exists(output_path):
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print(f"[INFO] ONNX 模型验证通过")
        print(f"[INFO] 输出保存至: {output_path}")

        # 打印输入/输出信息
        for inp in onnx_model.graph.input:
            shape = [d.dim_value for d in inp.type.tensor_type.shape.dim]
            print(f"  输入: {inp.name}  shape={shape}")
        for out in onnx_model.graph.output:
            shape = [d.dim_value for d in out.type.tensor_type.shape.dim]
            print(f"  输出: {out.name}  shape={shape}")
    else:
        print("[ERROR] 导出失败，未找到 onnx 文件")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".pt 转 .onnx 导出工具")
    parser.add_argument("--pt",       type=str, required=True,       help="输入 .pt 文件路径")
    parser.add_argument("--output",   type=str, default=None,        help="输出 .onnx 文件路径（默认同目录同名）")
    parser.add_argument("--imgsz",    type=int, default=640,         help="输入图像尺寸 (default: 640)")
    parser.add_argument("--batch",    type=int, default=1,           help="batch size (default: 1)")
    parser.add_argument("--opset",    type=int, default=17,          help="ONNX opset 版本 (default: 17)")
    parser.add_argument("--half",     action="store_true",           help="导出 FP16（需要 CUDA）")
    parser.add_argument("--dynamic",  action="store_true",           help="动态 batch/shape")
    parser.add_argument("--simplify", action="store_true", default=True, help="使用 onnx-simplifier 简化（default: True）")
    parser.add_argument("--channels", type=int, default=6,           help="输入通道数，双流=6，单流=3 (default: 6)")
    args = parser.parse_args()

    export_onnx(
        pt_path=args.pt,
        output_path=args.output,
        img_size=args.imgsz,
        batch_size=args.batch,
        opset=args.opset,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
        in_channels=args.channels,
    )
