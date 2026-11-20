# 训练
from ultralytics import YOLO
import ultralytics.nn.tasks
import torch
from ultralytics.nn.tasks import DetectionModel

# 设置PyTorch全局配置，如果需要 weights_only=True 加载模型
torch.serialization.add_safe_globals(
    [
        DetectionModel,
        torch.optim.SGD,  # 添加优化器类
        torch.optim.Adam,
    ]
)

# 模型选择：切换注释即可
model = YOLO('./yaml/dual-stream-vit-yolov8.yaml')  # 双流 ViT（窗口化双向跨模态注意力）
# model = YOLO('./yaml/hybrid-vif-yolo.yaml')     # Hybrid VIF-YOLO (Add + AttnFusion + LMF)
# model = YOLO('./runs/detect/train25/weights/best.pt')
# model = YOLO("./yaml/VIF-YOLO.yaml")  # VIF-YOLO (EMA + LMF)
# model = YOLO('./yaml/PC2f_MPF_yolov8n.yaml')  # PC2f_MPF (原始)

# Prefer CUDA when available; otherwise fall back to CPU to avoid startup errors.
device = 0 if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

results = model.train(
    data="/media/lyric/linux_extend/dsc/Fusion_Dataset/yolo/data.yaml",
    batch=16,
    epochs=200,
    device=device,
    patience=180,
    imgsz=640,
    mixup=0.1,
    copy_paste=0.1,
    close_mosaic=10,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    mosaic=1.0,
    auto_augment="randaugment",
    erasing=0.4,
)
