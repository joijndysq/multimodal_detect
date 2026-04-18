#训练
from ultralytics import YOLO
import ultralytics.nn.tasks
import torch
from ultralytics.nn.tasks import DetectionModel

# 设置PyTorch全局配置，如果需要 weights_only=True 加载模型
torch.serialization.add_safe_globals([
    DetectionModel,
    torch.optim.SGD,  # 添加优化器类
    torch.optim.Adam
])
model = YOLO('./yaml/PC2f_MPF_yolov8n.yaml')
# model = YOLO('runs/detect/train16/weights/best.pt')

# Prefer CUDA when available; otherwise fall back to CPU to avoid startup errors.
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

results = model.train(data='/home/lyric/ir_rgb/TwoStream_Yolov8/data/drone2.yaml', batch=16, epochs=3, device=device)
# 临时允许加载自定义模型
