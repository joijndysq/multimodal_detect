# LLVIP train
from ultralytics import YOLO
import ultralytics.nn.tasks
import torch
from ultralytics.nn.tasks import DetectionModel

torch.serialization.add_safe_globals([
    DetectionModel,
    torch.optim.SGD,
    torch.optim.Adam
])

model = YOLO('/home/lyric/ir_rgb/PC2f_MPF_yolov8n_llvip.yaml')

# Prefer CUDA when available; otherwise fall back to CPU to avoid startup errors.
device = 0 if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

results = model.train(data='/home/lyric/ir_rgb/dataset/yolo/data.yaml', batch=16, epochs=50, device=device)