from models.experimental import attempt_load
import torch

model = attempt_load("runs/train/yolov5n_3c3/weights/best.pt")
model.eval()

x = torch.zeros(1, 3, 640, 640)
with torch.no_grad():
    model(x)
