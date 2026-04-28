"""
Real-time webcam detector using a custom YOLOv5 3-class model.

Classes: 0=person (green), 1=face (blue), 2=AGV (red)

Usage:
    python detector.py
    python detector.py --weights runs/train/yolov5n_3c3/weights/best.pt --source 0

Dependencies:
    pip install torch torchvision opencv-python
    YOLOv5 repo must be on sys.path (run from inside yolov5/)
"""
import argparse
import time
import cv2
import torch


def parse_args():
    p = argparse.ArgumentParser(description="YOLOv5 3-class webcam detector")
    p.add_argument("--weights", default="runs/train/yolov5n_3c3/weights/best.pt",
                   help="Path to model weights (.pt)")
    p.add_argument("--source", type=int, default=0,
                   help="Webcam index (default: 0)")
    p.add_argument("--conf", type=float, default=0.35,
                   help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.45,
                   help="NMS IoU threshold")
    return p.parse_args()


CLASS_NAMES = ['person', 'face', 'AGV']
COLORS = {0: (0, 255, 0), 1: (255, 0, 0), 2: (0, 0, 255)}


def main():
    args = parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = torch.hub.load('ultralytics/yolov5', 'custom',
                           path=args.weights, force_reload=False)
    model.to(device).eval()
    model.conf = args.conf
    model.iou = args.iou

    cap = cv2.VideoCapture(args.source, cv2.CAP_DSHOW)
    assert cap.isOpened(), f'Failed to open webcam {args.source}'

    prev = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, size=640)
        det = results.xyxy[0].cpu().numpy()  # [x1,y1,x2,y2,conf,cls]

        for *xyxy, conf, cls in det:
            x1, y1, x2, y2 = map(int, xyxy)
            cls = int(cls)
            label = f'{CLASS_NAMES[cls]} {conf:.2f}'
            color = COLORS[cls]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        curr = time.time()
        fps = 1 / (curr - prev) if prev else 0
        prev = curr
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow('YOLOv5 webcam - press q to quit', frame)
        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
