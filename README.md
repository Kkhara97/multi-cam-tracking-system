# Multi-Camera Person Tracking System

Real-time person tracking pipeline on Raspberry Pi 5. Single-camera system is complete and deployed. Phase 2 extends to two cameras with cross-camera identity association.

**Stack:** YOLOv5n NCNN · OSNet Re-ID NCNN · Kalman tracker · SimplePose NCNN · MQTT · C++17

---

## Hardware

| Component | Model |
|-----------|-------|
| Edge node | Raspberry Pi 5 (×2 for Phase 2) |
| Camera 1 | UGREEN USB camera |
| Camera 2 | NexiGo USB camera |
| Server | Lenovo Legion laptop |

---

## Repository Structure

```
multi-cam-tracking-system/
├── pi_node/             # C++ pipeline: apps, components, config, CMakeLists
│   ├── apps/            # pi_main_vnext.cpp — main pipeline entry point
│   ├── components/      # yolo_detector, osnet_embedder, local_tracker, etc.
│   └── config/          # cam1.json runtime config
├── training/
│   ├── configs/         # yolo3c.yaml, hyp_custom.yaml
│   └── results/         # Training CSVs and plots (yolov5n_3c3, yolov5s_3c_ft)
├── reid/                # OSNet TorchScript export script
├── calibration/         # Camera calibration script and YAML outputs
├── server/              # MQTT subscriber + display script (Phase 2)
└── results/             # Pipeline benchmark CSVs, Re-ID baseline
```

---

## Model Weights

Binary model files are not stored in this repo. Download from [Release v1.0](../../releases/tag/v1.0).

| File | Purpose |
|------|---------|
| `yolov5n_3c_ts.ncnn.bin` + `.param` | Deployed 3-class detector (NCNN) |
| `osnet_embed_ts_new.ncnn.bin` + `.param` | Re-ID embedder (NCNN) |
| `Ultralight-Nano-SimplePose.bin` + `.param` | Pose estimator (NCNN) |
| `osnet_embed_ts_new.pt` | Re-ID TorchScript (for re-conversion) |
| `yolov5s_3c_ft_best.pt` | Best trained weights — not yet converted to NCNN |

Place `.bin` and `.param` files in `pi_node/models/` before building.

---

## Training

### Dataset

**People + faces:** [faces4coco](https://github.com/matteorr/coco-analyze) — COCO images with face annotations layered on top.

**AGV class:** 7 Roboflow Universe datasets merged into a single AGV class:

| Dataset | URL |
|---------|-----|
| YOLO dataset | https://universe.roboflow.com/yolo-klrw8/yolo-kiqlv |
| Symovo | https://universe.roboflow.com/amrs-htanh/symovo |
| DH:CONTACT | https://universe.roboflow.com/ink-studios/dh-contact-sglu9 |
| Dataset AGVS | https://universe.roboflow.com/barioni/dataset-agvs |
| AMR | https://universe.roboflow.com/amr-njd6r/amr-fvupu |
| Amazon | https://universe.roboflow.com/georgia-southern/amazon-bkhql |
| AGV-P | https://universe.roboflow.com/barioni/agv-p |

**Split:** 80/20 train/val at scene level (~9,500 train / ~1,500 val)  
**Classes:** 0 = person · 1 = face · 2 = AGV

> The face class is detected on-device for privacy blurring only — not used for identity.

### Results

| Run | Backbone | Epochs | Precision | Recall | mAP@0.5:0.95 | Status |
|-----|----------|--------|-----------|--------|--------------|--------|
| yolov5n_3c3 | YOLOv5n | 200 | 0.9049 | 0.8475 | 0.6394 | ✅ Deployed on Pi (NCNN) |
| yolov5s_3c_ft | YOLOv5s | 120 | 0.9237 | 0.8559 | 0.6721 | ⏳ Phase 2 (not yet converted to NCNN) |

Full result plots and CSVs: `training/results/`

> The deployed model is YOLOv5n (nano). YOLOv5s achieves higher mAP but has not been converted to NCNN and benchmarked on Pi hardware — nano vs small latency comparison is a Phase 2 task.

### Training Config

```yaml
# yolo3c.yaml — dataset config
nc: 3
names: ['person', 'face', 'AGV']

# hyp_custom.yaml — key differences from Ultralytics default
cls: 0.3        # default 0.5 — lower class loss weight
obj: 0.7        # default 1.0 — lower objectness loss weight
copy_paste: 0.1 # default 0.0 — enables copy-paste augmentation
degrees: 5.0    # default 0.0 — enables small rotation augmentation
```

Full configs: `training/configs/`

---

## Export Chain

All conversion runs on Windows. Outputs are copied manually to the Pi.  
**Tool versions:** NCNN 2025-05-03 · PNNX 2025-07-25

### YOLOv5 → NCNN

```bash
# Step 1 — export to TorchScript (run from inside yolov5/)
python export.py --weights runs/train/yolov5n_3c3/weights/best.pt --include torchscript
# Output: yolov5n_3c_ts.pt

# Step 2 — convert to NCNN via PNNX
pnnx yolov5n_3c_ts.pt inputshape=[1,3,640,640]
# Output: yolov5n_3c_ts.ncnn.bin, yolov5n_3c_ts.ncnn.param
# PNNX auto-applies: fp16=1, optlevel=2, device=cpu
```

### OSNet → NCNN

```bash
# Step 1 — export wrapped OSNet to TorchScript
python reid/export_oxnet.py --checkpoint <path/to/osnet_checkpoint.pth>
# Output: osnet_embed_ts_new.pt  (ImageNet normalisation + L2 norm baked in)

# Step 2 — convert to NCNN via PNNX
pnnx osnet_embed_ts_new.pt inputshape=[1,3,256,128]
# Output: osnet_embed_ts_new.ncnn.bin, osnet_embed_ts_new.ncnn.param
```

PNNX: https://github.com/pnnx/pnnx
