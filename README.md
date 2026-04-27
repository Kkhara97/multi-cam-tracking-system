# multi-cam-tracking-system

Single-camera C++ tracking pipeline (YOLOv5 NCNN + Kalman tracker + OSNet Re-ID + SimplePose + MQTT) running on Raspberry Pi 5. Phase 2 extends to two cameras with cross-camera identity association.

## Folder Structure

```
multi-cam-tracking-system/
├── training/
│   ├── dataset_prep/    # Dataset preparation scripts (COCO-face + AGV, 3-class)
│   ├── configs/         # yolo3c.yaml, hyp_custom.yaml
│   └── results/         # mAP curves, training CSVs
├── reid/                # OSNet Re-ID export and inspection scripts
├── export/              # Model validation scripts
├── pi_node/             # C++ pipeline source (apps, components, config, CMakeLists)
├── calibration/         # Camera calibration scripts and YAML outputs
├── server/              # MQTT subscriber display script
├── results/             # Benchmark CSVs, tracker comparison, Re-ID baseline
└── docs/                # Design decisions and export chain documentation
```

> Full build and run instructions: see README (Block H — in progress).
