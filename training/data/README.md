# Training Dataset

## Classes
| ID | Name   | Purpose |
|----|--------|---------|
| 0  | person | Primary detection target |
| 1  | face   | On-device blurring only — not used for identity |
| 2  | AGV    | Autonomous Guided Vehicle |

## Sources

### Person + Face
- **faces4coco** — face bounding box annotations layered onto COCO images.
  Provides co-located person and face labels in a single dataset.

### AGV
Seven Roboflow Universe datasets merged into a single AGV class:

| Dataset | URL |
|---------|-----|
| YOLO dataset | https://universe.roboflow.com/yolo-klrw8/yolo-kiqlv |
| Symovo | https://universe.roboflow.com/amrs-htanh/symovo |
| DH:CONTACT | https://universe.roboflow.com/ink-studios/dh-contact-sglu9 |
| Dataset AGVS | https://universe.roboflow.com/barioni/dataset-agvs |
| AMR | https://universe.roboflow.com/amr-njd6r/amr-fvupu |
| Amazon | https://universe.roboflow.com/georgia-southern/amazon-bkhql |
| AGV-P | https://universe.roboflow.com/barioni/agv-p |

## Split
- 80/20 train/val split at scene level
- ~9,500 train images / ~1,500 val images
- Label format: YOLO (normalized xywh)

## Notes
- Hard-case sampling (occluded, blurred persons) was attempted but COCO lacks
  `occluded`/`blur` annotation keys — all extra person images ended up as
  normal and multi-person categories only.
- The face class is trained for on-device privacy blurring. It is not used
  for identity recognition.
