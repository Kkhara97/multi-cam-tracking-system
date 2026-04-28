# Model Export Chain

Converts trained YOLOv5 and OSNet weights to NCNN format for deployment on Raspberry Pi 5.
All conversion was done on Windows; output files were copied manually to the Pi.

## Tool Versions
| Tool | Version |
|------|---------|
| NCNN | 2025-05-03 |
| PNNX | 2025-07-25 |

Install PNNX: https://github.com/pnnx/pnnx

---

## YOLOv5 → NCNN

**Step 1 — Export TorchScript** (run from inside `yolov5/`):
```bash
python export.py --weights runs/train/yolov5n_3c3/weights/best.pt --include torchscript
```
Output: `yolov5n_3c_ts.pt`

**Step 2 — TorchScript → NCNN via PNNX**:
```bash
pnnx yolov5n_3c_ts.pt inputshape=[1,3,640,640]
```
Outputs: `yolov5n_3c_ts.ncnn.bin`, `yolov5n_3c_ts.ncnn.param`

PNNX settings applied automatically: `fp16=1`, `optlevel=2`, `device=cpu`

---

## OSNet → NCNN

**Step 3 — Export OSNet TorchScript** (run from repo root):
```bash
python reid/export_oxnet.py --checkpoint osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth
```
Output: `osnet_embed_ts_new.pt`

**Step 4 — TorchScript → NCNN via PNNX**:
```bash
pnnx osnet_embed_ts_new.pt inputshape=[1,3,256,128]
```
Outputs: `osnet_embed_ts_new.ncnn.bin`, `osnet_embed_ts_new.ncnn.param`

---

## Deployed Models
| Model | Source weights | NCNN files |
|-------|---------------|------------|
| YOLOv5n 3-class | `yolov5n_3c3/weights/best.pt` | `yolov5n_3c_ts.ncnn.{bin,param}` |
| OSNet x0.25 | `osnet_x0_25_msmt17...flip.pth` | `osnet_embed_ts_new.ncnn.{bin,param}` |

> `yolov5s_3c_ft` was trained (mAP@0.5:0.95=0.6721) but not converted to NCNN.
> YOLOv5n (mAP@0.5:0.95=0.6394) is the deployed model. YOLOv5s NCNN conversion is Phase 2.

Binary model files are attached to [GitHub Release v1.0](https://github.com/Kkhara97/multi-cam-tracking-system/releases).
