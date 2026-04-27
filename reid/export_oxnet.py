# export_osnet_embed_ts.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreid

# 1) Load checkpoint safely
ckpt = torch.load(
    "osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth",
    map_location="cpu",
    weights_only=False  # set True if your PyTorch version supports and ckpt is safe
)
state = ckpt.get("state_dict", ckpt)
num_classes = state["classifier.weight"].shape[0]
print(f"Detected num_classes = {num_classes}")

# 2) Build model; 'loss' value no longer matters since we'll wrap it
base = torchreid.models.build_model(
    "osnet_x0_25",
    num_classes=num_classes,
    loss="softmax",  # any is fine; wrapper ignores tuple/logit branches
    pretrained=False
)
base.load_state_dict(state, strict=True)
base.eval()

# 3) Wrapper: RGB in [0,1], ImageNet mean/std, returns L2-normalized embedding
class OSNetEmbed(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,256,128], RGB, float in [0,1]
        x = (x - self.mean) / self.std
        out = self.base(x)
        # TorchReID may return (feat, logits) depending on loss; ensure Tensor
        if isinstance(out, (tuple, list)):
            feat = out[0]
        else:
            feat = out
        feat = F.normalize(feat, p=2, dim=1)  # L2 unit vector
        return feat  # shape [B, 512] for osnet_x0_25

net = OSNetEmbed(base).eval()

# 4) Prefer TRACE for this wrapper (avoids the typing issue in base.forward)
example = torch.rand(1, 3, 256, 128)  # RGB in [0,1]
ts = torch.jit.trace(net, example)
ts.save("osnet_embed_ts_new.pt")
print("✅ Saved osnet_embed_ts.pt (L2-normalized 512-D embeddings)")
