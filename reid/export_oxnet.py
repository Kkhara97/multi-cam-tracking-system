"""
Export OSNet x0.25 to TorchScript with baked-in ImageNet normalisation and L2 normalisation.

Input:  OSNet checkpoint (.pth) trained with TorchReID
Output: osnet_embed_ts_new.pt  — TorchScript model, input [B,3,256,128] RGB float [0,1],
                                  output [B,512] L2-normalised embedding

Usage:
    python export_oxnet.py
    python export_oxnet.py --checkpoint path/to/osnet.pth

Dependencies:
    pip install torch
    pip install git+https://github.com/KaiyangZhou/deep-person-reid.git
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchreid

DEFAULT_CKPT = (
    "osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0.003"
    "_b128_fb10_softmax_labelsmooth_flip.pth"
)


def parse_args():
    p = argparse.ArgumentParser(description="Export OSNet to TorchScript")
    p.add_argument("--checkpoint", default=DEFAULT_CKPT,
                   help="Path to OSNet .pth checkpoint")
    return p.parse_args()


class OSNetEmbed(nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,3,256,128], RGB, float in [0,1]
        x = (x - self.mean) / self.std
        out = self.base(x)
        if isinstance(out, (tuple, list)):
            feat = out[0]
        else:
            feat = out
        return F.normalize(feat, p=2, dim=1)  # [B, 512]


def main():
    args = parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    num_classes = state["classifier.weight"].shape[0]
    print(f"Detected num_classes = {num_classes}")

    base = torchreid.models.build_model(
        "osnet_x0_25", num_classes=num_classes, loss="softmax", pretrained=False
    )
    base.load_state_dict(state, strict=True)
    base.eval()

    net = OSNetEmbed(base).eval()
    example = torch.rand(1, 3, 256, 128)
    ts = torch.jit.trace(net, example)
    ts.save("osnet_embed_ts_new.pt")
    print("Saved osnet_embed_ts_new.pt")


if __name__ == "__main__":
    main()