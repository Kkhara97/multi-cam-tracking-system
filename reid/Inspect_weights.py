import torch

sd = torch.load(r"osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth", map_location="cpu")
# if it’s wrapped, unwrap:
if "state_dict" in sd: sd = sd["state_dict"]

for name in sd.keys():
    if "classifier" in name or "bnneck" in name:
        print(name)

