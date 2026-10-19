"""Grad-CAM family (jacobgil/pytorch-grad-cam) on the ModalFormer: where in (laser, frequency) the evidence for the mask sits.
The target layer is the frequency stem's last conv (conv3): its output is (B,128,L,P), one 128-channel token per laser and ~49 Hz band,
so a CAM on it is a laser x band map; the library upsamples it to the input's (L,F) = 100 x 1170 (it reads the input as (B,C,H,W),
so the model is wrapped to take (B,C,L,F)). Target per sample: soft iou of the predicted mask with the true mask.
  GradCAM   channel weights = mean gradient over (laser, band); ReLU(sum_c w_c A_c)
  HiResCAM  ReLU(sum_c grad_c * A_c) elementwise: provably faithful to the model for this head (Draelos & Carin 2020)
  LayerCAM  ReLU(sum_c ReLU(grad_c) * A_c): per-location weights, fine-grained
Caveats: CAMs are unsigned (ReLU: what argues FOR the mask, not against) and the library min-max normalizes each sample to [0, 1], so
they say where, not how much -- compare shapes with the occlusion maps (importance.py), not magnitudes.
  .venv/bin/python src8/gradcam.py RUN [--n 200]   -> runs/<RUN>/gradcam.pt"""
import argparse, sys
from pathlib import Path

import torch
from pytorch_grad_cam import GradCAM, HiResCAM, LayerCAM

sys.path.insert(0, str(Path(__file__).parent))
from probe import ROOT, eval_data, load_model, box_index, run_args
from importance import CORE, BAND

class Wrap(torch.nn.Module):
    """(B,C,L,F) -> (B,H,W) mask probabilities, for the box set in .box"""
    def __init__(self, m): super().__init__(); self.m, self.box = m, None
    def forward(self, x): return self.m(dict(fft=x.permute(0, 2, 3, 1), box=self.box))["mask_pred"].float()

class SoftIoU:
    def __init__(self, t): self.t = t
    def __call__(self, p): return torch.minimum(p, self.t).sum() / torch.maximum(p, self.t).sum().clamp_min(1e-6)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--n", type=int, default=200); ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    ra = run_args(a.run)
    X, y, labels, info = eval_data("gastronorm_plastic_cardboard", ra.signal, ra.window, laser_center=bool(getattr(ra, "laser_center", 0)))
    keep = torch.tensor([l.split("/")[1] in CORE or l == "cardboard/soap-dispenser-rotated" for l in labels])
    labels = [l for l, k in zip(labels, keep) if k]
    X, y, box = X[keep], y[keep], box_index(labels, info)
    nb, L, F_ = len(info["boxes"]), X.shape[1], X.shape[2]
    P = -(-F_ // BAND)
    g = torch.Generator().manual_seed(0)  # the same balanced subsample as importance.py's joint map
    sub = torch.cat([torch.nonzero(box == b)[:, 0][torch.randperm(int((box == b).sum()), generator=g)[:a.n]] for b in range(nb)])
    ck = sorted((ROOT / "runs" / a.run / "checkpoints").glob("ep*-rank0.pt"), key=lambda p: int(p.name[2:].split("-")[0]))[-1]
    m = load_model("src8", a.run, ck, info, {}).eval()
    enc = next(mod for mod in m.modules() if type(mod).__name__ == "ModalFormer")
    w = Wrap(m)
    out = dict(boxes=info["boxes"], grid=(info["n_laser_rows"], info["n_laser_cols"]), n=a.n, ckpt=ck.name, methods=["GradCAM", "HiResCAM", "LayerCAM"])
    for name, cls in (("GradCAM", GradCAM), ("HiResCAM", HiResCAM), ("LayerCAM", LayerCAM)):
        acc = torch.zeros(nb, L, F_)
        with cls(model=w, target_layers=[enc.conv3]) as cam:
            for k in range(0, len(sub), a.bs):
                i = sub[k:k + a.bs]
                w.box = box[i].cuda()
                c = torch.as_tensor(cam(input_tensor=X[i].float().permute(0, 3, 1, 2).cuda(), targets=[SoftIoU(t) for t in y[i].cuda()]))  # (n,L,F) in [0,1]
                for b in range(nb): acc[b] += c[box[i] == b].sum(0)
        acc /= torch.stack([(box[sub] == b).sum() for b in range(nb)]).float()[:, None, None]
        out[name] = acc                                                                        # (nb,L,F) mean normalized CAM
        out[name + "_band"] = torch.stack([acc[:, :, j * BAND:(j + 1) * BAND].mean(-1) for j in range(P)], -1)  # (nb,L,P)
        print(f"{name} done", flush=True)
    lo, hi = info["freq_range"]
    out |= dict(hz=torch.tensor([lo + (j * BAND + BAND / 2) * (hi - lo) / F_ for j in range(P)]), bin_hz=torch.linspace(lo, hi, F_))
    torch.save(out, ROOT / "runs" / a.run / "gradcam.pt")
    print(f"wrote runs/{a.run}/gradcam.pt", flush=True)

if __name__ == "__main__":
    main()
