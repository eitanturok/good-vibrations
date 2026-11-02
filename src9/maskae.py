"""mask autoencoder for transformer3 --maskae: a conv AE on the train split's true 64x64 masks (one per position), z = LN(32-d, no affine) + N(0, 0.1)
noise while training, bce + dice. Saves runs/maskae/maskae.pt and prints reconstruction iou (> 0.5) on train / eval id / eval ood / unseen-object.
  .venv/bin/python src9/maskae.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from dataset import CATEGORIES, key, load_splits, downsample
from callbacks import EvalSummary
from utils import bce_dice

class MaskAE(nn.Module):
    """(B,64,64) mask -> 4 stride-2 convs 32..256 -> Linear -> z = LN (B,z_dim), unit scale -> Linear -> 256 x 4x4 -> 4 transposed convs -> (B,64,64) logits"""
    def __init__(self, z_dim=32):
        super().__init__()
        self.z_dim, cs = z_dim, (1, 32, 64, 128, 256)
        self.enc = nn.Sequential(*[l for a, b in zip(cs, cs[1:]) for l in (nn.Conv2d(a, b, 4, 2, 1), nn.ReLU())], nn.Flatten(), nn.Linear(256 * 16, z_dim))
        self.dec = nn.Sequential(nn.Linear(z_dim, 256 * 16), nn.Unflatten(1, (256, 4, 4)),
                                 *[l for a, b in zip((256, 128, 64, 32), (128, 64, 32, 32)) for l in (nn.ConvTranspose2d(a, b, 4, 2, 1), nn.ReLU())], nn.Conv2d(32, 1, 3, padding=1))

    def encode(self, m): return F.layer_norm(self.enc(m[:, None]), (self.z_dim,))  # (B,64,64) -> (B,z_dim); tanh saturated to +-1 and collapsed to empty masks
    def decode(self, z): return self.dec(z)[:, 0]                                     # (B,z_dim) -> (B,64,64) logits

def load(path):
    """frozen MaskAE from a maskae.pt"""
    ck = torch.load(path)
    ae = MaskAE(**ck["args"]); ae.load_state_dict(ck["state"])
    return ae.requires_grad_(False).eval()

def masks(samples, idxs):
    """one 64x64 true mask per position, as dataset.process_y makes it (a with-lid sample takes its without-lid twin's)"""
    metas, where, seen, out = list(samples.values()), {(m["experiment"], m["position_id"], m["speaker"]): d for d, m in samples.items()}, set(), []
    dirs = list(samples)
    for i in idxs:
        if key(metas[i]) in seen: continue
        seen.add(key(metas[i]))
        d, m = dirs[i], metas[i]
        if "without_lid_position_id" in m: d = where[m["experiment"], m["without_lid_position_id"], m["speaker"]]
        out.append(downsample(np.asarray(Image.open(d / "image/03_smask.png").convert("F")) / 255.0, 64, 64))
    return torch.from_numpy(np.stack(out)) if out else torch.zeros(0, 64, 64)

def recon_iou(ae, y):
    with torch.no_grad(): b, o = ae.decode(ae.encode(y)) > 0, y > 0.5
    keep = o.flatten(1).any(1)
    return ((b & o).flatten(1).sum(1) / (b | o).flatten(1).sum(1).clamp_min(1))[keep].float()

def main(epochs=40, z_dim=32, noise=0.1, out="runs/maskae/maskae.pt"):
    torch.manual_seed(0)
    samples, idxs = load_splits()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    train = masks(samples, idxs["train"]).to(dev)
    cat = {f"eval/{b}/{s}": c for b, cs in CATEGORIES.items() for c, ss in cs.items() for s in ss}
    ev = {c: masks(samples, [i for n, ids in idxs.items() if cat.get(n) in cs for i in ids]).to(dev)
          for c, cs in (("eval id", EvalSummary.ID), ("eval ood", EvalSummary.OOD), ("unseen-object", ("unseen-object",)))}
    print({"train": len(train), **{k: len(v) for k, v in ev.items()}}, "positions")
    ae = MaskAE(z_dim).to(dev)
    opt = torch.optim.AdamW(ae.parameters(), 1e-3, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=epochs * ((len(train) + 63) // 64))
    for ep in range(epochs):
        ae.train()
        for i in torch.randperm(len(train), device=dev).split(64):
            y = train[i]
            y = torch.where(torch.rand(len(y), 1, 1, device=dev) < 0.5, y.flip(-1), y)  # random flips
            y = torch.where(torch.rand(len(y), 1, 1, device=dev) < 0.5, y.flip(-2), y)
            z = ae.encode(y)
            loss = bce_dice(ae.decode(z + noise * torch.randn_like(z)), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 10 == 9 or ep == epochs - 1:
            ae.eval()
            print(f"ep {ep + 1} loss {loss.item():.4f}  recon iou: train {recon_iou(ae, train).mean():.3f}  " + "  ".join(f"{k} {recon_iou(ae, v).mean():.3f}" for k, v in ev.items()))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=ae.cpu().state_dict(), args=dict(z_dim=z_dim)), out)
    print("saved", out)

if __name__ == "__main__":
    main()
