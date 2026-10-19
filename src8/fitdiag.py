"""Quick fitting diagnostic: how fast does an architecture fit, on the cached eval tensors (probe.py's cache: 4560 samples, log_power
tukey), mask loss only? 80/20 split by position. Prints train bce (last 50 steps) and held-out iou@0.08 per config.
  .venv/bin/python src8/fitdiag.py [config ...] [--steps 600]"""
import argparse, sys, time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from arch import Boombox
from probe import ROOT, box_index

CONFIGS = {  # name: (encoder, d_model, lr, modal kwargs); kwargs "decoder" goes to Boombox
    "conv":            ("conv", 1024, 1e-4, {}),
    "modal":           ("modal", 256, 1e-4, {}),
    "modal-lr5e-4":    ("modal", 256, 5e-4, {}),
    "modal-lr1e-3":    ("modal", 256, 1e-3, {}),
    "modal-noaxial":   ("modal", 256, 5e-4, dict(order="")),
    "modal-nogeo":     ("modal", 256, 5e-4, dict(geo=False)),
    "modal-d256":      ("modal", 256, 5e-4, dict(d=256)),
    "modal-nobn":      ("modal", 256, 5e-4, dict(box_norm=False)),
    "modal-attnpool":  ("modal", 256, 5e-4, dict(pool="attn")),
    "modal-grid8":     ("modal", 256, 5e-4, dict(grid=8)),
    "modal-dec4":      ("modal", 256, 5e-4, dict(dec_layers=4)),
    "modal-dec6":      ("modal", 256, 5e-4, dict(dec_layers=6)),
    "modal-dec4-noax": ("modal", 256, 5e-4, dict(dec_layers=4, order="")),
    "hybrid":          ("modal", 1024, 5e-4, dict(decoder="conv")),
    "hybrid-noax":     ("modal", 1024, 5e-4, dict(decoder="conv", order="")),
    "hybrid-lr1e-4":   ("modal", 1024, 1e-4, dict(decoder="conv")),
    "hybrid-noax-lr1e-4": ("modal", 1024, 1e-4, dict(decoder="conv", order="")),
    "conv-lr5e-4":     ("conv", 1024, 5e-4, {}),
    "hybrid-noax-linskip": ("modal", 1024, 1e-4, dict(decoder="conv", order="", lin_skip=256)),
    "conv-lr3e-4":     ("conv", 1024, 3e-4, {}),
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=list(CONFIGS))
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--bs", type=int, default=256)
    a = ap.parse_args()
    c = torch.load(ROOT / "runs/_probe_cache/gastronorm_plastic_cardboard_log_power_tukey.pt")
    X, y, labels = c["X"], c["y"], c["labels"]
    info = dict(n_laser_rows=10, n_laser_cols=10, n_freqs=X.shape[2], n_channels=2, freq_range=(100.0, 1000.0), objects=["a", "unseen"],
                n_objects=[0, 1, 2], boxes=["cardboard", "gastronorm", "green-plastic"], signal="log_power")
    box = box_index(labels, info)
    g = torch.Generator().manual_seed(0)
    test = torch.rand(len(X), generator=g) < 0.2  # sample-level; positions repeat over speakers, so this is an optimistic held-out
    tr, te = (~test).nonzero()[:, 0], test.nonzero()[:, 0]
    for name in a.configs:
        enc, dm, lr, kw = CONFIGS[name]
        torch.manual_seed(0)
        lw = dict(object=0, n_objects=0, com=0, area=0, box=0)
        kw = dict(kw); dec = kw.pop("decoder", "query") if enc == "modal" else "conv"
        m = Boombox(dm, info, 64, 64, encoder=enc, decoder=dec, loss_weights=lw, modal=kw).cuda()
        opt = torch.optim.AdamW(m.parameters(), lr, weight_decay=0.01)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 100))
        t0, losses = time.time(), []
        for step in range(a.steps):
            i = tr[torch.randint(len(tr), (a.bs,), generator=g)]
            b = dict(fft=X[i].cuda().float(), box=box[i].cuda(), mask_true=y[i].cuda())
            with torch.autocast("cuda", torch.bfloat16):
                out = m(b); loss = m.mask_loss(out["mask_logits"].float(), b["mask_true"])
            loss.backward(); opt.step(); opt.zero_grad(); sched.step()
            losses.append(loss.item())
        m.eval(); ious = []
        with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
            for j in range(0, len(te), 256):
                i = te[j:j + 256]
                p = m(dict(fft=X[i].cuda().float(), box=box[i].cuda()))["mask_pred"].float() > 0.08
                o = y[i].cuda() > 0.5
                k = o.flatten(1).any(1)
                ious.append(((p & o).flatten(1).sum(1) / (p | o).flatten(1).sum(1).clamp_min(1))[k].cpu())
        print(f"{name:16s} lr {lr:.0e}  train bce {sum(losses[-50:]) / 50:.4f} (step 100 {sum(losses[90:110]) / 20:.4f})  held-out iou@.08 {torch.cat(ious).mean():.3f}  {time.time() - t0:.0f}s", flush=True)
        del m, opt; torch.cuda.empty_cache()

if __name__ == "__main__": main()
