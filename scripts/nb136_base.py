# pass A: every sample once, eager fp32 -> latent, pred, truth, logits, input summaries
import sys, time, numpy as np, torch
from pathlib import Path
ROOT = Path("/home/ethantu/workspace/good-vibrations"); sys.path.insert(0, str(ROOT / "src9"))
from dataset import build_dataset, CATEGORIES
from arch.transformer4 import Transformer4
OUT = ROOT / "runs/_src9_cache/nb136"; OUT.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False

tr_dl, ev, tr_ev, info = build_dataset("tukey", num_workers=0)
model = Transformer4(info, decoder="res", latent_norm="rescale").cuda().eval()
sd = torch.load(ROOT / "runs/decres-rescale-aug10-v1/checkpoints/ep300-ba3000-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"]
print(model.load_state_dict(sd, strict=True))
torch.save({k: v for k, v in info.items()}, OUT / "info.pt")

jobs = [("train", tr_ev.dataloader.dataloader.dataset)] + [(e.label, e.dataloader.dataloader.dataset) for e in ev]
idx, DS = [], {"train": jobs[0][1].dataset, "eval": jobs[1][1].dataset}
for name, sub in jobs: idx += [(name, i) for i in sub.indices]
print(len(idx), "samples")
R = {k: [] for k in "u p y obj box spec lb split i".split()}
t0 = time.time()
with torch.no_grad():
    for s in range(0, len(idx), 128):
        rows = [DS["train" if n == "train" else "eval"][i] for n, i in idx[s:s + 128]]
        x = torch.stack([r["fft"] for r in rows]).cuda()
        o = model(dict(fft=x))
        R["u"].append(o["latent"].half().cpu()); R["p"].append((o["mask_pred"] * 255).round().byte().cpu())
        R["y"].append(torch.stack([r["mask_true"] > 0.5 for r in rows])); R["obj"].append(o["object_logits"].cpu()); R["box"].append(o["box_logits"].cpu())
        lx = (x + 1e-3).log()                                                    # (B,L,F,C) log |X| (each sample already / its std)
        R["spec"].append(lx.mean(1).half().cpu())                                # (B,F,C) laser-mean log spectrum
        R["lb"].append(lx.view(len(x), 100, 18, 65, 2).mean((3, 4)).half().cpu())  # (B,L,18) per-laser per-band mean log mag
        R["split"] += [n for n, _ in idx[s:s + 128]]; R["i"] += [i for _, i in idx[s:s + 128]]
        if s % 2560 == 0: print(s, f"{time.time() - t0:.0f}s", flush=True)
np.savez(OUT / "base.npz", **{k: (torch.cat(v).numpy() if isinstance(v[0], torch.Tensor) else np.array(v)) for k, v in R.items()})
print("done", time.time() - t0)
