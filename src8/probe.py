"""What does a trained model read? Eval every eval sample under input perturbations and score iou@0.08 / miss@0.08 (score.py).
  .venv/bin/python src8/probe.py RUN [--ckpt PATH] [--signal log_power] [--window tukey] [--src src7]
X for every eval sample is cached once per (signal, window) in runs/_probe_cache (float16)."""
import argparse, importlib, json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]

def eval_data(dataset, signal, window, speakers=(1, 3, 5, 7), laser_center=False):
    """X (N,L,F,C) float16, y (N,H,W), labels [box/split], data_info, metas; the eval splits of `dataset` as run.py builds them"""
    import dataset as D
    f = ROOT / f"runs/_probe_cache/{dataset}_{signal}_{window}{'_center' if laser_center else ''}.pt"
    splits = D.DATASETS[dataset](0.2, list(speakers))
    exps = sorted({s["experiment"] for s in splits.values()})
    samples = {d: m for e in exps for d, m in D.collect_samples(D.EXPERIMENTS / e).items()}
    metas, dirs = list(samples.values()), list(samples)
    idxs = D.build_splits(splits, metas)
    info = D.get_data_info(samples, idxs["train"])
    order = [(name[5:], i) for name, ids in idxs.items() if name != "train" for i in ids]
    if f.exists(): c = torch.load(f); return c["X"], c["y"], c["labels"], info
    sub = {dirs[i]: metas[i] for _, i in order}
    X = torch.from_numpy(np.stack([x.astype(np.float16) for x in D.process_x(sub, signal, "std", None, window, laser_center=laser_center)]))
    pos = {d: k for k, d in enumerate(sub)}
    ys = D.process_y(sub, info["objects"], info["boxes"], max(info["n_objects"]), 64, 64, False)
    X = X[[pos[dirs[i]] for _, i in order]]
    y = torch.from_numpy(np.stack([ys[pos[dirs[i]]]["y"] for _, i in order]))
    labels = [n for n, _ in order]
    f.parent.mkdir(parents=True, exist_ok=True); torch.save(dict(X=X, y=y, labels=labels), f)
    return X, y, labels, info

BOXES = {"gastro": "gastronorm", "plastic": "green-plastic", "cardboard": "cardboard"}  # eval label prefix -> metadata box
def box_index(labels, info): return torch.tensor([info["boxes"].index(BOXES[l.split("/")[0]]) for l in labels])

def run_args(run):
    """a src8 run's args, with flags added after it ran filled in from run.py's defaults (so old runs rebuild); None if not src8"""
    f = ROOT / "runs" / run / "args.json"
    if not f.exists(): return None
    import os; os.environ["PYTHONUTF8"] = "1"  # run.py re-execs itself without it
    sys.path.insert(0, str(ROOT / "src8"))
    defaults = vars(importlib.import_module("run").get_parser().parse_args([]))
    return argparse.Namespace(**(defaults | json.loads(f.read_text())))

def load_model(src, run, ckpt, info, kwargs):
    """a src8 run with runs/<run>/args.json: rebuilt by run.build_model; else Boombox(**kwargs) of `src` (e.g. the src7 baseline)"""
    sys.path.insert(0, str(ROOT / src))
    if (args := run_args(run)) is not None:
        import os; os.environ["PYTHONUTF8"] = "1"  # run.py re-execs itself without it
        info = info | dict(signal=args.signal, laser_center=bool(args.laser_center))
        model = importlib.import_module("run").build_model(args, info)
    else: model = importlib.import_module("arch").Boombox(kwargs.pop("d_model", 1024), info, 64, 64, **kwargs)
    sd = torch.load(ckpt, weights_only=False, map_location="cpu")["state"]["model"]
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in sd.items()}, strict=False)
    return model.cuda().eval()

def perturbations(L=100, F=1170, rows=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    blocks = torch.randperm(F // 65, generator=g)                       # 18 blocks of 65 bins
    block_perm = (blocks[:, None] * 65 + torch.arange(65)).flatten()
    local = (torch.arange(F) // 9 * 9 + torch.stack([torch.randperm(9, generator=g) for _ in range(F // 9)]).flatten())
    laser_perm = torch.randperm(L, generator=g)
    grid = torch.arange(L).view(rows, -1)
    smooth = lambda x, k: torch.nn.functional.avg_pool1d(x.permute(0, 1, 3, 2).reshape(-1, 1, F), k, 1, k // 2, count_include_pad=False).view(x.shape[0], x.shape[1], x.shape[3], F).permute(0, 1, 3, 2)
    return {
        "none":               lambda x: x,
        "freq flip":          lambda x: x.flip(2),
        "freq roll 10%":      lambda x: x.roll(117, 2),
        "freq roll 1%":       lambda x: x.roll(12, 2),
        "freq block shuffle": lambda x: x[:, :, block_perm],
        "freq local shuffle": lambda x: x[:, :, local],
        "smooth 33 bins":     lambda x: smooth(x, 33),
        "level only":         lambda x: x.mean(2, keepdim=True).expand_as(x),
        "remove laser mean":  lambda x: x - x.mean(1, keepdim=True) + x.mean((1, 2), keepdim=True),
        "laser shuffle":      lambda x: x[:, laser_perm],
        "laser grid roll 1":  lambda x: x[:, grid.roll(1, 1).flatten()],
        "laser grid flip lr": lambda x: x[:, grid.flip(1).flatten()],
        "swap x/y channel":   lambda x: x.flip(3),
    }

@torch.no_grad()
def predict(model, X, pert, box, bs=108):
    out = []
    for i in range(0, len(X), bs):
        x = pert(X[i:i + bs].cuda().float())
        with torch.autocast("cuda", torch.bfloat16): out.append(model(dict(fft=x, box=box[i:i + bs].cuda()))["mask_pred"].float().cpu())
    return torch.cat(out)

def score(pred, y, labels, thr=0.08):
    b, obj = pred > thr, y > 0.5
    iou = (b & obj).flatten(1).sum(1) / (b | obj).flatten(1).sum(1).clamp_min(1)
    miss = (~(b & obj).flatten(1).any(1)).float()
    df = pd.DataFrame(dict(split=labels, iou=iou.numpy(), miss=miss.numpy()))
    sp = df.groupby("split")[["iou", "miss"]].mean()
    from dataset import CATEGORIES
    cat = {f"{b}/{s}": c for b, cs in CATEGORIES.items() for c, ss in cs.items() for s in ss}
    sp = sp[[cat.get(s) != "control" for s in sp.index]]
    idm = [cat.get(s) == "id" for s in sp.index]
    return dict(iou=sp.iou.mean(), miss=sp.miss.mean(), iou_id=sp.iou[idm].mean(), iou_ood=sp.iou[[not v for v in idm]].mean())

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--ckpt"); ap.add_argument("--src", default="src7")
    ap.add_argument("--signal", default="log_power"); ap.add_argument("--window", default="tukey")
    ap.add_argument("--dataset", default="gastronorm_plastic_cardboard")
    ap.add_argument("--kwargs", default="{}", help="Boombox kwargs as json")
    a = ap.parse_args()
    if (ra := run_args(a.run)) is not None: a.signal, a.window, a.src = ra.signal, ra.window, "src8"
    sys.path.insert(0, str(ROOT / a.src))
    X, y, labels, info = eval_data(a.dataset, a.signal, a.window, laser_center=bool(getattr(ra, "laser_center", 0)) if ra else False)
    ckpt = a.ckpt or sorted((ROOT / "runs" / a.run / "checkpoints").glob("ep*-rank0.pt"), key=lambda p: int(p.name[2:].split("-")[0]))[-1]
    model = load_model(a.src, a.run, ckpt, info, json.loads(a.kwargs))
    box = box_index(labels, info)
    res = {k: score(predict(model, X, p, box), y, labels) for k, p in perturbations(X.shape[1], X.shape[2]).items()}
    df = pd.DataFrame(res).T
    print(f"\n{a.run} ({Path(ckpt).name}), {len(X)} eval samples\n" + df.round(3).to_string())
    df.to_csv(ROOT / "runs" / a.run / "probe.csv")
