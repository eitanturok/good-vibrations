"""flowmask: a conditional flow-matching mask generator on top of a FROZEN transformer3 encoder (default runs/base300-v1, its last checkpoint).
per-pixel losses average over position uncertainty -> blurred masks; a generator samples one sharp mask instead.
  .venv/bin/python src9/flowmask.py --run-name flowmask-v1
1. the base model (eval mode, no aug, no token dropout) -> u (latent, 512) and its sigmoid mask (the prior) for every train / eval sample, once
   (cached in runs/<run-name>/cache.pt)
2. a small UNet on 64x64: input [x_t, 2 prior - 1 (--prior 1)], time t (sinusoidal) + u -> FiLM (scale, shift) in every res block -> v
   rectified flow: x1 = 2 mask - 1, x0 ~ N(0, 1), x_t = (1 - t) x0 + t x1, loss = |v - (x1 - x0)|^2; EMA 0.999 weights are evaluated
3. eval: Euler --steps from t = 0 (seeded noise, fixed per eval sample) -> 8 samples; flow1 = sample 0 > 0, flow8 = mean of the 8 binarized samples
   (prob map; at THR 0.08 = their union), std = per-pixel std. iou08 / miss08 / obj_recall / biou / area_ratio for base (the prior), flow1, flow8,
   pooled per sample over EvalSummary.ID / OOD categories, and per <box>/cat/<category> (mean over its splits) -> metrics.json, samples.png"""
import argparse, inspect, json, math, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from composer.utils.reproducibility import seed_all

from dataset import CATEGORIES, build_dataset
from callbacks import EvalSummary
from arch.transformer3 import Transformer3
from utils import iou08, miss08, obj_recall, biou, area_ratio

M = dict(iou08=iou08, miss08=miss08, obj_recall=obj_recall, biou=biou, area_ratio=area_ratio)

#*** 1 precompute u + prior with the frozen base model

@torch.no_grad()
def encode(model, loader, limit):
    """-> dict u (N,512), prior (N,64,64) sigmoid, mask (N,64,64), objs (N,K,64,64), all on the cpu"""
    out = {k: [] for k in ("u", "prior", "mask", "objs")}
    for i, b in enumerate(loader.dataloader):
        if limit and i >= limit: break
        with torch.autocast("cuda", torch.bfloat16): o = model(dict(fft=b["fft"].cuda(non_blocking=True)))
        out["u"].append(o["latent"].float().cpu()); out["prior"].append(o["mask_pred"].float().cpu())
        out["mask"].append(b["mask_true"].float()); out["objs"].append(b["objs"].bool())
    return {k: torch.cat(v) for k, v in out.items()}

def precompute(base, limit):
    a = json.loads(Path(f"runs/{base}/args.json").read_text())
    _, eval_dl, train_eval, info = build_dataset(a["window"], eval_batch_size=a["eval_batch_size"], height=a["height"], width=a["width"])
    model = Transformer3(info, **{k: a[k] for k in inspect.signature(Transformer3).parameters if k in a})
    model.load_state_dict(torch.load(f"runs/{base}/checkpoints/latest-rank0.pt", map_location="cpu", weights_only=False)["state"]["model"])
    model.cuda().eval()
    train = encode(model, train_eval.dataloader, limit)
    del train["objs"]
    return dict(train=train, eval={e.label[5:]: encode(model, e.dataloader, limit) for e in eval_dl})  # "<box>/<split>" -> dict

#*** 2 the generator

class ResBlock(nn.Module):
    """GroupNorm -> SiLU -> conv, FiLM (1 + scale, shift) from the embedding, GroupNorm -> SiLU -> conv, + skip"""
    def __init__(self, c_in, c_out, emb):
        super().__init__()
        self.n1, self.c1, self.film = nn.GroupNorm(8, c_in), nn.Conv2d(c_in, c_out, 3, padding=1), nn.Linear(emb, 2 * c_out)
        self.n2, self.c2 = nn.GroupNorm(8, c_out), nn.Conv2d(c_out, c_out, 3, padding=1)
        self.skip = nn.Conv2d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x, e):
        h = self.c1(F.silu(self.n1(x)))
        s, b = self.film(e)[:, :, None, None].chunk(2, 1)
        return self.skip(x) + self.c2(F.silu(self.n2(h) * (1 + s) + b))

def t_embed(t, dim):
    """(B,) in [0,1] -> (B,dim) sinusoidal"""
    f = (1000 * t[:, None]) * torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2))
    return torch.cat([f.sin(), f.cos()], -1)

class UNet(nn.Module):
    """(B,c_in,64,64), t (B,), u (B,latent) -> v (B,64,64): 2 res blocks per level at 64/32/16/8 px (chans), 2 in the middle, skip concat on the way up"""
    def __init__(self, c_in, latent=512, chans=(48, 96, 192, 192), emb=256):
        super().__init__()
        self.t_mlp, self.u_proj = nn.Sequential(nn.Linear(128, emb), nn.SiLU(), nn.Linear(emb, emb)), nn.Linear(latent, emb)
        self.inp = nn.Conv2d(c_in, chans[0], 3, padding=1)
        prev, self.downs = chans[0], nn.ModuleList()
        for c in chans: self.downs.append(nn.ModuleList([ResBlock(prev, c, emb), ResBlock(c, c, emb)])); prev = c
        self.mid = nn.ModuleList([ResBlock(prev, prev, emb), ResBlock(prev, prev, emb)])
        self.ups = nn.ModuleList()
        for c in reversed(chans): self.ups.append(nn.ModuleList([ResBlock(prev + c, c, emb), ResBlock(c, c, emb)])); prev = c
        self.out_norm, self.out = nn.GroupNorm(8, prev), nn.Conv2d(prev, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, x, t, u):
        e = F.silu(self.t_mlp(t_embed(t, 128)) + self.u_proj(u))
        x, skips = self.inp(x), []
        for i, (b1, b2) in enumerate(self.downs):
            x = b2(b1(x, e), e); skips.append(x)
            if i < len(self.downs) - 1: x = F.avg_pool2d(x, 2)
        for b in self.mid: x = b(x, e)
        for i, (b1, b2) in enumerate(self.ups):
            x = b2(b1(torch.cat([x, skips.pop()], 1), e), e)
            if i < len(self.ups) - 1: x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.out(F.silu(self.out_norm(x)))[:, 0]

def inputs(x, prior, use_prior): return torch.stack([x, 2 * prior - 1], 1) if use_prior else x[:, None]

@torch.no_grad()
def sample(net, u, prior, noise, steps, use_prior):
    """Euler from t = 0 (noise) to 1 -> (B,64,64) in ~[-1,1]"""
    x, dt = noise, 1 / steps
    for i in range(steps):
        with torch.autocast("cuda", torch.bfloat16): v = net(inputs(x, prior, use_prior), torch.full((len(x),), i * dt, device=x.device), u)
        x = x + dt * v.float()
    return x

#*** 3 eval

def evaluate(net, ev, steps, use_prior, n=8, seed=0, bs=32):
    """-> per split {base, flow1, flow8: {metric: (N,) per sample}}, and the maps for the png: {split: (prior, samples (N,n,64,64))}"""
    per, maps = {}, {}
    g = torch.Generator().manual_seed(seed)
    for s, d in ev.items():                                                          # fixed split order + one cpu generator = the same noise every run
        noise = torch.randn(len(d["u"]), n, 64, 64, generator=g)
        xs = []
        for i in range(0, len(d["u"]), bs):
            u, p, z = d["u"][i:i + bs].cuda(), d["prior"][i:i + bs].cuda(), noise[i:i + bs].cuda()
            xs.append(sample(net, u.repeat_interleave(n, 0), p.repeat_interleave(n, 0), z.flatten(0, 1), steps, use_prior).view(-1, n, 64, 64).cpu())
        xs = torch.cat(xs) if xs else torch.zeros(0, n, 64, 64)
        b = (xs > 0).float()
        batch = dict(mask_true=d["mask"], objs=d["objs"].float())
        preds = dict(base=d["prior"], flow1=b[:, 0], flow8=b.mean(1))
        per[s] = {k: {m: f(dict(mask_pred=v), batch) for m, f in M.items()} for k, v in preds.items()}
        maps[s] = (d["prior"], b)
    return per, maps

def summarize(per):
    """per-sample pooled id / ood (EvalSummary.ID / OOD categories), and <box>/cat/<category> = mean over its splits of the split means"""
    cat = {f"{b}/{s}": c for b, cs in CATEGORIES.items() for c, ss in cs.items() for s in ss}
    res = {}
    for k in next(iter(per.values())):
        r = {}
        for g, cs in (("id", EvalSummary.ID), ("ood", EvalSummary.OOD)):
            r[g] = {m: torch.cat([v[k][m] for s, v in per.items() if cat.get(s) in cs]).mean().item() for m in M}
        for box, cs in CATEGORIES.items():
            for c, ss in cs.items():
                if vs := [per[f"{box}/{s}"][k] for s in ss if f"{box}/{s}" in per]:
                    r[f"{box}/cat/{c}"] = {m: float(np.nanmean([v[m].mean().item() for v in vs])) for m in M}
        res[k] = r
    return res

def grid(ev, maps, path, n=16):
    """n eval samples spread over every split: gt | base | flow sample | 8-sample mean | std"""
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    items = [(s, i) for s, d in ev.items() for i in range(len(d["u"]))]
    items = [items[j] for j in np.linspace(0, len(items) - 1, n).round().astype(int)]
    fig, ax = plt.subplots(n, 5, figsize=(10, 2 * n))
    for r, (s, i) in enumerate(items):
        prior, b = maps[s]
        for c, (im, t) in enumerate([(ev[s]["mask"][i], "gt"), (prior[i], "base"), (b[i, 0], "flow sample"), (b[i].mean(0), "8-sample mean"), (b[i].std(0, unbiased=False), "std")]):
            ax[r, c].imshow(im, cmap="gray", vmin=0, vmax=0.5 if t == "std" else 1); ax[r, c].set_xticks([]); ax[r, c].set_yticks([])
            if r == 0: ax[r, c].set_title(t)
        ax[r, 0].set_ylabel(f"{s}\n#{i}", fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)

#*** 4 main

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--base", default="base300-v1", help="the frozen transformer3 run (its latest checkpoint)")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--steps", type=int, default=25, help="Euler sampling steps")
    p.add_argument("--prior", type=int, default=1, help="1 = the base model's sigmoid mask is a 2nd input channel")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0, help="smoke test: only the first <limit> batches of every loader (0 = all)")
    p.add_argument("--wandb", type=int, default=1)
    args = p.parse_args()
    seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    out = Path(f"runs/{args.run_name}"); out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=1))

    cache = out / "cache.pt"
    if not cache.exists(): torch.save(precompute(args.base, args.limit), cache)
    data = torch.load(cache, weights_only=False)
    tr, ev = {k: v.cuda() for k, v in data["train"].items()}, data["eval"]
    N = len(tr["u"])
    print(f"train {N} samples, eval {sum(len(d['u']) for d in ev.values())} samples in {len(ev)} splits")

    net = UNet(2 if args.prior else 1).cuda()
    ema = torch.optim.swa_utils.AveragedModel(net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    print(f"unet: {sum(q.numel() for q in net.parameters()):,} parameters")
    opt = torch.optim.AdamW(net.parameters(), args.lr, weight_decay=0.01)
    steps_per_ep = max(1, N // args.batch_size)
    total = args.epochs * steps_per_ep
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, i / 500) * 0.5 * (1 + math.cos(math.pi * min(i / total, 1))))  # 500 warmup steps, cosine
    if args.wandb:
        import wandb
        wandb.init(project="better-tsa", group="src9", name=args.run_name, id=args.run_name, resume="allow", config=vars(args))
    t0 = time.time()
    for ep in range(args.epochs):
        net.train()
        perm, losses = torch.randperm(N, device="cuda"), []
        for j in range(steps_per_ep):
            idx = perm[j * args.batch_size:(j + 1) * args.batch_size]
            x1, u, prior = 2 * tr["mask"][idx] - 1, tr["u"][idx], tr["prior"][idx]
            x0, t = torch.randn_like(x1), torch.rand(len(x1), device="cuda")
            xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
            with torch.autocast("cuda", torch.bfloat16): v = net(inputs(xt, prior, args.prior), t, u)
            loss = F.mse_loss(v.float(), x1 - x0)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); ema.update_parameters(net)
            losses.append(loss.detach())
        l = torch.stack(losses).mean().item()
        if ep % 10 == 0 or ep == args.epochs - 1: print(f"ep {ep} loss {l:.4f} lr {sched.get_last_lr()[0]:.2e} {time.time() - t0:.0f}s", flush=True)
        if args.wandb: wandb.log({"loss/train/total": l, "lr": sched.get_last_lr()[0], "epoch": ep})
    torch.save(dict(net=net.state_dict(), ema=ema.module.state_dict(), args=vars(args)), out / "flowmask.pt")

    ema.module.eval()
    per, maps = evaluate(ema.module, ev, args.steps, args.prior)
    res = summarize(per)
    res["std_mean"] = {"id_ood": float(np.mean([maps[s][1].std(1, unbiased=False).mean().item() for s in maps if len(maps[s][1])]))}
    (out / "metrics.json").write_text(json.dumps(res, indent=1))
    grid(ev, maps, out / "samples.png")
    fmt = lambda d: "  ".join(f"{k} {v:.3f}" for k, v in d.items())
    for k in ("base", "flow1", "flow8"):
        for g in ("id", "ood"): print(f"{k:5s} {g:3s} {fmt(res[k][g])}")
    if args.wandb:
        wandb.log({f"metrics/eval/{k}/{g}/{m}": v for k in ("base", "flow1", "flow8") for g, d in res[k].items() for m, v in d.items()})
        wandb.log({"samples": wandb.Image(str(out / "samples.png"))}); wandb.finish()

if __name__ == "__main__":
    main()
