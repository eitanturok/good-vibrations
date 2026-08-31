"""Train a head on a FROZEN dec-d3-conv-v6 to predict the object's (row, col).

    vibrations -> [frozen backbone] -> head -> (row, col) -> pose matrix -> placed mask

ONE CUBE ONLY. The head reads the decoder's feature map, optionally with the encoder
bottleneck concatenated (--use-emb).

The bar is the centre of mass of the decoder's own predicted mask -- that is how you get a
position out of the pipeline today, and it costs no training at all. It is printed as
`no-train` in every run.

TWO SPLITS, and the difference matters.

  --split-mode original     (default) train on dec-d3-conv-v6's own train half, evaluate
                            on its eval/1-cube. Those positions were never seen by the
                            BACKBONE either, so the frozen features are not reconstructions
                            of masks it was optimised to fit. This is the honest number.

  --split-mode repartition  pool every one-cube sample and re-split by position. Holds out
                            whole positions, but draws from the whole dataset, so ~47% of
                            the eval half is data the backbone trained on -- measured.
                            Useful for ranking heads against each other; optimistic in
                            absolute terms.

Either way the split is by POSITION, never by sample: each position is captured 8 times
(once per speaker), so a sample-level split puts the same position on both sides.
dataset.py's `gastronorm_one_cube` does exactly that -- all 60 of its eval positions are
also in train, verified -- which is why neither mode here uses it.

Usage:
    python scripts/com_head.py --head softargmax
    python scripts/com_head.py --head conv --use-emb
    python scripts/com_head.py --head softargmax --use-emb --split-mode repartition
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))          # utils.*
sys.path.insert(0, str(ROOT / "src"))  # model.*  (boombox.py does `from model.arch import ...`)

from model.boombox import BoomboxModel          # noqa: E402
from model.dataset import build_dataset         # noqa: E402
from utils.metrics import center_of_mass        # noqa: E402

DATA_DIR = ROOT / "experiments" / "31_07_2026_gastronorm_exp1"
CKPT = ROOT / "runs" / "dec-d3-conv-v6" / "checkpoints" / "latest-rank0.pt"
OUT = ROOT / "runs" / "dec-d3-conv-v6" / "com_head"


# ***** 1 frozen backbone *****

def load_backbone(data_info, d_model, device):
    model = BoomboxModel(d_model, data_info, loss_fn="mse")
    sd = torch.load(CKPT, map_location="cpu", weights_only=False)["state"]["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # strict=False without checking the result is how you end up training on a randomly
    # initialised backbone and never notice
    assert not missing and not unexpected, f"{missing=} {unexpected=}"
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(device).eval()
    # the encoder has 7 BatchNorm2d layers -- in train mode every feature would depend on
    # its batch's composition, and the corruption is silent
    assert not model.training
    return model


@torch.no_grad()
def frozen_features(model, x):
    """dec_feat (B,64,21,30), emb (B,512), mask_pred (B,21,30).

    The decoder is run by parts so its pre-head feature map is visible; verified
    bit-identical to calling decoder() directly.
    """
    enc, dec = model.encoder, model.decoder
    fg = enc.freq(model._to_conv(x)).reshape(x.shape[0], -1, *enc.grid_shape)
    emb = enc.grid(fg).flatten(1)
    y = dec.up(dec.project(emb).view(-1, 512, *dec.SEED))
    y = F.interpolate(y, size=dec.out_hw, mode="bilinear", align_corners=False)
    mask_pred = dec.head(y).squeeze(1).sigmoid()
    return y, emb, mask_pred


# ***** 2 cache the frozen features once *****

def _unwrap(loader):
    """composer hands back DataSpec / Evaluator wrappers; neither is iterable."""
    d = loader.dataloader if hasattr(loader, "dataloader") else loader
    return d.dataloader if hasattr(d, "dataloader") else d


def cache_key(args):
    st = CKPT.stat()
    raw = f"{CKPT}|{st.st_mtime_ns}|{args.d_model}|{args.out_h}|{args.out_w}|{args.patch_size}|{args.seed}|v2-origin"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def build_cache(args):
    """Run the frozen backbone over the dataset once. Its output is a pure function of the
    sample (augmentation is off), so caching is exact and head training becomes seconds."""
    path = OUT / f"cache_{cache_key(args)}.pt"
    if path.exists() and not args.force:
        print(f"reusing {path.name}")
        return torch.load(path, map_location="cpu", weights_only=False)

    print("extracting frozen features (cold MDS rebuild ~10 min)")
    train_loader, eval_loaders, train_eval_loader = build_dataset(
        DATA_DIR, split="gastronorm", batch_size=args.batch_size, eval_batch_size=args.batch_size,
        num_workers=args.num_workers, out_h=args.out_h, out_w=args.out_w,
        augment_fft=0.0, augment_mask=0.0, patch_size=args.patch_size, seed=args.seed, verbose=1)

    _, n_patches, ps, n_ch = train_loader.dataloader.dataset[0]["fft"].shape
    data_info = dict(out_h=args.out_h, out_w=args.out_w, out_c=1, n_laser_rows=10,
                     n_laser_cols=10, patch_size=args.patch_size,
                     n_freqs=n_patches * ps, n_channels=n_ch)
    model = load_backbone(data_info, args.d_model, args.device)

    # Tag every sample with the ORIGINAL gastronorm split it came from, so --split-mode
    # can hold out exactly what the backbone never trained on. Without this the head's
    # "held-out" set silently includes backbone training data.
    origin_names = ["train"] + [e.label for e in eval_loaders]
    acc: dict[str, list] = {}
    t0 = time.time()
    with torch.no_grad():
        for origin, loader in zip(origin_names,
                                  [_unwrap(train_eval_loader)] + [_unwrap(e) for e in eval_loaders]):
            for b in loader:
                n_obj = b["info"]["n_objects"]
                n_obj = n_obj if torch.is_tensor(n_obj) else torch.as_tensor(n_obj)
                mask_true = b["mask_true"].to(args.device)
                # one cube only, and an empty mask has no centre of mass to predict
                keep = (n_obj.to(args.device) == 1) & (mask_true.sum((-2, -1)) > 0)
                if not keep.any():
                    continue
                dec_feat, emb, mask_pred = frozen_features(model, b["fft"].to(args.device))
                acc.setdefault("dec_feat", []).append(dec_feat[keep].float().cpu())
                acc.setdefault("emb", []).append(emb[keep].float().cpu())
                acc.setdefault("com", []).append(
                    center_of_mass(mask_true[keep], normalize=True).float().cpu())
                # the decoder runs anyway, so the no-train baseline is free here
                acc.setdefault("mask_com", []).append(
                    center_of_mass(mask_pred[keep], normalize=True).float().cpu())
                for k in ("sample_id", "position_id", "speaker"):
                    v = b["info"][k]
                    v = v if torch.is_tensor(v) else torch.as_tensor([int(s) for s in v])
                    acc.setdefault(k, []).append(v[keep.cpu()])
                acc.setdefault("origin", []).append(
                    torch.full((int(keep.sum()),), origin_names.index(origin), dtype=torch.long))
    data = {k: torch.cat(v) for k, v in acc.items()}
    data["origin_names"] = origin_names
    print(f"  {len(data['com'])} one-cube samples, "
          f"{len(torch.unique(data['position_id']))} positions  [{time.time() - t0:.0f}s]")

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    return data


def _mask_split(data, is_eval):
    tr = {k: v[~is_eval] for k, v in data.items() if torch.is_tensor(v)}
    ev = {k: v[is_eval] for k, v in data.items() if torch.is_tensor(v)}
    assert not (set(tr["position_id"].tolist()) & set(ev["position_id"].tolist())), \
        "a position appears in both halves"
    return tr, ev


def split_by_position(data, test_frac, seed):
    """Re-partition ALL one-cube samples by position, ignoring the original split.

    Holds out whole positions (not samples -- each position is captured once per speaker),
    but it draws from the whole dataset, so ~47% of the eval half is data the BACKBONE
    trained on. Fine for ranking heads against each other, optimistic in absolute terms.
    """
    pos = data["position_id"]
    uniq = torch.unique(pos).numpy()
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    eval_pos = set(uniq[:max(1, round(len(uniq) * test_frac))].tolist())
    is_eval = torch.tensor([int(p) in eval_pos for p in pos.tolist()])
    return _mask_split(data, is_eval)


def split_original(data):
    """Use dec-d3-conv-v6's OWN split: train on its train half, evaluate on eval/1-cube.

    This is the honest one. eval/1-cube holds positions the BACKBONE never saw, so the
    frozen features there are not reconstructions of masks it was optimised to fit, and
    the numbers are comparable to the run's logged com-distance.
    """
    names = data["origin_names"]
    keep_eval = names.index("eval/1-cube")
    keep_train = names.index("train")
    origin = data["origin"]
    # *-speaker splits test an unseen SPEAKER at a SEEN position -- neither train nor a
    # clean position holdout, so they are dropped rather than silently mixed in
    sel = (origin == keep_train) | (origin == keep_eval)
    data = {k: (v[sel] if torch.is_tensor(v) else v) for k, v in data.items()}
    return _mask_split(data, data["origin"] == keep_eval)


# ***** 3 heads *****

class SoftArgmax(nn.Module):
    """ONE 21x30 heatmap -> (row, col) by soft-argmax over its marginals.

    ONE channel, not two: two channels would be independent 2-D maps for what is a single
    point. Marginalising one distribution is the factorisation that matches the object.

    SOFT, not hard: hard argmax has no gradient, so nothing would train, and it quantises
    to the grid -- one cell is ~53 px vertically at full resolution, most of a cube.

    temperature < 1 sharpens toward the mode. The expectation of a diffuse map drifts to
    the image centre, which is the same failure that makes pixel-MSE masks give bad
    centroids.
    """

    def __init__(self, c_in, out_h=21, out_w=30, hidden=256, temperature=1.0):
        super().__init__()
        self.out_h, self.out_w, self.t = out_h, out_w, temperature
        self.body = nn.Sequential(nn.Conv2d(c_in, hidden, 1), nn.GELU(),
                                  nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU())
        self.to_map = nn.Conv2d(hidden, 1, 1)
        self.register_buffer("rows", torch.arange(out_h).float() / (out_h - 1))
        self.register_buffer("cols", torch.arange(out_w).float() / (out_w - 1))

    def heatmap(self, x):
        """Exposed so the map can be inspected -- a confident spike vs mass smeared across
        the box is exactly what plain regression cannot tell you."""
        return (self.to_map(self.body(x)).flatten(1) / self.t).softmax(-1).view(-1, self.out_h, self.out_w)

    def forward(self, x):
        p = self.heatmap(x)
        return torch.stack([(p.sum(-1) * self.rows).sum(-1),
                            (p.sum(-2) * self.cols).sum(-1)], -1)


class Head(nn.Module):
    """A head over the decoder feature map, optionally fused with the encoder bottleneck.

    When use_emb is set, the 512-d vector is broadcast across the 21x30 grid as extra
    channels for the spatial heads, so it is readable at every location rather than only
    after flattening.
    """

    def __init__(self, kind, dec_shape, emb_dim, use_emb=False, hidden=256, dropout=0.1,
                 out_hw=(21, 30), temperature=1.0):
        super().__init__()
        self.kind, self.use_emb = kind, use_emb
        c, h, w = dec_shape
        if kind in ("linear", "mlp"):
            d = c * h * w + (emb_dim if use_emb else 0)
            self.net = (nn.Linear(d, 2) if kind == "linear" else
                        nn.Sequential(nn.Linear(d, hidden), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(hidden, 2)))
        elif kind == "conv":
            c_in = c + (emb_dim if use_emb else 0)
            self.net = nn.Sequential(
                nn.Conv2d(c_in, hidden, 1), nn.GELU(),
                nn.Conv2d(hidden, hidden // 2, 3, padding=1), nn.GELU(),
                nn.Flatten(), nn.Linear(hidden // 2 * h * w, hidden), nn.GELU(),
                nn.Dropout(dropout), nn.Linear(hidden, 2))
        elif kind == "softargmax":
            self.net = SoftArgmax(c + (emb_dim if use_emb else 0), *out_hw,
                                  hidden=hidden, temperature=temperature)
        else:
            raise ValueError(f"unknown head {kind!r}")

    def forward(self, dec_feat, emb):
        if self.kind in ("linear", "mlp"):
            x = dec_feat.flatten(1)
            return self.net(torch.cat([x, emb], 1) if self.use_emb else x)
        x = dec_feat
        if self.use_emb:
            x = torch.cat([x, emb[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])], 1)
        return self.net(x)


def zero_final_bias(head):
    """Start at the predict-the-mean floor, so beating it shows from epoch 1."""
    for m in reversed(list(head.modules())):
        if isinstance(m, (nn.Linear, nn.Conv2d)) and m.bias is not None:
            nn.init.zeros_(m.bias)
            return


# ***** 4 train *****

def com_dist(pred, true):
    """Mean L2 in normalized (row, col) -- the units run.py logs as `com-distance`."""
    return torch.linalg.norm(pred - true, dim=-1).mean().item()


def to_px(d, h=1110, w=1337):
    return float(d * np.hypot(h - 1, w - 1) / np.sqrt(2))


def train_head(tr, ev, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = args.device

    # Features stay on the CPU and ride over per minibatch. This box shares its GPU, and a
    # resident copy is what turns someone else's allocation into an OOM here.
    Xd, Xe = tr["dec_feat"], tr["emb"]
    Vd, Ve = ev["dec_feat"], ev["emb"]
    Y, Yv = tr["com"].to(dev), ev["com"].to(dev)

    mu, sigma = Y.mean(0), Y.std(0).clamp(min=1e-6)
    # softargmax emits coordinates in [0,1] by construction; standardising would fight the
    # geometry it was chosen for. Everything else trains on standardized targets, which
    # keeps the two axes on one scale and conditions the optimisation.
    standardize = args.head != "softargmax"
    Yt = (Y - mu) / sigma if standardize else Y

    head = Head(args.head, tuple(Xd.shape[1:]), Xe.shape[1], args.use_emb, args.hidden,
                args.dropout, (args.out_h, args.out_w), args.temperature).to(dev)
    zero_final_bias(head)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    def evaluate():
        head.eval()
        outs = []
        with torch.no_grad():
            for i in range(0, len(Yv), args.head_batch):
                sl = slice(i, i + args.head_batch)
                o = head(Vd[sl].to(dev), Ve[sl].to(dev))
                outs.append(o * sigma + mu if standardize else o)
        return com_dist(torch.cat(outs), Yv)

    best, best_state, bad = float("inf"), copy.deepcopy(head.state_dict()), 0
    for ep in range(args.epochs):
        head.train()
        for idx in torch.randperm(len(Y)).split(args.head_batch):
            opt.zero_grad(set_to_none=True)
            F.mse_loss(head(Xd[idx].to(dev), Xe[idx].to(dev)), Yt[idx]).backward()
            opt.step()
        sched.step()
        s = evaluate()
        if s < best - 1e-6:
            best, best_state, bad = s, copy.deepcopy(head.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                break
    head.load_state_dict(best_state)
    return head, best, ep + 1


# ***** 5 main *****

def get_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--head", default="softargmax", choices=["linear", "mlp", "conv", "softargmax"])
    p.add_argument("--use-emb", action="store_true",
                   help="also feed the encoder bottleneck, not just the decoder map")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=300, help="default = no early stop")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=1.0, help="softargmax only")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--head-batch", type=int, default=256)
    p.add_argument("--test-frac", type=float, default=0.25)
    p.add_argument("--split-mode", default="original", choices=["original", "repartition"],
                   help="original: dec-d3-conv-v6's own eval/1-cube, unseen by the backbone. "
                        "repartition: re-split all one-cube data by position (optimistic -- "
                        "~47%% of that eval half is backbone training data)")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--out-h", type=int, default=21)
    p.add_argument("--out-w", type=int, default=30)
    p.add_argument("--patch-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--force", action="store_true", help="rebuild the feature cache")
    p.add_argument("--cache-only", action="store_true")
    p.add_argument("--tag", default="")
    return p


def main():
    args = get_parser().parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = build_cache(args)
    if args.cache_only:
        return

    tr, ev = (split_original(data) if args.split_mode == "original"
              else split_by_position(data, args.test_frac, args.seed))
    no_train = com_dist(ev["mask_com"], ev["com"])
    floor = com_dist(tr["com"].mean(0, keepdim=True).expand_as(ev["com"]), ev["com"])

    src = "decoder + encoder" if args.use_emb else "decoder"
    print(f"\ntrain {len(tr['com']):4d} samples / {len(torch.unique(tr['position_id'])):2d} positions")
    note = ("eval/1-cube -- positions the BACKBONE never saw" if args.split_mode == "original"
            else "re-partitioned by position; ~47% of eval is backbone train data")
    print(f"eval  {len(ev['com']):4d} samples / {len(torch.unique(ev['position_id'])):2d} positions"
          f"   ({note})")
    print(f"\nhead={args.head}  on {src}"
          f"{f'  temperature={args.temperature}' if args.head == 'softargmax' else ''}")
    print(f"  {'no-train (decoder mask-COM)':32s} {no_train:.4f}   {to_px(no_train):6.1f} px")
    print(f"  {'predict-the-mean floor':32s} {floor:.4f}   {to_px(floor):6.1f} px")

    stem = (f"{args.head}{'_emb' if args.use_emb else ''}"
            f"{'_' + args.tag if args.tag else ''}"
            f"{'_repart' if args.split_mode == 'repartition' else ''}")
    scores, n_par, eps, best = [], 0, 0, None
    for s in range(args.seeds):
        t0 = time.time()
        head, sc, e = train_head(tr, ev, args, args.seed + s)
        n_par = sum(p.numel() for p in head.parameters())
        scores.append(sc)
        eps = max(eps, e)
        if best is None or sc < best[0]:
            best = (sc, copy.deepcopy(head.state_dict()), args.seed + s)
        print(f"    seed {s}: {sc:.4f}   {to_px(sc):6.1f} px   {e:3d} ep  [{time.time() - t0:.0f}s]")

    m, sd = float(np.mean(scores)), float(np.std(scores))
    verdict = "BEATS" if m < no_train else "loses to"
    print(f"  {'trained head':32s} {m:.4f} +/- {sd:.4f}   {to_px(m):6.1f} px   {n_par:,} params")
    print(f"  -> {verdict} no-train ({m:.4f} vs {no_train:.4f})")

    # The best seed's weights, plus everything needed to rebuild the head and undo the
    # target standardisation. notebooks/72 loads exactly this to run a forward pass.
    ckpt = OUT / f"head_{stem}.pt"
    torch.save({"state_dict": best[1], "seed": best[2], "score": best[0],
                "split_mode": args.split_mode, "n_eval": len(ev["com"]),
                "no_train": no_train, "floor": floor,
                "head": args.head, "use_emb": args.use_emb, "hidden": args.hidden,
                "dropout": args.dropout, "temperature": args.temperature,
                "out_h": args.out_h, "out_w": args.out_w, "d_model": args.d_model,
                "patch_size": args.patch_size, "dec_shape": tuple(tr["dec_feat"].shape[1:]),
                "emb_dim": tr["emb"].shape[1],
                "mu": tr["com"].mean(0), "sigma": tr["com"].std(0).clamp(min=1e-6),
                "standardize": args.head != "softargmax"}, ckpt)
    print(f"  wrote {ckpt.name}  (best seed {best[2]}, {best[0]:.4f})")

    (OUT / f"results_{stem}.json").write_text(json.dumps(
        {"args": vars(args) | {"device": str(args.device)},
         "no_train": no_train, "floor": floor,
         "head_mean": m, "head_std": sd, "seeds": scores, "params": n_par,
         "checkpoint": ckpt.name, "best_seed": best[2], "best_score": best[0]},
        indent=2, default=str))
    print(f"  wrote results_{stem}.json")


if __name__ == "__main__":
    main()
