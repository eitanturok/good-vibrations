import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from composer import Callback
from composer.core import Time, TimeUnit

from arch import iou, biou, n_objects_err, box_acc, com_err, mask_area, outline, ray_angles
sys.path.append(str(Path(__file__).resolve().parents[1]))  # repo root, for viz's scoring in utils/metrics.py
from utils.metrics import localization, contour_f

MAX_WANDB_IMAGES = 108  # hard limit: wandb drops anything past 108 images in one log_images call

def render(pred, true, caption, aspect, panel_w=320):
    """predicted | true side by side under a caption; pred/true are (H,W) masks or (H,W,3) rgb in [0,1].
    The dataset squashes every box to the same out_h x out_w grid, so panels are drawn at the box's real aspect (w/h)."""
    top, panel_h, font = 56, round(panel_w / aspect), ImageFont.load_default(size=13)
    canvas = Image.new("RGB", (2 * panel_w + 4, top + panel_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((canvas.width // 2, 4), caption, fill="black", font=font, anchor="ma", align="center")
    for x, arr, label in [(0, pred, "predicted"), (panel_w + 4, true, "true")]:
        canvas.paste(Image.fromarray((arr * 255).astype(np.uint8)).resize((panel_w, panel_h), Image.NEAREST), (x, top))
        draw.text((x + panel_w // 2, top - 16), label, fill="black", font=font, anchor="mt")
    return np.array(canvas)

def render_polar(score, pred, true, rc, rays, keep, coms, caption, aspect, panel_w=320):
    """polar head: center score | drawn mask + every peak's rays and outline | true mask, under a caption. score/pred/true (H,W) in [0,1];
    rc (N,2) peak pixel centers and rays (N,n) ray lengths, px of the H x W grid; keep (N,) which peaks are objects; coms (K,2) true coms
    in [0,1]. Kept peaks red, dropped ones gray; predicted com = the peak (+), true com = green x"""
    H, W = pred.shape
    top, panel_h, font = 56, round(panel_w / aspect), ImageFont.load_default(size=13)
    sx, sy = panel_w / W, panel_h / H
    canvas = Image.new("RGB", (3 * panel_w + 8, top + panel_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((canvas.width // 2, 4), caption, fill="black", font=font, anchor="ma", align="center")
    th = ray_angles(rays.shape[1]).numpy()
    for j, (arr, label) in enumerate([(score / max(score.max(), 1e-6), f"center score (max {score.max():.2f})"), (pred, "drawn mask + rays"), (true, "true")]):
        x0 = j * (panel_w + 4)
        canvas.paste(Image.fromarray((arr * 255).astype(np.uint8)).resize((panel_w, panel_h), Image.NEAREST), (x0, top))
        draw.text((x0 + panel_w // 2, top - 16), label, fill="black", font=font, anchor="mt")
        for (cy, cx), r, k in zip(rc, rays, keep):
            if j == 2: continue
            color = (220, 40, 40) if k else (150, 150, 150)
            px, py = x0 + cx * sx, top + cy * sy
            ends = [(px + ri * np.cos(t) * sx, py + ri * np.sin(t) * sy) for ri, t in zip(r, th)]  # ray j: row += sin, col += cos
            if j == 1:
                for e in ends: draw.line([(px, py), e], fill=color, width=1)
                draw.polygon(ends, outline=color, width=2)
            draw.line([(px - 6, py), (px + 6, py)], fill=color, width=2); draw.line([(px, py - 6), (px, py + 6)], fill=color, width=2)
        for cy, cx in coms:
            px, py = x0 + cx * panel_w, top + cy * panel_h
            draw.line([(px - 5, py - 5), (px + 5, py + 5)], fill=(40, 180, 40), width=2); draw.line([(px - 5, py + 5), (px + 5, py - 5)], fill=(40, 180, 40), width=2)
    return np.array(canvas)

# each callback runs on eval batches and, during training, on train batches: both every `interval` epochs. Train batches
# skip epoch 0, which the eval before training already covers -- they'd otherwise overwrite its train/ files.
def due(state, interval, train): return state.timestamp.epoch.value % interval == 0 and not (train and state.timestamp.epoch.value == 0)

class VisualizeSMask(Callback):
    """log the first `n_images` predicted vs true masks of each loader's first batch to wandb as SMask/<loader label>, every `interval`
    epochs (a multiple of the eval interval). Only a sample: every batch every eval was ~4GB per 3k-epoch run, and viz has every
    output anyway (OutputSaver)"""
    def __init__(self, interval="100ep", n_images=16): self.interval, self.n = Time.from_input(interval, TimeUnit.EPOCH).value, min(n_images, MAX_WANDB_IMAGES)

    def log(self, state, logger):
        b = state.batch
        pred = state.outputs["mask_pred"][:self.n].detach().float().cpu()  # float: numpy has no bf16
        true = b["mask_true"][:self.n].float().cpu()
        iou = torch.minimum(pred, true).flatten(1).sum(1) / torch.maximum(pred, true).flatten(1).sum(1).clamp_min(1e-6)
        images = [render(pred[i].numpy(), true[i].numpy(),
                         f"{b['position_id'][i]}-{b['speaker'][i]} iou={iou[i]:.2f}",
                         aspect=b["crop_w"][i].item() / b["crop_h"][i].item())
                  for i in range(len(pred))]
        logger.log_images(images, name=f"SMask/{state.dataloader_label}", channels_last=True, use_table=False)
        if "edge_logits" in state.outputs:
            pe, te = state.outputs["edge_logits"][:self.n].detach().float().sigmoid().cpu(), outline(true)
            images = [render(pe[i].numpy(), te[i].numpy(), f"{b['position_id'][i]}-{b['speaker'][i]} edge", aspect=b["crop_w"][i].item() / b["crop_h"][i].item())
                      for i in range(len(pe))]
            logger.log_images(images, name=f"Edge/{state.dataloader_label}", channels_last=True, use_table=False)
        if "peak_rays" in state.outputs:  # polar head
            o = {k: state.outputs[k][:self.n].detach().float().cpu() for k in ("center_logits", "centerness_logits", "peak_rc", "peak_rays", "peak_keep", "peak_scores") if k in state.outputs}
            score = o["center_logits"].sigmoid() * o["centerness_logits"].sigmoid() if "center_logits" in o else torch.zeros_like(pred)  # slots: no center map
            images = [render_polar(score[i].numpy(), pred[i].numpy(), true[i].numpy(), o["peak_rc"][i].numpy(), o["peak_rays"][i].numpy(), o["peak_keep"][i].numpy(),
                                   b["coms"][i][:b["n_objects"][i]].float().cpu().numpy(),
                                   f"{b['position_id'][i]}-{b['speaker'][i]} iou={iou[i]:.2f} peaks " + " ".join(f"{v:.2f}" for v in o["peak_scores"][i]),
                                   aspect=b["crop_w"][i].item() / b["crop_h"][i].item())
                      for i in range(len(pred))]
            logger.log_images(images, name=f"Polar/{state.dataloader_label}", channels_last=True, use_table=False)

    def after_forward(self, state, logger):
        if self.n and due(state, self.interval, train=True) and state.timestamp.batch_in_epoch.value == 0: self.log(state, logger)

    def eval_after_forward(self, state, logger):
        if self.n and due(state, self.interval, train=False) and state.eval_timestamp.batch.value == 0: self.log(state, logger)

def sample_metrics(out, batch):
    """every metric per sample, each (B,) and NaN where it doesn't apply: training's (arch.METRICS, which average over objects or
    drop rows before logging), the confidence scores below, and viz's localization + contour (utils.metrics). Also the crosshair
    geometry viz draws, from the same object matching as localization. None for rgb runs (no mask to score)."""
    pred, true = out["mask_pred"].detach().float(), batch["mask_true"].float()
    if pred.ndim != 3: return None, None
    obj, n, area = batch["object"], batch["n_objects"], mask_area(true)
    nan = torch.full_like(area, torch.nan)
    real = torch.arange(out["com"].shape[1], device=n.device) < n[:, None]   # (B,K) real object slots
    com = torch.full(real.shape, torch.nan, device=pred.device)
    com[real] = com_err(out["com"], batch["coms"], n)                          # (n_real,) -> (B,K), NaN padding
    m = dict(iou=iou(out, batch),
             boundary_iou=biou(pred, true),
             object_acc=(out["object_logits"].argmax(-1) == obj).float().where(obj < out["object_logits"].shape[-1], nan),  # unseen: NaN
             n_objects_err=n_objects_err(out, batch),
             box_acc=box_acc(out, batch),
             com_err=com.nanmean(1),                                           # mean over the sample's objects
             area_rel_err=((out["log_area"].expm1() - area).abs() / area).where(area > 0, nan))
    m.update({k: v for k, v in scores(pred, true).items() if k not in ("iou", "obj_px", "bg_px")})  # iou: same as above; px: not per sample
    loc, geom = localization(pred, true, geometry=True)
    m.update(localization_rel=loc["localization_rel"], localization_raw=loc["localization_raw"], contour=contour_f(pred, true))
    return {k: v.detach().float().cpu() for k, v in m.items()}, geom

HEADS = ("object_logits", "n_objects_logits", "com", "log_area", "box_logits")  # the aux heads' raw outputs, saved per sample
POLAR = ("peak_scores", "peak_rc", "peak_rays", "peak_keep")  # polar head: its peaks, saved per sample too

def polar_outline(out):
    """polar head outputs -> the saved angles + outlines: ray_angles (n,) radians, ray j points row += sin, col += cos; polygon (B,N,n,2)
    each peak's n outline points as (row,col) fractions of the box, like com (peak_rc and peak_rays are px of the out_h x out_w grid)"""
    rc, r = out["peak_rc"].detach().float().cpu(), out["peak_rays"].detach().float().cpu()  # (B,N,2), (B,N,n)
    th = ray_angles(r.shape[-1])
    H, W = out["mask_pred"].shape[-2:]
    poly = rc[:, :, None] + r[..., None] * torch.stack([th.sin(), th.cos()], -1)            # (B,N,1,2) + (B,N,n,1) * (n,2) -> (B,N,n,2) px
    return dict(ray_angles=th, polygon=poly / torch.tensor([H, W]))

class OutputSaver(Callback):
    """save each batch's predicted masks + aux head outputs (HEADS) + sample ids + per-sample metrics (sample_metrics) to
    runs/<run>/outputs_history/<loader label>/ep{epoch:04d}-ba{batch:06d}.pt, the files the viz dashboard reads.
    objects names object_logits' classes, so the predicted class is objects[object_logits.argmax(-1)]. Polar head: also its peaks
    (POLAR) and their outlines (polar_outline), mask_pred being the drawn masks"""
    def __init__(self, interval="50ep", objects=None): self.interval, self.objects = Time.from_input(interval, TimeUnit.EPOCH).value, objects

    def save(self, state, batch):
        b = state.batch
        info = dict(sample_id=[int(s) if s.isdigit() else -1 for s in b["sample_id"]],  # -1: viz names the dir from position + speaker, e.g. 001464-1
                    position_id=b["position_id"].cpu(), speaker=b["speaker"].cpu(), experiment=list(b["experiment"]))
        path = Path(f"runs/{state.run_name}/outputs_history/{state.dataloader_label}/ep{state.timestamp.epoch.value:04d}-ba{batch:06d}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        metrics, geometry = sample_metrics(state.outputs, b)
        heads = {k: state.outputs[k].detach().float().cpu() for k in HEADS + POLAR if k in state.outputs}
        if "peak_rays" in state.outputs: heads.update(polar_outline(state.outputs))
        torch.save(dict(mask_pred=state.outputs["mask_pred"].detach().half().cpu(), **heads, objects=self.objects,
                        info=info, metrics=metrics, geometry=geometry), path)

    def after_forward(self, state, logger):
        if due(state, self.interval, train=True): self.save(state, state.timestamp.batch_in_epoch.value)

    def eval_after_forward(self, state, logger):
        if due(state, self.interval, train=False): self.save(state, state.eval_timestamp.batch.value)

class PositionLogger(Callback):
    """every position each loader serves goes into runs/<run>/positions/<loader label>.txt (once, as '<experiment> <position_id>'),
    and every batch asserts train and eval never share a position: the third, end-to-end position-leak check"""
    def __init__(self): self.seen = {}  # loader label -> positions served so far

    def log(self, state):
        label, b = state.dataloader_label, state.batch
        positions = {f"{e} {p}" for e, p in zip(b["experiment"], b["position_id"].tolist())}
        other_side = set().union(*[s for l, s in self.seen.items() if (l == "train") != (label == "train")])
        assert not positions & other_side, f"{label} served positions the other side (train/eval) also served: {sorted(positions & other_side)[:5]}"
        new = positions - self.seen.setdefault(label, set())
        self.seen[label] |= new
        path = Path(f"runs/{state.run_name}/positions/{label}.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f: f.writelines(p + "\n" for p in sorted(new))

    def after_forward(self, state, logger): self.log(state)
    def eval_after_forward(self, state, logger): self.log(state)

def scores(pred, true):
    """Per-sample scores (B,): peak probability overall, inside the true object, and in the background; fill (mean probability inside
    the object); soft iou; and miss (no pixel over 0.5 on the object). Object scores are NaN on an empty box.
    Pixel scores (every pixel): obj_px, the predictions inside the object, and bg_px, the background predictions over 0.01 (most of
    the background is ~0, which would bury everything else). The median of all pixels is useless here: it's background, ~0 throughout."""
    pred, area = pred.detach().float(), true.flatten(1).sum(1)
    obj, bg = true > 0.5, true == 0
    nan = lambda v: v.where(area > 0, torch.nan)
    peak = lambda m: pred.masked_fill(~m, 0).flatten(1).amax(1)
    return dict(peak=peak(torch.ones_like(obj)), peak_obj=nan(peak(obj)), peak_bg=peak(bg),
                fill=nan((pred * true).flatten(1).sum(1) / area),
                iou=nan(torch.minimum(pred, true).flatten(1).sum(1) / torch.maximum(pred, true).flatten(1).sum(1)),
                miss=nan((((pred > 0.5) & obj).flatten(1).sum(1) == 0).float()),
                obj_px=pred[obj], bg_px=pred[bg & (pred > 0.01)])

def log_scores(state, logger, group, pred, true):
    """each score's mean, its median and +-2 std band (as the 2.5/50/97.5% quantiles: scores live in [0,1], where mean +- 2 std
    overshoots). No wandb histograms: one per score per object every 10 batches was most of the run's history (~0.6GB)"""
    for k, v in scores(pred, true).items():
        v = v[~v.isnan()]
        v = v[::len(v) // 2**24 + 1]  # torch.quantile takes at most 2^24 values; strided, so the training rng is untouched
        if not len(v): continue
        q = torch.quantile(v, torch.tensor([0.025, 0.5, 0.975], device=v.device)).tolist()
        logger.log_metrics({f"conf/{group}/{k}_mean": v.mean().item(), f"conf/{group}/{k}_p2.5": q[0], f"conf/{group}/{k}_p50": q[1], f"conf/{group}/{k}_p97.5": q[2]})

class ConfidenceLogger(Callback):
    """scores (see above) per eval split, and per object on every `interval`-th train batch: evals are too sparse to catch the jump
    (at ~2000 steps train fill goes 0.5 -> 1 within ~400 steps while eval fill drops)"""
    def __init__(self, objects, interval=10): self.objects, self.interval, self.evals = [o or "empty" for o in objects], interval, []

    def after_forward(self, state, logger):
        if state.timestamp.batch.value % self.interval: return
        pred, true, obj = state.outputs["mask_pred"], state.batch["mask_true"], state.batch["object"]
        for i in obj.unique().tolist(): log_scores(state, logger, f"train-batch/{self.objects[i]}", pred[obj == i], true[obj == i])

    def eval_after_forward(self, state, logger): self.evals.append((state.outputs["mask_pred"].detach(), state.batch["mask_true"]))

    def eval_end(self, state, logger):  # composer fires this once per eval split
        pred, true = zip(*self.evals)
        log_scores(state, logger, state.dataloader_label, torch.cat(pred), torch.cat(true))
        self.evals = []

def nanmean(vs): return {k: torch.tensor([v[k] for v in vs]).nanmean().item() for k in vs[0] if k != "n"} | {"n": sum(v["n"] for v in vs)}

class EvalSummary(Callback):
    """per box with CATEGORIES (dataset.py): each category's mean over its eval splits, id, ood (mean over the categories not in
    not_ood) and DELTAS, logged as metrics/eval/<box>/{id,ood,cat/<category>,delta/<name>}/<metric>, next to composer's
    metrics/eval/<box>/<split>/<metric>. A split counts once, as its mean over samples (what composer logs), so big splits don't
    drown small ones; n = samples behind each number. train is never averaged."""
    def __init__(self, categories, deltas, not_ood): self.categories, self.deltas, self.not_ood, self.splits = categories, deltas, not_ood, {}

    def eval_end(self, state, logger):  # composer fires this once per eval split
        label, metrics = state.dataloader_label, state.eval_metrics[state.dataloader_label]
        if label.startswith("eval/"): self.splits[label] = {k: m.compute().item() for k, m in metrics.items()} | {"n": int(metrics["iou"].weight.item())}

    def summarize(self, logger):
        splits, self.splits = self.splits, {}
        for box, cats in self.categories.items():
            have = {l.split("/", 2)[2]: v for l, v in splits.items() if l.split("/")[1] == box}  # eval/<box>/<split>
            if not have: continue
            listed = [s for ss in cats.values() for s in ss]
            assert len(listed) == len(set(listed)), f"{box}: a split is in two categories"
            assert not set(have) - set(listed), f"{box}: eval splits in no category: {sorted(set(have) - set(listed))}"
            mean = lambda names: nanmean([have[s] for s in names if s in have]) if any(s in have for s in names) else None
            out = {f"cat/{c}": m for c, ss in cats.items() if (m := mean(ss))}
            if "cat/id" in out: out["id"] = out["cat/id"]
            if ood := [m for c, m in out.items() if c.split("/")[-1] not in self.not_ood and c.startswith("cat/")]: out["ood"] = nanmean(ood)
            for name, (a, b) in self.deltas.get(box, {}).items():
                if (ma := mean(a)) and (mb := mean(b)): out[f"delta/{name}"] = {k: ma[k] - mb[k] for k in ma if k != "n"}
            logger.log_metrics({f"metrics/eval/{box}/{g}/{k}": v for g, m in out.items() for k, v in m.items()})

    def eval_after_all(self, state, logger): self.summarize(logger)       # evals during fit
    def eval_standalone_end(self, state, logger): self.summarize(logger)  # trainer.eval before/after training
