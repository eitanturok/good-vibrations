from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from composer import Callback
from composer.core import Time, TimeUnit

# each callback runs on eval batches and, during training, on train batches, every `interval` epochs; train skips epoch 0 (the
# eval before training covers it)
def due(state, interval, train): return state.timestamp.epoch.value % interval == 0 and not (train and state.timestamp.epoch.value == 0)
def every(interval): return Time.from_input(interval, TimeUnit.EPOCH).value

def render(pred, true, caption, aspect, w=320):
    """predicted | true under a caption, at the box's real aspect (every box is squashed to the same grid)"""
    top, h, font = 40, round(w / aspect), ImageFont.load_default(size=13)
    canvas = Image.new("RGB", (2 * w + 4, top + h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((canvas.width // 2, 4), caption, fill="black", font=font, anchor="ma")
    for x, a, label in [(0, pred, "predicted"), (w + 4, true, "true")]:
        canvas.paste(Image.fromarray((a * 255).astype(np.uint8)).resize((w, h), Image.NEAREST), (x, top))
        draw.text((x + w // 2, top - 16), label, fill="black", font=font, anchor="mt")
    return np.array(canvas)

class VisualizeSMask(Callback):
    """the first n predicted vs true masks of each loader's first batch, to wandb as SMask/<label>"""
    def __init__(self, interval="100ep", n=16): self.interval, self.n = every(interval), n

    def log(self, state, logger):
        b, pred = state.batch, state.outputs["mask_pred"][:self.n].detach().float().cpu()
        true = b["mask_true"][:self.n].float().cpu()
        iou = torch.minimum(pred, true).flatten(1).sum(1) / torch.maximum(pred, true).flatten(1).sum(1).clamp_min(1e-6)
        imgs = [render(pred[i].numpy(), true[i].numpy(), f"{b['position_id'][i]}-{b['speaker'][i]} iou={iou[i]:.2f}", b["crop_w"][i].item() / b["crop_h"][i].item())
                for i in range(len(pred))]
        logger.log_images(imgs, name=f"SMask/{state.dataloader_label}", channels_last=True, use_table=False)

    def after_forward(self, state, logger):
        if due(state, self.interval, True) and state.timestamp.batch_in_epoch.value == 0: self.log(state, logger)
    def eval_after_forward(self, state, logger):
        if due(state, self.interval, False) and state.eval_timestamp.batch.value == 0: self.log(state, logger)

HEADS = ("object_logits", "n_objects_logits", "com", "log_area", "box_logits")

class OutputSaver(Callback):
    """each batch's mask_pred + aux heads + sample ids to runs/<run>/outputs_history/<label>/ep{epoch:04d}-ba{batch:06d}.pt (score.py, viz)"""
    def __init__(self, interval="50ep", objects=None): self.interval, self.objects = every(interval), objects

    def save(self, state, batch):
        b = state.batch
        info = dict(sample_id=[int(s) if s.isdigit() else -1 for s in b["sample_id"]], position_id=b["position_id"].cpu(),
                    speaker=b["speaker"].cpu(), experiment=list(b["experiment"]))
        path = Path(f"runs/{state.run_name}/outputs_history/{state.dataloader_label}/ep{state.timestamp.epoch.value:04d}-ba{batch:06d}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        heads = {k: state.outputs[k].detach().float().cpu() for k in HEADS if k in state.outputs}
        torch.save(dict(mask_pred=state.outputs["mask_pred"].detach().half().cpu(), **heads, objects=self.objects, info=info), path)

    def after_forward(self, state, logger):
        if due(state, self.interval, True): self.save(state, state.timestamp.batch_in_epoch.value)
    def eval_after_forward(self, state, logger):
        if due(state, self.interval, False): self.save(state, state.eval_timestamp.batch.value)

class PositionLogger(Callback):
    """every position each loader serves -> runs/<run>/positions/<label>.txt, and asserts train and eval never share one"""
    def __init__(self): self.seen = {}

    def log(self, state):
        label, b = state.dataloader_label, state.batch
        pos = {f"{e} {p}" for e, p in zip(b["experiment"], b["position_id"].tolist())}
        other = set().union(*[s for l, s in self.seen.items() if (l == "train") != (label == "train")])
        assert not pos & other, f"{label} served positions the other side also served: {sorted(pos & other)[:5]}"
        new = pos - self.seen.setdefault(label, set())
        self.seen[label] |= new
        path = Path(f"runs/{state.run_name}/positions/{label}.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f: f.writelines(p + "\n" for p in sorted(new))

    def after_forward(self, state, logger): self.log(state)
    def eval_after_forward(self, state, logger): self.log(state)

def nanmean(vs): return {k: float(np.nanmean([v[k] for v in vs])) for k in vs[0]}

class EvalSummary(Callback):
    """means over eval splits (each split once): core (4 base objects x 3 boxes), hard (cardboard rotated soap), unseen_angle,
    unseen_position, and per box id / ood / cat/<category>, as metrics/eval/<group>/<metric>.
    + nb113's metrics/eval/id/iou08 and metrics/eval/ood/iou08: per-sample means pooled over every box's id+multiple / ood splits"""
    CORE, HARD, ANGLE, POS = ("candle", "cube", "soap-dispenser", "vase"), "cardboard/soap-dispenser-rotated", "cardboard/soap-dispenser-unseen-angle", "unseen-position"
    ID, OOD = ("id", "multiple"), ("rotated", "cylinder", "unseen-object", "unseen-speaker", "unseen-laser", "lid")  # = nb113's ID_CATS, OOD_CATS

    def __init__(self, categories, not_ood): self.categories, self.not_ood, self.splits, self.sums = categories, not_ood, {}, {}

    def eval_end(self, state, logger):
        label = state.dataloader_label
        if not label.startswith("eval/"): return
        self.splits[label[5:]] = {k: m.compute().item() for k, m in state.eval_metrics[label].items()}
        m = state.eval_metrics[label]["iou08"]
        self.sums[label[5:]] = (m.mean_value.item(), m.weight.item())  # MeanMetric: iou08 sum, n samples

    def summarize(self, logger):
        splits, self.splits, sums, self.sums = self.splits, {}, self.sums, {}
        if not splits: return
        cat = {f"{b}/{s}": c for b, cs in self.categories.items() for c, ss in cs.items() for s in ss}  # "<box>/<split>" -> category
        for g, cats in (("id", self.ID), ("ood", self.OOD)):
            if vs := [v for s, v in sums.items() if cat.get(s) in cats]:
                n = sum(w for _, w in vs)
                logger.log_metrics({f"metrics/eval/{g}/iou08": sum(t for t, _ in vs) / n if n else float("nan")})
        groups = {"core": [v for s, v in splits.items() if s.split("/")[1] in self.CORE],
                  "hard": [splits[self.HARD]] if self.HARD in splits else [], "unseen_angle": [splits[self.ANGLE]] if self.ANGLE in splits else [],
                  "unseen_position": [v for s, v in splits.items() if s.split("/")[1] == self.POS]}
        for box, cats in self.categories.items():
            have = {s.split("/", 1)[1]: v for s, v in splits.items() if s.startswith(box + "/")}
            cm = {c: nanmean([have[s] for s in ss if s in have]) for c, ss in cats.items() if any(s in have for s in ss)}
            for c, m in cm.items(): logger.log_metrics({f"metrics/eval/{box}/cat/{c}/{k}": v for k, v in m.items()})
            if ood := [m for c, m in cm.items() if c not in self.not_ood]: groups[f"{box}/ood"] = ood
        for g, vs in groups.items():
            if vs: logger.log_metrics({f"metrics/eval/{g}/{k}": v for k, v in nanmean(vs).items()})

    def eval_after_all(self, state, logger): self.summarize(logger)
    def eval_standalone_end(self, state, logger): self.summarize(logger)

class SetEpoch(Callback):
    """model.epoch = the current epoch, for epoch-scheduled losses (transformer3's boundary ramp)"""
    def epoch_start(self, state, logger): state.model.epoch = state.timestamp.epoch.value

def cpu(v): return {k: cpu(x) for k, x in v.items()} if isinstance(v, dict) else v.detach().cpu() if torch.is_tensor(v) else v
def nonfinite(ts):
    """{name: tensor} -> {name: (n_nan, n_inf, numel)} of the floating tensors that aren't all finite (one GPU sync when all are)"""
    ts = {k: t for k, t in ts.items() if torch.is_tensor(t) and t.is_floating_point()}
    if not ts or torch.stack([t.isfinite().all() for t in ts.values()]).all(): return {}
    return {k: (int(t.isnan().sum()), int(t.isinf().sum()), t.numel()) for k, t in ts.items() if not t.isfinite().all()}

class NaNDebug(Callback):
    """run.py --debug-nan 1 (transformer3): on every train microbatch, the outputs (+ the post-aug input x_in and the drop masks m_l, m_b), the loss
    and the accumulated grads must be finite. The first time one isn't: dump to runs/<run>/nan_debug/ (dump.pt = event, step, lr, the exact microbatch,
    x_in / m_l / m_b, the CPU + CUDA RNG states before its forward, what is non-finite; model.pt / optim.pt = the state before this step's update),
    replay that microbatch eagerly (same autocast) from x_in / m_l / m_b with a forward hook on every module (first module to output non-finite)
    and the backward under detect_anomaly, write report.txt, then raise. Off = not in the callback list and model.debug_nan unset."""
    def __init__(self, model, optimizer): self.model, self.opt, self.mb, self.rng = model, optimizer, 0, None

    def batch_start(self, state, logger): self.mb = 0
    def before_forward(self, state, logger):
        self.rng = dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state() if torch.cuda.is_available() else None)
    def after_forward(self, state, logger):
        bad = nonfinite(state.outputs)
        if bad: self.dump(state, "forward", dict(outputs=bad))
    def after_loss(self, state, logger):
        bad = nonfinite(state.loss if isinstance(state.loss, dict) else dict(total=state.loss))
        if bad: self.dump(state, "loss", dict(loss=bad))
    def after_backward(self, state, logger):
        bad = nonfinite({n: q.grad for n, q in self.model.named_parameters() if q.grad is not None})
        if bad: self.dump(state, "grad", dict(grads=bad))
        self.mb += 1

    def replay(self, batch, x, m_l, m_b):
        """eager forward + backward of the saved microbatch -> report lines"""
        m, first, hooks, lines = self.model, [], [], []
        def hook(name):
            def f(mod, inp, out):
                outs = out.values() if isinstance(out, dict) else out if isinstance(out, (tuple, list)) else [out]
                if not first and any(torch.is_tensor(o) and o.is_floating_point() and not o.isfinite().all() for o in outs): first.append(f"{name} ({type(mod).__name__})")
            return f
        for name, mod in m.named_modules(): hooks.append(mod.register_forward_hook(hook(name or "<model>")))
        m.zero_grad(set_to_none=True)
        try:
            with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda), torch.autograd.detect_anomaly(check_nan=True):
                out = type(m).forward_tokens(m, x, m_l, m_b)
                loss = m.loss(out, batch)
                lines.append(f"replay forward: first module with a non-finite output: {first[0] if first else 'none (forward finite)'}")
                lines.append(f"replay loss: { {k: float(v) for k, v in loss.items()} }")
                loss["total"].backward()
            lines.append("replay backward: detect_anomaly raised nothing")
        except RuntimeError as e: lines.append(f"replay: {type(e).__name__}: {e}")
        finally:
            for h in hooks: h.remove()
        g = nonfinite({n: q.grad for n, q in m.named_parameters() if q.grad is not None})
        lines.append(f"replay grads: {len(g)} non-finite tensors {list(g)[:20]}")
        return lines

    def dump(self, state, event, bad):
        d = Path(f"runs/{state.run_name}/nan_debug"); d.mkdir(parents=True, exist_ok=True)
        out = state.outputs if isinstance(state.outputs, dict) else {}
        x, m_l, m_b = (out.get(k) for k in ("x_in", "m_l", "m_b"))
        info = dict(event=event, step=state.timestamp.batch.value, epoch=state.timestamp.epoch.value, batch_in_epoch=state.timestamp.batch_in_epoch.value,
                    microbatch=self.mb, lr=[g["lr"] for g in self.opt.param_groups], bad=bad, rng_before_forward=self.rng,
                    batch=cpu(state.batch), x_in=cpu(x), m_l=cpu(m_l), m_b=cpu(m_b))
        torch.save(info, d / "dump.pt")
        torch.save(self.model.state_dict(), d / "model.pt")                       # this step's update hasn't happened yet: the pre-step state
        torch.save(self.opt.state_dict(), d / "optim.pt")
        lines = [f"debug-nan: non-finite {event} at step {info['step']} (epoch {info['epoch']}, batch_in_epoch {info['batch_in_epoch']}, microbatch {self.mb}), lr {info['lr']}"]
        lines += [f"  {k}: {v}" for k, v in bad.items()]
        lines.append(f"  x_in non-finite: {nonfinite(dict(x_in=x)) if x is not None else 'n/a'}; kept lasers min {m_l.sum(1).min().item() if m_l is not None else 'n/a'}, kept bands min {m_b.sum(1).min().item() if m_b is not None else 'n/a'}")
        if x is not None: lines += ["  " + l for l in self.replay(state.batch, x.detach(), m_l, m_b)]
        (d / "report.txt").write_text("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)
        raise RuntimeError(f"debug-nan: non-finite {event} at step {info['step']}, dump in {d}")
