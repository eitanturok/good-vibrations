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
    unseen_position, and per box id / ood / cat/<category>, as metrics/eval/<group>/<metric>"""
    CORE, HARD, ANGLE, POS = ("candle", "cube", "soap-dispenser", "vase"), "cardboard/soap-dispenser-rotated", "cardboard/soap-dispenser-unseen-angle", "unseen-position"

    def __init__(self, categories, not_ood): self.categories, self.not_ood, self.splits = categories, not_ood, {}

    def eval_end(self, state, logger):
        label = state.dataloader_label
        if label.startswith("eval/"): self.splits[label[5:]] = {k: m.compute().item() for k, m in state.eval_metrics[label].items()}

    def summarize(self, logger):
        splits, self.splits = self.splits, {}
        if not splits: return
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
