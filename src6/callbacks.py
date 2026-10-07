from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from composer import Callback
from composer.core import Time, TimeUnit

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

# each callback runs on eval batches and, during training, on train batches: both every `interval` epochs. Train batches
# skip epoch 0, which the eval before training already covers -- they'd otherwise overwrite its train/ files.
def due(state, interval, train): return state.timestamp.epoch.value % interval == 0 and not (train and state.timestamp.epoch.value == 0)

class VisualizeSMask(Callback):
    """log each batch's predicted vs true masks to wandb as SMask/<loader label>"""
    def __init__(self, interval="50ep"): self.interval = Time.from_input(interval, TimeUnit.EPOCH).value

    def log(self, state, logger):
        b = state.batch
        pred = state.outputs["mask_pred"][:MAX_WANDB_IMAGES].detach().float().cpu()  # float: numpy has no bf16
        true = b["mask_true"][:MAX_WANDB_IMAGES].float().cpu()
        iou = torch.minimum(pred, true).flatten(1).sum(1) / torch.maximum(pred, true).flatten(1).sum(1).clamp_min(1e-6)
        images = [render(pred[i].numpy(), true[i].numpy(),
                         f"{b['position_id'][i]}-{b['speaker'][i]} iou={iou[i]:.2f}",
                         aspect=b["crop_w"][i].item() / b["crop_h"][i].item())
                  for i in range(len(pred))]
        logger.log_images(images, name=f"SMask/{state.dataloader_label}", channels_last=True, use_table=False)

    def after_forward(self, state, logger):
        if due(state, self.interval, train=True): self.log(state, logger)

    def eval_after_forward(self, state, logger):
        if due(state, self.interval, train=False): self.log(state, logger)

class OutputSaver(Callback):
    """save each batch's predicted masks + sample ids to runs/<run>/outputs_history/<loader label>/ep{epoch:04d}-ba{batch:06d}.pt,
    the files the viz dashboard reads"""
    def __init__(self, interval="50ep"): self.interval = Time.from_input(interval, TimeUnit.EPOCH).value

    def save(self, state, batch):
        b = state.batch
        info = dict(sample_id=[int(s) if s.isdigit() else -1 for s in b["sample_id"]],  # -1: viz names the dir from position + speaker, e.g. 001464-1
                    position_id=b["position_id"].cpu(), speaker=b["speaker"].cpu(), experiment=list(b["experiment"]))
        path = Path(f"runs/{state.run_name}/outputs_history/{state.dataloader_label}/ep{state.timestamp.epoch.value:04d}-ba{batch:06d}.pt")
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(dict(mask_pred=state.outputs["mask_pred"].detach().float().cpu(), info=info), path)

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
