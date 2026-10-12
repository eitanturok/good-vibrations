import itertools, math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from composer import ComposerModel
from torchmetrics import MeanMetric
from torchmetrics.functional.image.lpips import _NoTrainLpips

#*** 1 layers

class ConvBlock(nn.Module):
    """conv -> batchnorm -> leaky relu"""
    def __init__(self, c_in, c_out, kernel, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, kernel, stride, padding)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), 0.2)

class TwoBranchUp(nn.Module):
    """The paper's 2x upsampling layer: a transposed conv, and a conv then a transposed conv, concatenated.
    Branch b's extra conv is what stops the two branches collapsing to the same function."""
    def __init__(self, c_in, c_out):
        super().__init__()
        self.up_a = nn.ConvTranspose2d(c_in, c_out // 2, kernel_size=4, stride=2, padding=1)
        self.up_b = nn.ConvTranspose2d(c_in, c_out // 2, kernel_size=4, stride=2, padding=1)  # before conv_b: src/ draws init in this order
        self.conv_b = nn.Conv2d(c_in, c_in, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        a = self.up_a(x)                                   # (B,c_in,H,W) -> (B,c_out/2,2H,2W)
        b = F.relu(self.conv_b(x))                         # (B,c_in,H,W) -> (B,c_in,H,W)
        b = self.up_b(b)                                   # (B,c_in,H,W) -> (B,c_out/2,2H,2W)
        x = torch.cat([a, b], dim=1)                       # 2 x (B,c_out/2,2H,2W) -> (B,c_out,2H,2W)
        return F.relu(self.bn(x))                          # (B,c_out,2H,2W) -> (B,c_out,2H,2W)

#*** 2 encoder

class FreqEncoder(nn.Module):
    """(B,L,F,C) spectrum -> (B,256,L): a 256-dim feature per laser.
    Convolves frequency only (kernel height 1), so lasers don't mix here: the laser grid samples global standing waves,
    not local texture. Four stride-4 convs shrink frequency, then whatever is left of it is averaged out."""
    def __init__(self, n_channels):
        super().__init__()
        self.conv1 = ConvBlock(n_channels, 32, kernel=(1, 7), stride=(1, 4), padding=(0, 3))
        self.conv2 = ConvBlock(32, 64, kernel=(1, 5), stride=(1, 4), padding=(0, 2))
        self.conv3 = ConvBlock(64, 128, kernel=(1, 5), stride=(1, 4), padding=(0, 2))
        self.conv4 = ConvBlock(128, 256, kernel=(1, 5), stride=(1, 4), padding=(0, 2))

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)                          # (B,L,F,C) -> (B,C,L,F)
        x = self.conv1(x)                                  # (B,C,L,F) -> (B,32,L,F/4)        e.g. F 1170 -> 293
        x = self.conv2(x)                                  # (B,32,L,F/4) -> (B,64,L,F/16)    293 -> 74
        x = self.conv3(x)                                  # (B,64,L,F/16) -> (B,128,L,F/64)  74 -> 19
        x = self.conv4(x)                                  # (B,128,L,F/64) -> (B,256,L,F/256) 19 -> 5
        return x.mean(dim=-1)                              # (B,256,L,F/256) -> (B,256,L)

class LaserGridEncoder(nn.Module):
    """(B,256,L) per-laser features -> (B,d_model).
    Lays the lasers out on their (rows,cols) grid, then two stride-2 convs (10x10 -> 5x5 -> 3x3) and one conv the size of
    what's left (3x3 -> 1x1) mix the whole grid into one embedding."""
    def __init__(self, d_model, n_laser_rows, n_laser_cols):
        super().__init__()
        self.grid_shape = (n_laser_rows, n_laser_cols)
        h, w = math.ceil(n_laser_rows / 4), math.ceil(n_laser_cols / 4)
        self.conv1 = ConvBlock(256, 512, kernel=3, stride=2, padding=1)
        self.conv2 = ConvBlock(512, 1024, kernel=3, stride=2, padding=1)
        self.conv3 = ConvBlock(1024, d_model, kernel=(h, w), stride=1, padding=0)

    def forward(self, x):
        x = x.view(len(x), 256, *self.grid_shape)          # (B,256,L) -> (B,256,rows,cols)  e.g. 10x10
        x = self.conv1(x)                                  # (B,256,10,10) -> (B,512,5,5)
        x = self.conv2(x)                                  # (B,512,5,5) -> (B,1024,3,3)
        x = self.conv3(x)                                  # (B,1024,3,3) -> (B,d_model,1,1)
        return x.flatten(1)                                # (B,d_model,1,1) -> (B,d_model)

#*** 3 decoder

class Decoder(nn.Module):
    """(B,d_model) -> (B,out_c,out_h,out_w).
    Seed a 512x4x4 map and double it until it covers the output (4 -> 8 -> 16 -> 32 -> 64), halving channels each time,
    resize to exactly (out_h,out_w) if that isn't a power of 2, then a 3-conv head emits the output."""
    def __init__(self, d_model, out_h, out_w, out_c):
        super().__init__()
        self.out_h, self.out_w, self.out_c = out_h, out_w, out_c
        # each upsampling stage doubles the 4x4 seed and halves the channels (floored at 16), until the map covers the output:
        # out 64 -> 4 stages, 4 -> 8 -> 16 -> 32 -> 64 with channels 512 -> 256 -> 128 -> 64 -> 32
        n_ups = math.ceil(math.log2(max(out_h, out_w) / 4))
        widths = [max(512 // 2 ** i, 16) for i in range(n_ups + 1)]
        self.project = nn.Linear(d_model, 512 * 4 * 4)
        self.ups = nn.ModuleList([TwoBranchUp(c_in, c_out) for c_in, c_out in zip(widths, widths[1:])])  # how many depends on the output size
        self.conv1 = nn.Conv2d(widths[-1], 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, self.out_c, kernel_size=3, padding=1)

    def forward(self, emb):
        x = self.project(emb)                              # (B,d_model) -> (B,512*4*4)
        x = x.view(-1, 512, 4, 4)                          # (B,512*4*4) -> (B,512,4,4)
        for up in self.ups:
            x = up(x)                                      # (B,c,S,S) -> (B,c/2,2S,2S): 4 -> 8 -> 16 -> 32 -> 64 at out 64x64
        if x.shape[-2:] != (self.out_h, self.out_w):       # out_h/out_w not powers of 2, e.g. 21x30
            x = F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)  # (B,c,S,S) -> (B,c,out_h,out_w)
        x = F.relu(self.conv1(x))                          # (B,c,out_h,out_w) -> (B,32,out_h,out_w)
        x = F.relu(self.conv2(x))                          # (B,32,out_h,out_w) -> (B,32,out_h,out_w)
        return self.conv3(x)                               # (B,32,out_h,out_w) -> (B,out_c,out_h,out_w)

#*** 4 helpers

def mask_area(mask): return mask.flatten(1).sum(1)  # (B,H,W) -> (B,)

def com_err(pred, true, n_objects):
    """(B,K,2) predicted and true per-object (row,col) coms -> the distance of each real object's com.
    A sample with n objects uses the first n predicted slots. The model can't know which object is 'first', so try every
    order of those slots (up to K! orders, small for a handful of objects) and keep the closest. Padding slots are dropped."""
    B, K, _ = pred.shape
    orders = torch.tensor(list(itertools.permutations(range(K))), device=pred.device)  # (P,K), P = K!
    real = torch.arange(K, device=pred.device) < n_objects[:, None]                     # (B,K)
    allowed = ((orders < n_objects[:, None, None]) | ~real[:, None]).all(-1)            # (B,P): real objects only take real slots
    d = (pred[:, orders] - true[:, None]).norm(dim=-1)     # (B,P,K,2) - (B,1,K,2) -> (B,P,K)
    cost = (d * real[:, None]).sum(-1).masked_fill(~allowed, float("inf"))  # (B,P,K) -> (B,P), total distance of each order
    best = cost.argmin(1)                                  # (B,P) -> (B,), the closest allowed order
    d = d[torch.arange(B), best]                           # (B,P,K) -> (B,K)
    return d[real]                                         # (B,K) -> (n_real_objects,)

def paste(mask, obj, bg, bank):
    """option B: retrieve the train sample whose mask best matches `mask` (same object class first, then highest iou) and paste its
    photo, inside its mask, onto the empty box. (B,H,W) mask, (B,) object, (B,H,W,3) bg -> (B,H,W,3) photo, (B,H,W) pasted mask"""
    with torch.no_grad(), torch.autocast(mask.device.type, enabled=False):  # fp32: bf16 matmul is too coarse to rank ious
        p, m = mask.float().flatten(1), bank["mask"].flatten(1)            # (B,HW), (N,HW)
        inter = p @ m.T                                                    # (B,HW) @ (HW,N) -> (B,N) soft intersection
        iou = inter / (p.sum(1, keepdim=True) + m.sum(1) - inter).clamp_min(1e-6)  # (B,N)
        best = (iou + (obj[:, None] == bank["object"])).argmax(1)          # (B,N) -> (B,): a class match beats any iou
        alpha = bank["mask"][best]                                         # (B,H,W)
        return bg * (1 - alpha[..., None]) + bank["rgb"][best] * alpha[..., None], alpha

#*** 5 losses

# mask losses: (mask logits, true mask or rgb image)
def bce_loss(logits, true): return F.binary_cross_entropy_with_logits(logits, true)
def mse_loss(logits, true): return F.mse_loss(logits.sigmoid(), true)
def dice_loss(logits, true): return 1 - (2 * (logits.sigmoid() * true).flatten(1).sum(1) + 1e-6).div((logits.sigmoid() + true).flatten(1).sum(1) + 1e-6).mean()  # soft, product form
def bce_dice_loss(logits, true, alpha=0.5): return alpha * bce_loss(logits, true) + (1 - alpha) * dice_loss(logits, true)

LOSS_FN = {"bce": bce_loss, "mse": mse_loss, "dice": dice_loss, "bce-dice": bce_dice_loss}

# aux losses
def object_loss(logits, obj, unseen): return F.cross_entropy(logits, obj, ignore_index=unseen)  # objects never seen in train are ignored
def n_objects_loss(logits, n_objects): return F.cross_entropy(logits, n_objects)
def com_loss(pred, true, n_objects): return com_err(pred, true, n_objects).sum() / n_objects.sum().clamp_min(1)  # mean over real objects
def area_loss(log_area, true): return F.l1_loss(log_area, mask_area(true).log1p())

#*** 6 metrics

def iou(out, batch): return torch.minimum(out["mask_pred"], batch["mask_true"]).flatten(1).sum(1) / torch.maximum(out["mask_pred"], batch["mask_true"]).flatten(1).sum(1).clamp_min(1e-6)  # soft, sum(min) / sum(max)
def object_acc(out, batch): return (out["object_logits"].argmax(-1) == batch["object"])[batch["object"] < out["object_logits"].shape[-1]].float()  # unseen objects (no logit) are left out
def n_objects_err(out, batch): return (out["n_objects_logits"].argmax(-1) - batch["n_objects"]).abs().float()  # counts are ordered: off by 2 is worse than off by 1
def com_error(out, batch): return com_err(out["com"], batch["coms"], batch["n_objects"])  # as a fraction of the box
def area_rel_err(out, batch): return ((out["log_area"].expm1() - mask_area(batch["mask_true"])).abs() / mask_area(batch["mask_true"]))[mask_area(batch["mask_true"]) > 0]

# rgb metrics, for each predicted photo: rgb (A, direct regression), paste (B, from the predicted mask + object), oracle (B from the
# true mask + object: the ceiling of pasting, what B would score with perfect sensing). Per sample, NaN where it doesn't apply
_lpips = None
def lpips(pred, true):
    global _lpips
    if _lpips is None: _lpips = _NoTrainLpips(net="alex").eval()  # frozen alexnet + lpips weights; not part of the model
    to = lambda x: x.float().clamp(0, 1).permute(0, 3, 1, 2) * 2 - 1  # (B,H,W,3) in [0,1] -> (B,3,H,W) in [-1,1]
    with torch.no_grad(): return _lpips.to(pred.device)(to(pred), to(true)).flatten()
def psnr(pred, true): return -10 * (pred.float().clamp(0, 1) - true).pow(2).flatten(1).mean(1).log10()
def obj_l1(pred, true, mask):  # mean abs error inside the true objects: psnr is mostly the (easy, constant) background
    area = mask.flatten(1).sum(1)
    return ((pred.float().clamp(0, 1) - true).abs().mean(-1) * mask).flatten(1).sum(1).div(area).where(area > 0, torch.nan)

RGB_METRICS = {f"{k}_{name}": (lambda out, batch, k=k, f=f: f(out[f"{k}_pred"], batch["rgb_true"], batch["mask_true"]))
               for k in ("rgb", "paste", "oracle")
               for name, f in [("psnr", lambda p, t, m: psnr(p, t)), ("obj_l1", obj_l1), ("lpips", lambda p, t, m: lpips(p, t))]}
RGB_METRICS |= {f"{k}_iou": (lambda out, batch, k=k: iou(dict(mask_pred=out[f"{k}_mask"]), batch)) for k in ("paste", "oracle")}

METRICS = {**RGB_METRICS, "iou": iou, "object_acc": object_acc, "n_objects_err": n_objects_err, "com_err": com_error, "area_rel_err": area_rel_err}

#*** 7 model

class Boombox(ComposerModel):
    def __init__(self, d_model: int, data_info: dict, bank: dict, out_h: int, out_w: int,
                 loss_fn: str = "bce", alpha: float = 0.5, loss_weights: dict | None = None):
        super().__init__()
        self.freq_encoder = FreqEncoder(data_info["n_channels"])
        self.grid_encoder = LaserGridEncoder(d_model, data_info["n_laser_rows"], data_info["n_laser_cols"])
        self.decoder = Decoder(d_model, out_h, out_w, out_c=4)  # mask logits + rgb residual on the empty box
        for k, v in bank.items(): self.register_buffer(f"bank_{k}", v, persistent=False)  # train samples to paste from; rebuilt with the data, not checkpointed

        # aux heads on the embedding. objects[-1] is "unseen": never a train target, so it gets no logit
        self.n_known = len(data_info["objects"]) - 1
        max_objects = max(data_info["n_objects"])
        mlp = lambda n_out: nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, n_out))
        self.object_head = mlp(self.n_known)       # which objects
        self.n_objects_head = mlp(max_objects + 1) # how many objects, 0..max_objects
        self.com_head = mlp(2 * max_objects)       # (row,col) com of each object
        self.area_head = mlp(1)                    # log(1 + mask area)

        # loss = mask loss + weighted aux losses (a weight of 0 means that head doesn't train)
        self.mask_loss = partial(bce_dice_loss, alpha=alpha) if loss_fn == "bce-dice" else LOSS_FN[loss_fn]
        self.loss_weights = loss_weights or dict(object=0.1, n_objects=0.1, com=1.0, area=0.1, rgb=1.0)

        # metrics, the same ones on train and test: tag each with its key so update_metric knows which one it was handed
        self.train_metrics = {key: MeanMetric() for key in METRICS}
        self.test_metrics = {key: MeanMetric() for key in METRICS}
        for key, metric in [*self.train_metrics.items(), *self.test_metrics.items()]: metric.key = key

    def forward(self, batch):
        x = self.freq_encoder(batch["fft"])                # (B,L,F,C) -> (B,256,L)
        emb = self.grid_encoder(x)                         # (B,256,L) -> (B,d_model)
        x = self.decoder(emb)                              # (B,d_model) -> (B,4,H,W)
        logits, object_logits = x[:, 0], self.object_head(emb)
        bank = dict(mask=self.bank_mask, rgb=self.bank_rgb, object=self.bank_object)
        paste_pred, paste_mask = paste(logits.sigmoid(), object_logits.argmax(-1), batch["bg"], bank)
        oracle_pred, oracle_mask = paste(batch["mask_true"], batch["object"], batch["bg"], bank)
        return dict(
            mask_logits=logits,                                       # (B,H,W)
            mask_pred=logits.sigmoid(),                               # (B,H,W) -> (B,H,W) in [0,1]
            rgb_pred=batch["bg"] + x[:, 1:].permute(0, 2, 3, 1),      # (B,H,W,3): option A, the empty box + a predicted residual
            paste_pred=paste_pred, paste_mask=paste_mask,             # (B,H,W,3), (B,H,W): option B
            oracle_pred=oracle_pred, oracle_mask=oracle_mask,         # (B,H,W,3), (B,H,W): option B from the true mask + object
            object_logits=object_logits,                              # (B,d_model) -> (B,n_known)
            n_objects_logits=self.n_objects_head(emb),                # (B,d_model) -> (B,max_objects+1)
            com=self.com_head(emb).sigmoid().view(len(emb), -1, 2),   # (B,d_model) -> (B,max_objects,2) in [0,1]
            log_area=self.area_head(emb).squeeze(-1),                 # (B,d_model) -> (B,)
        )

    def loss(self, out, batch):
        losses = dict(
            mask=self.mask_loss(out["mask_logits"], batch["mask_true"]),
            object=object_loss(out["object_logits"], batch["object"], self.n_known),
            n_objects=n_objects_loss(out["n_objects_logits"], batch["n_objects"]),
            com=com_loss(out["com"], batch["coms"], batch["n_objects"]),
            area=area_loss(out["log_area"], batch["mask_true"]),
            rgb=F.l1_loss(out["rgb_pred"], batch["rgb_true"]),
        )
        total = losses["mask"] + sum(weight * losses[key] for key, weight in self.loss_weights.items())
        return dict(total=total, **losses)  # composer backprops 'total' and logs every key

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.test_metrics

    def update_metric(self, batch, out, metric):
        v = METRICS[metric.key](out, batch)
        metric.update(v[~v.isnan()])  # NaN: doesn't apply to that sample, e.g. obj_l1 on an empty box
