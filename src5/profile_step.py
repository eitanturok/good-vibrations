"""Times each component of one training step (encoder, decoder, frozen-AE decode, latent
loading, loss, backward) so speedups target the actual bottleneck instead of guessing. Also
reports peak GPU memory, since that's what caps batch size.
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]

import torch  # noqa: E402
from torch.utils.data import default_collate  # noqa: E402
from model.dataset import build_dataset  # noqa: E402
from src5.latent_model import LDMSegVibrationModel, load_latents, foreground_prob  # noqa: E402
from src5.losses import loss_ce, loss_mask  # noqa: E402

DATA_DIR = REPO / "experiments" / "31_07_2026_gastronorm_exp1"
WARMSTART = REPO / "runs" / "pp-baseline-v2" / "checkpoints" / "latest-rank0.pt"


def timed(fn, *a, n=5, **kw):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*a, **kw)
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) / n


def get_batch(batch_size: int):
    train_loader, _, _ = build_dataset(str(DATA_DIR), batch_size=batch_size, eval_batch_size=batch_size,
                                        num_workers=0, split="gastronorm", out_h=32, out_w=32,
                                        signal_mode="magnitude", normalize_mode="std", patch_size=32, verbose=0)
    ds = train_loader.dataloader.dataset
    items = [ds[i] for i in range(batch_size)]
    return default_collate(items)


def build_model(batch_size: int):
    train_loader, _, _ = build_dataset(str(DATA_DIR), batch_size=batch_size, eval_batch_size=batch_size,
                                        num_workers=0, split="gastronorm", out_h=32, out_w=32,
                                        signal_mode="magnitude", normalize_mode="std", patch_size=32, verbose=0)
    _, _, _, n_channels = train_loader.dataloader.dataset[0]["fft"].shape
    base = train_loader.dataloader.dataset.dataset
    base = getattr(base, "dataset", base)
    n_laser_rows, n_laser_cols = base.grid_shape
    data_info = dict(out_h=32, out_w=32, n_channels=n_channels, n_laser_rows=n_laser_rows, n_laser_cols=n_laser_cols)
    model = LDMSegVibrationModel(data_dir=DATA_DIR, data_info=data_info, warmstart_checkpoint=str(WARMSTART)).cuda()
    return model, data_info


def profile(batch_size: int = 8, n: int = 5, compile_model: bool = False) -> None:
    model, data_info = build_model(batch_size)
    if compile_model:
        model.encoder = torch.compile(model.encoder)
        model.decoder = torch.compile(model.decoder)
    batch = get_batch(batch_size)
    batch["fft"] = batch["fft"].cuda()
    batch["mask_true"] = batch["mask_true"].cuda()

    optimizer = torch.optim.AdamW(model.parameters(), 1e-4, fused=True)

    print(f"\n=== batch_size={batch_size} compile={compile_model} ===")
    torch.cuda.reset_peak_memory_stats()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, t_enc = timed(lambda: model.encoder(model._to_conv(batch["fft"])), n=n)
        emb = model.encoder(model._to_conv(batch["fft"]))
        _, t_dec = timed(lambda: model.decoder(emb).permute(0, 3, 1, 2).contiguous(), n=n)
        z_pred = model.decoder(emb).permute(0, 3, 1, 2).contiguous()

        _, t_load = timed(lambda: load_latents(batch["info"]["sample_id"], model.data_dir, z_pred.device), n=n)
        latents = load_latents(batch["info"]["sample_id"], model.data_dir, z_pred.device)

        _, t_aedecode = timed(lambda: model.mask_ae.decode(z_pred, interpolate=True), n=n)
        outputs = model.mask_ae.decode(z_pred, interpolate=True)

        _, t_fgprob = timed(lambda: foreground_prob(outputs, latents["target_classes"], 32, 32), n=n)

        def loss_fn():
            latent_l = torch.nn.functional.mse_loss(z_pred, latents["z_gt"])
            ce_l = loss_ce(outputs, latents["target_class_map"])
            bce_l, dice_l = loss_mask(outputs, latents["target_classes"], latents["instance_masks"])
            return latent_l + ce_l + bce_l + dice_l
        _, t_loss = timed(loss_fn, n=n)

    def full_step():
        optimizer.zero_grad()
        out = model.forward(batch)
        loss = model.loss(out, batch)
        total = sum(loss.values()) if isinstance(loss, dict) else loss
        total.backward()
        optimizer.step()
    _, t_full = timed(full_step, n=n)

    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"encoder forward:     {t_enc*1000:7.1f} ms")
    print(f"decoder forward:     {t_dec*1000:7.1f} ms")
    print(f"load_latents (I/O):  {t_load*1000:7.1f} ms")
    print(f"frozen AE decode:    {t_aedecode*1000:7.1f} ms")
    print(f"foreground_prob:     {t_fgprob*1000:7.1f} ms")
    print(f"loss (ce+mask+lat):  {t_loss*1000:7.1f} ms")
    print(f"FULL step (fwd+bwd+opt): {t_full*1000:7.1f} ms  ->  {batch_size / t_full:.1f} samples/sec")
    print(f"peak GPU memory:     {peak_mem:.2f} GB")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--compile", action="store_true")
    args = p.parse_args()
    profile(args.batch_size, args.n, args.compile)
