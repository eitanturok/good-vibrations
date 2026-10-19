import atexit, json, random, shutil, tempfile, textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.signal import butter, chirp, sosfiltfilt
from scipy.signal.windows import tukey
from tqdm import tqdm
from tabulate import tabulate, SEPARATING_LINE
from torch.utils.data import DataLoader, Subset
from composer.core import Evaluator
from streaming import StreamingDataset, MDSWriter

EXPERIMENTS = Path("/home/ethantu/workspace/good-vibrations/experiments")

#*** 1 collect samples

FFT_FILES = ("vibration/04_ffts.npz", "vibration/04_fft.npz")  # newer captures write 04_ffts, older 04_fft

def fft_path(d: Path) -> Path: return next(d / f for f in FFT_FILES if (d / f).exists())
def load_fft(d: Path) -> torch.Tensor: return torch.from_numpy(np.load(fft_path(d))["fft"].squeeze(0))  # (1,L,F,C) -> (L,F,C) complex64
# window="tukey": redo post_process.clean + fft from the raw shifts with a short tukey taper instead of hann over the whole 1.3s capture.
# a log sweep maps freq to time (the chirp starts ~0.19s in), so that hann was a freq-dependent gain (~0.2 at 100Hz, 1 at 300Hz,
# ~0.07 at 1000Hz) on the signal but not on the noise, which is spread over the whole capture: ~10 dB less SNR at the band edges.
# window="deconv": also divide by the played chirp's spectrum (a fixed per-bin gain: flattens the log sweep's -3 dB/oct, so every
# freq counts the same after the per-sample std norm; nb100's com probe: cardboard cube .79 -> .82)
# gate_ms > 0: also deconvolve -> impulse response -> keep [peak - 20ms, peak + gate_ms], dropping the noise off the sweep's
# time-freq path (and the harmonic-distortion echoes, which land before the peak)
def load_fft_raw(d: Path, meta: dict, deconv: bool = False, gate_ms: float = 0, alpha: float = 0.02) -> torch.Tensor:
    s, fps, lo, hi = np.load(d / "vibration/02_raw_shifts.npy"), float(meta["fps"]), float(meta["min_freq"]), float(meta["max_freq"])  # (L,T,2)
    T, band = s.shape[1], (np.fft.rfftfreq(s.shape[1], 1 / fps) >= lo) & (np.fft.rfftfreq(s.shape[1], 1 / fps) <= hi)
    X = np.fft.rfft(sosfiltfilt(butter(5, [lo, hi], fs=fps, btype="band", output="sos"), s, axis=1) * tukey(T, alpha)[None, :, None], axis=1)  # (L,T//2+1,2); alpha=0.02 -> 13ms tapers
    if deconv or gate_ms:
        c = chirp(np.arange(int(meta["chirp_t_sec"] * fps)) / fps, f0=meta["chirp_f_start"], t1=meta["chirp_t_sec"], f1=meta["chirp_f_end"], method="logarithmic")
        X = np.where(band[None, :, None], X / np.fft.rfft(c, T)[None, :, None], 0)                                       # (L,T//2+1,2) transfer function
    if gate_ms:
        h = np.fft.irfft(X, n=T, axis=1)                                                                                 # (L,T,2) impulse response
        pre, n = int(0.02 * fps), int(0.02 * fps) + int(gate_ms / 1000 * fps)
        h = np.roll(h, pre - (h ** 2).mean((0, 2)).argmax(), axis=1)                                                     # peak at 20ms
        X = np.fft.rfft(h * np.pad(tukey(n, 0.2), (0, T - n))[None, :, None], axis=1)
    return torch.from_numpy(X[:, band].astype(np.complex64))                                                             # (L,F,C), the same bins as 04_ffts

# window="farina": the tukey spectrum (channels 0..C-1, the usual input) + the spectrum of the 2nd-harmonic impulse response (C..2C-1).
# Deconvolving an exponential sweep puts harmonic k's impulse response at -L ln k (L = T / ln(f1/f0); k=2: ~0.30 s before the
# linear one), separate from it (Farina 2000). src8/signal_probe.py: the harmonics alone decode position (com skill .39-.68) and
# transfer across speakers best of all arms; it may be rattle / contact nonlinearity the linear spectrum doesn't show.
def load_fft_farina(d: Path, meta: dict) -> torch.Tensor:
    s, fps, lo, hi = np.load(d / "vibration/02_raw_shifts.npy"), float(meta["fps"]), float(meta["min_freq"]), float(meta["max_freq"])
    T, f = s.shape[1], np.fft.rfftfreq(s.shape[1], 1 / fps)
    band = (f >= lo) & (f <= hi)
    X = np.fft.rfft(sosfiltfilt(butter(5, [lo, hi], fs=fps, btype="band", output="sos"), s, axis=1) * tukey(T, 0.02)[None, :, None], axis=1)
    c = chirp(np.arange(int(meta["chirp_t_sec"] * fps)) / fps, f0=meta["chirp_f_start"], t1=meta["chirp_t_sec"], f1=meta["chirp_f_end"], method="logarithmic")
    C = np.fft.rfft(c, T)
    h = np.fft.irfft(np.where(band[None, :, None], X / C[None, :, None], 0), n=T, axis=1)                         # (L,T,2) impulse responses
    t2 = int((h ** 2).mean((0, 2)).argmax()) - int(round(meta["chirp_t_sec"] / np.log(meta["chirp_f_end"] / meta["chirp_f_start"]) * np.log(2) * fps))
    n = int(0.08 * fps)                                                                                              # 80 ms window, from 10 ms before
    i = (np.arange(n) + t2 - int(0.01 * fps)) % T
    H2 = np.fft.rfft(h[:, i] * tukey(n, 0.3)[None, :, None], n=T, axis=1) * np.median(np.abs(C[band]))  # back to |X|-like units (log floor 1e-6)
    return torch.from_numpy(np.concatenate([X[:, band], H2[:, band]], -1).astype(np.complex64))                     # (L,F,2C)
def load_meta(d: Path) -> dict: return {k: v for l in (d / "metadata.jsonl").read_text().splitlines() if l.strip() for k, v in json.loads(l).items()}

def collect_samples(exp_dir: Path) -> dict[Path, dict]:
    """{sample_dir: metadata}"""
    samples, skipped = {}, []
    for d in tqdm(sorted((exp_dir / "samples").iterdir()), desc=f"collecting {exp_dir.name}"):
        if (d / "metadata.jsonl").exists() and (d / "image/03_smask.png").exists() and any((d / f).exists() for f in FFT_FILES):
            samples[d] = dict(load_meta(d), experiment=exp_dir.name, sample_id=d.name)  # not every capture writes sample_id
        else: skipped.append(d.name)
    print(f"{exp_dir.name}: {len(samples)} samples, skipped {len(skipped)} incomplete: {skipped}")
    return samples

#*** 2 process x

def phase_ref(z: torch.Tensor, ref: str) -> torch.Tensor:
    """The reference r[f,c] that phase_norm rotates by, from z (L,F,C) complex: one value per (freq, channel), shared by every laser.
      avg:  r = mean_l z_l
      real: r = sqrt(sum_l z_l^2), so arg r = 1/2 arg sum_l z_l^2 -- the rotation that puts the most energy in the real part,
            max_th sum_l Re(e^{-i th} z_l)^2 (Re(w)^2 = (|w|^2 + Re(w^2))/2). Squaring doubles angles, so antiphase lasers add
            instead of cancelling as they do in the mean (viz2: avg is weak in ~1/3-2/3 of bins, real in ~10%)."""
    if ref == "avg": return z.mean(0)                                       # (F,C)
    r = (z * z).sum(0).sqrt()                                               # (F,C); principal root
    # real is only defined up to a pi flip, which would flip every laser's (cos,sin) at that bin at random -- pure noise to
    # the model. Fix the sign: the strongest laser at each (f,c) gets Re >= 0 after the rotation.
    top = z.gather(0, z.abs().argmax(0, keepdim=True))[0]                   # (F,C) the strongest laser's value
    return torch.where((top * r.conj()).real < 0, -r, r)                    # (F,C)

def phase_norm(z: torch.Tensor, ref: str) -> torch.Tensor:
    """x'[s,l,f,c] = x[s,l,f,c] * conj(r[s,f,c]) / |r[s,f,c]|, one sample at a time: every laser at a (sample, freq, channel)
    rotated by the same angle, so magnitudes and laser-to-laser phase differences survive. ref: none | avg | real (phase_ref)."""
    if ref == "none": return z
    r = phase_ref(z, ref)                                                   # (F,C)
    return z * (r.conj() / (r.abs() + 1e-12))[None]                         # (L,F,C)

def process_x(samples: dict[Path, dict], signal: str, norm: str, patch_size: int | None, window: str = "hann", gate_ms: float = 0, laser_norm: bool = False,
              phase: str = "none", phase_ref: str = "none", laser_center: bool = False):
    """fft -> (L,P,patch_size,C') tokens, or (L,F,C') unpatched if patch_size is None.
    phase="none": C' = C, the magnitude features. phase="cossin": C' = 2C, [mag_x, mag_y, phi_x, phi_y], phi the phase after
    phase_norm(phase_ref) in radians. Boombox.forward turns phi into cos and sin on the GPU (the model sees 3C channels): storing
    cos and sin here made a 1024-sample batch 2.9 GB and OOM-killed the run in RAM (2026-10-16); the angle alone is 1.9 GB."""
    # process lazily since FFTs are large (10GB+ for all samples)
    for d, meta in tqdm(samples.items(), desc="processing x"):
        z = phase_norm(load_fft(d) if window == "hann" else load_fft_farina(d, meta) if window == "farina" else load_fft_raw(d, meta, window == "deconv", gate_ms), phase_ref)  # (L,F,C) complex
        x = z.abs()                                                           # (L,F,C) |fft|; phase_norm leaves it unchanged
        x = x / x.pow(2).mean((1, 2), keepdim=True).sqrt() if laser_norm else x  # (L,F,C); each laser to rms 1: removes its gain (nb53 β_n; nb101 §10)
        x = torch.log(x + 1e-3) if signal == "log_magnitude" else x.sqrt() if signal == "sqrt_magnitude" else x  # (L,F,C); 1e-3 floors the dead bins (notebook 65); sqrt: nb101 §10
        x = x.pow(2) if signal == "power" else torch.log(x.pow(2) + 1e-6) if signal == "log_power" else x  # (L,F,C) |fft|^2; 1e-6 = the 1e-3 floor squared
        if laser_center: x = x - x.mean(0, keepdim=True)                      # (L,F,C); log signals: the speaker's S(f) is common to every laser, so it cancels
        x = (x - x.mean()) / x.std() if norm == "z" else x / x.std()          # (L,F,C), stats over the whole sample
        if phase == "cossin": x = torch.cat([x, z.angle()], dim=-1)           # (L,F,2C); bounded already: not part of the magnitude's stats
        if patch_size is None: yield x.numpy(); continue                      # (L,F,C), no patching
        (L, F_, C), P = x.shape, -(-x.shape[1] // patch_size)                 # P = ceil(F / patch_size)
        x = F.pad(x, (0, 0, 0, P * patch_size - F_))                          # (L,P*patch_size,C); zero-pad, truncating drops the top freqs
        yield x.reshape(L, P, patch_size, C).numpy()                          # (L,P,patch_size,C)

#*** 3 process y

def process_y(samples: dict[Path, dict], objects: list[str], boxes: list[str], max_objects: int, out_h: int, out_w: int, rgb: bool) -> list[dict]:
    """downsample mask image + each object's own mask, map object and box to index, map coms to tensor"""
    ys = []
    # a with-lid sample takes its targets from the same spot (and speaker) without the lid: its own photo shows the lid, not the object
    where = {(meta["experiment"], meta["position_id"], meta["speaker"]): d for d, meta in samples.items()}
    for d, meta in tqdm(samples.items(), desc="processing y"):
        if "without_lid_position_id" in meta:
            d = where[meta["experiment"], meta["without_lid_position_id"], meta["speaker"]]
            meta = samples[d]
        # y: the mask (h,w) or overhead photo (h,w,3), downsampled. Float before BOX resampling, so edge cells keep partial coverage
        img = Image.open(d / ("image/02_cropped_overhead.png" if rgb else "image/03_smask.png")).convert("RGB" if rgb else "F")
        y = np.clip(np.asarray(img.resize((out_w, out_h), Image.BOX), dtype=np.float32) / 255.0, 0.0, 1.0)
        # object: index into objects; combos never seen in train map to "unseen"
        k = "+".join(sorted(meta["objects"]))
        obj = objects.index(k if k in objects else "unseen")
        # coms: (max_objects,2) per-object (row,col) com as a fraction of the cropped overhead, zero-padded past n_objects.
        # metadata stores each object's com as [[row, col]] in pixels
        H, W = meta["crop_overhead_shape"][:2]
        coms = [[c[0][0] / H, c[0][1] / W] for c in meta["coms"]]
        # inst: (max_objects,h,w) each object's own mask (image/smasks/<object><i>.npy), downsampled like y, zero-padded past n_objects
        inst = [np.asarray(Image.fromarray((np.load(f) > 0).astype(np.float32)).resize((out_w, out_h), Image.BOX), dtype=np.float32).clip(0, 1)
                for f in sorted((d / "image/smasks").glob("*.npy"))] if not rgb else []
        inst = np.stack(inst + [np.zeros((out_h, out_w), np.float32)] * (max_objects - len(inst)))[:max_objects]
        ys.append(dict(y=y, inst=inst, object=obj, box=boxes.index(meta["box"]), coms=np.array(coms + [[0.0, 0.0]] * (max_objects - len(coms)), dtype=np.float32)))
    return ys

#*** 4 datasets

def split(layouts, eval_frac, speakers=None, keep=lambda position_id: True):  # speakers=None -> all speakers
    return dict(layouts=layouts, eval_frac=eval_frac, speakers=speakers, keep=keep)

# speakers=None trains and evals on every speaker and holds none out: the unseen-speaker splits go, and their
# positions become one more held-out-positions split, so every other split keeps the same eval positions
def gastronorm(test_size=0.2, speakers=(1, 3, 5, 7)) -> dict[str, dict]:
    # 35 gastro cube positions held out of gastro/cube, to compare an unseen speaker (4) with the trained ones at the same positions
    UNSEEN_SPK = {8, 17, 18, 20, 21, 27, 49, 52, 54, 57, 62, 76, 84, 86, 106, 118, 119, 120, 122, 127,
                  133, 143, 148, 150, 180, 182, 185, 193, 204, 224, 226, 239, 245, 269, 291}
    gastro = {
        "empty-box":                    split(["empty-box-1", "empty-box-2", "empty-box-3", "empty-box-4"], 0.0, speakers),
        "vase":                         split(["vase-grid-1"], test_size, speakers),
        "candle":                       split(["candle"], test_size, speakers),
        "cube":                         split(["red-cube-grid1", "red-cube-grid2", "red-cube-grid3_test"], test_size, speakers, keep=lambda p: p not in UNSEEN_SPK),
        "cube-unseen-speaker-baseline": split(["red-cube-grid1", "red-cube-grid2", "red-cube-grid3_test"], 1.0, speakers, keep=lambda p: p in UNSEEN_SPK),
        "cube-unseen-speaker-unseen":   split(["red-cube-grid1", "red-cube-grid2", "red-cube-grid3_test"], 1.0, (4,), keep=lambda p: p in UNSEEN_SPK),
        "soap-dispenser":               split(["soap-dispenser-grid-1", "soap-dispenser-grid-2", "soap-dispenser-grid-3"], test_size, speakers),
        "cube-vase":                    split(["cube-vase"], test_size, speakers),
        "two-cubes":                    split(["two-cubes-grid-1", "two-cubes-grid-2"], test_size, speakers),
        "cube-with-lid":                split(["red-cube-with-lid"], 1.0, speakers),     # told apart by spectrum (the lid's photos still show the cube)
        "cube-without-lid":             split(["red-cube-without-lid"], 1.0, speakers),
        "cylinder-1000g":               split(["cylinder-1000g"], 1.0, speakers),
        "cylinder-500g":                split(["cylinder-500g"], 1.0, speakers),
        "cylinder-100g":                split(["cylinder-100g"], 1.0, speakers),
        "coffee-pot":                   split(["coffee-pot"], 1.0, speakers),
        "soap-dispenser-sideways":      split(["soap-dispenser-sideways"], 1.0, speakers),
    }
    if speakers is None:
        del gastro["cube-unseen-speaker-baseline"], gastro["cube-unseen-speaker-unseen"]
        gastro["cube-35pos"] = split(["red-cube-grid1", "red-cube-grid2", "red-cube-grid3_test"], test_size, None, keep=lambda p: p in UNSEEN_SPK)
    return {f"gastro/{k}": dict(v, experiment="2026_09_27_gastronorm_four_objs") for k, v in gastro.items()}

def plastic(test_size=0.2, speakers=(1, 3, 5, 7)) -> dict[str, dict]:
    plastic = {
        "empty-box":                      split(["empty-box-grid-1", "empty-box-2", "empty-box-3", "empty-box-4", "empty-box-5"], 0.0, speakers),
        "soap-dispenser":                 split([f"soap-dispenser-grid-{i}" for i in range(1, 6)], test_size, speakers),
        "vase":                           split(["vase-grid-1", "vase-grid-2", "vase-grid-3"], test_size, speakers),
        "candle":                         split(["candle-grid-1", "candle-grid-2", "candle-grid-3"], test_size, speakers),
        "cube":                           split(["cube-box-grid-1", "cube-box-grid-2", "cube-box-grid-3"], test_size, speakers),
        "coffee-pot":                     split(["coffee-pot-grid-1", "coffee-pot-grid-2"], 1.0, speakers),
        "two-cubes":                      split(["two-cubes-grid-1", "two-cubes-grid-2"], test_size, speakers),
        "cube-vase":                      split(["cube-vase"], test_size, speakers),
        "cube-unseen-speaker-baseline":   split(["cube-5-speakers"], 1.0, speakers),
        "cube-unseen-speaker-unseen":     split(["cube-5-speakers"], 1.0, (4,)),
        "cube-with-lid":                  split(["red-cube-with-lid"], 1.0, speakers),     # same 8 spots as cube-without-lid; the lid hides the cube from the overhead camera
        "cube-without-lid":               split(["red-cube-without-lid"], 1.0, speakers),
        "portable-charger":               split(["portable-charger"], 1.0, speakers),
        "cylinder-200g":                  split(["cylinder-200g"], 1.0, speakers),
        "soap-dispenser-sideways":        split(["soap-dispenser-sideways"], 1.0, speakers),
        "cylinder-100g":                  split(["cylinder-100g"], 1.0, speakers),
        "cylinder-500g":                  split(["cylinder-500g"], 1.0, speakers),
        "cylinder-1000g":                 split(["cylinder-1000g"], 1.0, speakers),
        "cube-shifted-lasers-1":          split(["cube-shifted-lasers-1"], 1.0, speakers),
        "cube-shifted-lasers-2":          split(["cube-shifted-lasers-2"], 1.0, speakers),
        "cube-shifted-lasers-3-backside": split(["cube-shifted-lasers-3-backside"], 1.0, speakers),
    }
    if speakers is None:
        del plastic["cube-unseen-speaker-baseline"], plastic["cube-unseen-speaker-unseen"]
        plastic["cube-5-speakers"] = split(["cube-5-speakers"], test_size, None)
    return {f"plastic/{k}": dict(v, experiment="2026_10_05_green_plastic_four_objs") for k, v in plastic.items()}

def cardboard(test_size=0.2, speakers=(1, 3, 5, 7)) -> dict[str, dict]:
    cardboard = {
        "empty-box":                    split(["empty-box-3", "empty-box-4", "empty-box-5", "empty-box-6", "empty-box-grid-1", "empty-box-grid-2"], 0.0, speakers),
        "biscoff":                      split(["biscoff"], 1.0, speakers),
        "candle":                       split(["candle-grid-1", "candle-grid-2", "candle-grid-3"], test_size, speakers),
        "vase":                         split(["vase-grid-1", "vase-grid-2", "vase-grid-3"], test_size, speakers),
        "vase-laser-shifted-2":         split(["vase-laser-shifted-2"], 1.0, speakers),
        "vase-laser-shifted-3":         split(["vase-laser-shifted-3"], 1.0, speakers),
        "vase-laser-shifted-4":         split(["vase-laser-shifted-4", "empty-laser-shifted-4"], 1.0, speakers),
        "vase-with-lid":                split(["vase-with-lid"], 1.0, speakers),     # same 9 spots as vase-without-lid; the lid hides the vase from the overhead camera
        "vase-without-lid":             split(["vase-without-lid"], 1.0, speakers),
        "vase-unseen-speaker-baseline": split(["vase-unseeen-speaker4"], 1.0, speakers),
        "vase-unseen-speaker-unseen":   split(["vase-unseeen-speaker4"], 1.0, (4,)),
        "cube-vase":                    split(["cube-vase"], test_size, speakers),
        "soap-dispenser-rotated":       split(["soap-dispenser-rotated", "soap-dispenser-rotated-2", "soap-dispenser-sideways"], test_size, speakers),
        "coffee-pot":                   split(["coffee-pot"], 1.0, speakers),
        "cube-rotated":                 split(["cube-different-rotations"], test_size, speakers),
        "cube":                         split(["cube-grid-1", "cube-grid-2", "cube-random_poses", "cube-random-pos2"], test_size, speakers),
        "cylinder-1000g":               split(["cylinder-1000g", "cylinder-1000g-grid-1"], test_size, speakers),
        "cylinder-500g":                split(["cylinder-500g-grid-1", "cylinder-500g-grid-2", "cylinder-500g-grid-2."], test_size, speakers),
        "cylinder-100g":                split(["cylinder-100g-grid-1"], test_size, speakers),
        "cylinder-200g":                split(["cylinder-200g-grid-1", "cylinder-200g-grid-2"], test_size, speakers),
        "glass-cup":                    split(["glass-cup"], test_size, speakers),
        "glass-cup-broken":             split(["glass-cup-broken"], test_size, speakers),
        "portable-charger":             split(["portable-charger"], 1.0, speakers),
        "portable-charger-sideways":    split(["portable-charger-sideways"], 1.0, speakers),
        "soap-dispenser":               split(["soap-dispenser-grid-1", "soap-dispenser-grid-2", "soap-dispenser-grid-3"], test_size, speakers),
        "tomato-sauce":                 split(["tomato-sauce"], 1.0, speakers),
        "two-cubes":                    split(["two-cubes"], test_size, speakers),
    }
    if speakers is None:
        del cardboard["vase-unseen-speaker-baseline"], cardboard["vase-unseen-speaker-unseen"]
        cardboard["vase-5-speakers"] = split(["vase-unseeen-speaker4"], test_size, None)
    return {f"cardboard/{k}": dict(v, experiment="2026_10_11_cardboard_four_objs") for k, v in cardboard.items()}

# cardboard with every rotated layout held out entirely (eval only), so rotations are never trained on
def cardboard_no_rotations(test_size=0.2, speakers=(1, 3, 5, 7)) -> dict[str, dict]:
    return {k: dict(v, eval_frac=1.0) if k in ("cardboard/soap-dispenser-rotated", "cardboard/cube-rotated") else v for k, v in cardboard(test_size, speakers).items()}

def gastronorm_plastic(test_size=0.2, speakers=(1, 3, 5, 7)) -> dict[str, dict]: return gastronorm(test_size, speakers) | plastic(test_size, speakers)
def gastronorm_plastic_cardboard(test_size=0.2, speakers=(1, 3, 5, 7)) -> dict[str, dict]: return gastronorm_plastic(test_size, speakers) | cardboard(test_size, speakers)

# eval splits grouped into categories, per box: every eval split of a box is in exactly one category (the eval
# callback asserts it). control splits are only there as the reference of a delta, so neither id nor ood averages them
CATEGORIES = {
    "gastro": {
        "id":                    ["candle", "cube", "soap-dispenser", "vase", "cube-35pos"],
        "cylinder":              ["cylinder-1000g", "cylinder-500g", "cylinder-100g"],
        "multiple":              ["two-cubes", "cube-vase"],
        "rotated":               ["soap-dispenser-sideways"],
        "lid":                   ["cube-with-lid"],
        "unseen-object":         ["coffee-pot"],
        "unseen-speaker":        ["cube-unseen-speaker-unseen"],
        "control":               ["cube-without-lid", "cube-unseen-speaker-baseline"],
    },
    "plastic": {
        "id":                    ["candle", "cube", "soap-dispenser", "vase", "cube-5-speakers"],
        "cylinder":              ["cylinder-1000g", "cylinder-500g", "cylinder-200g", "cylinder-100g"],
        "multiple":              ["two-cubes", "cube-vase"],
        "rotated":               ["soap-dispenser-sideways"],
        "lid":                   ["cube-with-lid"],
        "unseen-object":         ["coffee-pot", "portable-charger"],
        "unseen-speaker":        ["cube-unseen-speaker-unseen"],
        "unseen-laser":          ["cube-shifted-lasers-1", "cube-shifted-lasers-2", "cube-shifted-lasers-3-backside"],
        "control":               ["cube-without-lid", "cube-unseen-speaker-baseline"],
    },
    "cardboard": {
        "id":                    ["candle", "cube", "soap-dispenser", "vase", "vase-5-speakers"],
        "cylinder":              ["cylinder-1000g", "cylinder-500g", "cylinder-200g", "cylinder-100g"],
        "multiple":              ["two-cubes", "cube-vase"],
        "rotated":               ["soap-dispenser-rotated", "cube-rotated"],
        "lid":                   ["vase-with-lid"],
        "broken":                ["glass-cup-broken"],
        "unseen-object":         ["biscoff", "tomato-sauce", "coffee-pot", "portable-charger"],
        "unseen-rotated-object": ["portable-charger-sideways"],
        "unseen-speaker":        ["vase-unseen-speaker-unseen"],
        "unseen-laser":          ["vase-laser-shifted-2", "vase-laser-shifted-3", "vase-laser-shifted-4"],
        "control":               ["vase-without-lid", "glass-cup", "vase-unseen-speaker-baseline"],
    },
}
NOT_OOD = ("id", "control")  # ood = mean over every other category
# delta = mean of the first splits' means - mean of the second's: the same positions (or object) with and without the change
DELTAS = {
    "gastro": {
        "lid":     (["cube-with-lid"], ["cube-without-lid"]),
        "speaker": (["cube-unseen-speaker-unseen"], ["cube-unseen-speaker-baseline"]),
    },
    "plastic": {
        "lid":     (["cube-with-lid"], ["cube-without-lid"]),
        "speaker": (["cube-unseen-speaker-unseen"], ["cube-unseen-speaker-baseline"]),
        "laser":   (["cube-shifted-lasers-1", "cube-shifted-lasers-2", "cube-shifted-lasers-3-backside"], ["cube"]),
    },
    "cardboard": {
        "lid":     (["vase-with-lid"], ["vase-without-lid"]),
        "broken":  (["glass-cup-broken"], ["glass-cup"]),
        "speaker": (["vase-unseen-speaker-unseen"], ["vase-unseen-speaker-baseline"]),
        "laser":   (["vase-laser-shifted-2", "vase-laser-shifted-3", "vase-laser-shifted-4"], ["vase"]),
    },
}

DATASETS = {"gastronorm": gastronorm, "plastic": plastic, "cardboard": cardboard, "cardboard_no_rotations": cardboard_no_rotations, "gastronorm_plastic": gastronorm_plastic, "gastronorm_plastic_cardboard": gastronorm_plastic_cardboard}

# positions are numbered per capture, so a position is (experiment, position_id)
def key(meta: dict) -> tuple[str, int]: return meta["experiment"], meta["position_id"]

def rows(split: dict, metas: list[dict]) -> list[int]:
    return [i for i, meta in enumerate(metas) if meta["experiment"] == split["experiment"] and meta["layout"] in split["layouts"]
            and (split["speakers"] is None or meta["speaker"] in split["speakers"]) and split["keep"](meta["position_id"])]

def build_splits(splits: dict, metas: list[dict], seed: int = 42) -> dict[str, list[int]]:
    """{"train": idxs, "eval/<name>": idxs} into metas"""
    idxs = {"train": []}
    for name, split in splits.items():
        rs = rows(split, metas)
        positions = sorted({metas[i]["position_id"] for i in rs})
        random.Random(seed).shuffle(positions)  # split by position, so every speaker at a position lands on the same side
        held = set(positions[:round(split["eval_frac"] * len(positions))])
        idxs["train"] += [i for i in rs if metas[i]["position_id"] not in held]
        if held: idxs[f"eval/{name}"] = [i for i in rs if metas[i]["position_id"] in held]
    # leak check 1: no position is in both train and an eval split
    train = {key(metas[i]) for i in idxs["train"]}
    for name, ids in idxs.items():
        if name != "train": assert not train & {key(metas[i]) for i in ids}, f"{name} shares positions with train"
    return {name: sorted(ids) for name, ids in idxs.items()}  # sorted, so the shuffle doesn't depend on the order splits are listed in

def wrap(layouts: list[str]) -> str:
    """layouts as lines that fit next to the other columns (~100 chars), breaking only between names, so a wide row
    wraps inside its own column instead of the terminal wrapping it back to column 0"""
    return textwrap.fill(", ".join(layouts), max(20, shutil.get_terminal_size().columns - 100), break_on_hyphens=False, break_long_words=False)

def print_splits(splits: dict, metas: list[dict], idxs: dict[str, list[int]]) -> None:
    """one table per experiment, counts are positions; total rows count each position once"""
    n_pos = lambda ids: len({key(metas[i]) for i in ids})
    evals = [i for name, ids in idxs.items() if name != "train" for i in ids]
    def total(label, keep):
        train, evl = [i for i in idxs["train"] if keep(i)], [i for i in evals if keep(i)]
        n_train, n_eval = n_pos(train), n_pos(evl)
        return ["", label, n_train + n_eval, n_train, n_eval, round(100 * n_eval / (n_train + n_eval)), sorted({metas[i]["speaker"] for i in train + evl})]
    headers = ["#", "split", "total", "train", "eval", "eval %", "speakers", "layouts"]
    exps = sorted({split["experiment"] for split in splits.values()})
    for exp in exps:
        table = []
        for name, split in splits.items():
            if split["experiment"] != exp: continue
            n, n_eval = n_pos(rows(split, metas)), n_pos(idxs.get(f"eval/{name}", []))
            table.append([name.split("/", 1)[-1], n, n - n_eval, n_eval, round(100 * split["eval_frac"]), split["speakers"] or "all", wrap(split["layouts"])])  # drop the box prefix
        table = [[k, *row] for k, row in enumerate(sorted(table, key=lambda row: -row[1]))]  # biggest split first
        table.append(total("total", lambda i: metas[i]["experiment"] == exp))
        print(f"\n{exp}\n" + tabulate(table, headers, tablefmt="simple_grid"))
    table = sorted((total(exp, lambda i, exp=exp: metas[i]["experiment"] == exp)[1:] for exp in exps), key=lambda row: -row[1])
    table = [[k, *row] for k, row in enumerate(table)]
    table += [total("total", lambda i: True)]
    print("\nall experiments\n" + tabulate(table, ["#", "experiment", *headers[2:-1]], tablefmt="simple_grid"))

#*** 6 cast to mds

def write_mds(mds_dir: Path, metas: list[dict], Xs, ys: list[dict]) -> None:
    """Xs, ys: each sample's processed input and targets, in the same order as metas; row i of the MDS is metas[i]"""
    shutil.rmtree(mds_dir, ignore_errors=True)
    columns = {"X": "ndarray:float32", "y": "ndarray:float32", "inst": "ndarray:float32", "object": "int", "box": "int", "coms": "ndarray:float32", "experiment": "str", "position_id": "int"}
    with MDSWriter(out=str(mds_dir), columns=columns, size_limit="200MB") as w:
        for meta, X, y in zip(metas, Xs, ys):
            w.write({"X": X, **y, "experiment": meta["experiment"], "position_id": int(meta["position_id"])})
    print(f"wrote {len(metas)} samples to {mds_dir} ({sum(f.stat().st_size for f in mds_dir.rglob('*')) / 1e9:.1f} GB)")

#*** 7 data info

def get_data_info(samples: dict[Path, dict], train_idxs: list[int], phase: str = "none", window: str = "hann") -> dict:
    """what the model and losses need to know about the data"""
    metas = list(samples.values())
    _, n_freqs, n_channels = load_fft(next(iter(samples))).shape  # (L,F,C); n_freqs = real bins, before tokenize pads to whole patches
    n_channels *= 2 if window == "farina" else 1                    # + the 2nd harmonic's x, y
    n_channels *= 3 if phase == "cossin" else 1                     # the MODEL's input: [mag, cos, sin] per channel (Boombox.phase_features)
    return dict(phase=phase, n_laser_rows=metas[0]["n_rows"], n_laser_cols=metas[0]["n_cols"], n_freqs=n_freqs, n_channels=n_channels,
                freq_range=(float(metas[0]["min_freq"]), float(metas[0]["max_freq"])),                  # Hz of the first and last bin
                objects=sorted({"+".join(sorted(metas[i]["objects"])) for i in train_idxs}) + ["unseen"],  # object combos seen in train, + "unseen" for any other
                n_objects=sorted({meta["n_objects"] for meta in metas}),                     # every object count in the data, train and eval
                boxes=sorted({meta["box"] for meta in metas}))                               # every box type in the data, train and eval

#*** 8 dataloader

class VibrationDataset(StreamingDataset):
    def __init__(self, mds_dir: Path, metas: list[dict], banned: set[tuple[str, int]]):
        super().__init__(local=str(mds_dir), shuffle=False)
        self.metas = metas    # metadata of each sample
        self.banned = banned  # positions this dataset cannot use: eval positions can't use train positions, train positions can't use eval positions

    def __getitem__(self, i):
        row, meta = super().__getitem__(i), self.metas[i]

        # leak check 2: on every item served, train never sees an eval position and eval never sees a train position
        assert (row["experiment"], row["position_id"]) == key(meta), f"MDS row {i} != metadata {key(meta)}"
        assert key(meta) not in self.banned, f"served banned position {key(meta)}"

        return dict(fft=torch.from_numpy(row["X"].copy()), mask_true=torch.from_numpy(row["y"].copy()), inst=torch.from_numpy(row["inst"].copy()),
                    object=row["object"], box=row["box"], coms=torch.from_numpy(row["coms"].copy()),
                    experiment=meta["experiment"], position_id=meta["position_id"], speaker=meta["speaker"],
                    sample_id=meta["sample_id"], n_objects=meta["n_objects"], layout=meta["layout"],
                    crop_h=meta["crop_overhead_shape"][0], crop_w=meta["crop_overhead_shape"][1])  # full-res box size, so viz can draw the real aspect

#*** 9 build dataset

def build_dataset(dataset: str = "gastronorm_plastic", test_size: float = 0.2, speakers=(1, 3, 5, 7),
                  signal: str = "magnitude", laser_center: bool = False, norm: str = "std", patch_size: int | None = None, window: str = "hann", gate_ms: float = 0, laser_norm: bool = False,
                  phase: str = "none", phase_ref: str = "none",
                  out_h: int = 64, out_w: int = 64, rgb: bool = False,
                  batch_size: int = 128, eval_batch_size: int = 108, num_workers: int = 4):

    # choose how to split the dataset
    splits = DATASETS[dataset](test_size, speakers) # list of split: (layouts, eval_frac, speakers, optional keep position_ids)

    # collect the samples in this dataset
    exps = sorted({split["experiment"] for split in splits.values()})
    samples = {d: meta for e in exps for d, meta in collect_samples(EXPERIMENTS / e).items()}  # {sample_dir: metadata}
    metas = list(samples.values())

    # actually split the dataset
    idxs = build_splits(splits, metas)
    print_splits(splits, metas, idxs)

    # process inputs: compute fft magnitude, normalize fft, tokenize
    Xs = process_x(samples, signal, norm, patch_size, window, gate_ms, laser_norm, phase, phase_ref, laser_center)

    # process outputs: downsample mask, map object and box to index, map coms to tensor
    data_info = get_data_info(samples, idxs["train"], phase, window) | dict(signal=signal, laser_center=laser_center)
    ys = process_y(samples, data_info["objects"], data_info["boxes"], max(data_info["n_objects"]), out_h, out_w, rgb)

    # save as MDS for fast training + delete when done
    (EXPERIMENTS / "mds").mkdir(exist_ok=True)
    mds_dir = Path(tempfile.mkdtemp(prefix=f"{dataset}_", dir=EXPERIMENTS / "mds"))
    atexit.register(shutil.rmtree, mds_dir, ignore_errors=True)
    write_mds(mds_dir, metas, Xs, ys)

    # build train/eval dataloaders
    train_pos = {key(metas[i]) for i in idxs["train"]}
    eval_pos = {key(metas[i]) for name, ids in idxs.items() if name != "train" for i in ids}
    train_ds, eval_ds = VibrationDataset(mds_dir, metas, eval_pos), VibrationDataset(mds_dir, metas, train_pos)
    dl = lambda ds, idxs, bs, shuffle: DataLoader(Subset(ds, idxs), batch_size=bs, shuffle=shuffle, drop_last=shuffle, num_workers=num_workers,
                                              pin_memory=True, persistent_workers=num_workers > 0)
    train_dl = dl(train_ds, idxs["train"], batch_size, True)
    eval_dl = [Evaluator(label=name, dataloader=dl(eval_ds, ids, eval_batch_size, False)) for name, ids in idxs.items() if name != "train"]
    return train_dl, eval_dl, data_info
