import atexit, csv, json, random, shutil, tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.signal import butter, sosfiltfilt
from scipy.signal.windows import tukey
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader, Subset
from composer.core import Evaluator
from streaming import StreamingDataset, MDSWriter

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
CACHE = ROOT / "runs/_src9_cache"   # footprint-overlap cache
MDS = EXPERIMENTS / "mds_src9"      # per-run temp MDS copies (~15 GB each, deleted at exit)
EXCLUDE = Path(__file__).parent / "exclude.csv"  # FFT outliers (nb96 re-run): mistimed chirps (|lag| >= 40 ms) + 2 broadband-noise captures

# the held-out floor region: (row0, row1, col0, col1) as box fractions, lasers on the row-1 wall (src8's, left of centre). Held out for
# REGION_OBJECTS only, by FOOTPRINT: no trained sample has a pixel of one of them in it (multi-object scenes too); every other object trains
# there. <box>/unseen-position = their single-object positions with >= REGION_EVAL of the footprint inside (REGION.md)
REGION, REGION_EVAL, REGION_OBJECTS = (0.4, 0.6, 0.3, 0.5), 0.5, ("vase",)

#*** 1 samples

FFT_FILES = ("vibration/04_ffts.npz", "vibration/04_fft.npz")  # newer captures write 04_ffts, older 04_fft

def load_meta(d: Path) -> dict: return {k: v for l in (d / "metadata.jsonl").read_text().splitlines() if l.strip() for k, v in json.loads(l).items()}
def fft_path(d: Path) -> Path: return next(d / f for f in FFT_FILES if (d / f).exists())

def load_fft(d: Path, meta: dict, window: str) -> torch.Tensor:
    """(L,F,C) complex spectrum. hann: the saved 04_ffts (hann over the whole capture). tukey: redone from the raw shifts with a 13 ms
    taper: the log sweep maps freq to time, so hann was a freq-dependent gain on the signal but not the noise (~10 dB SNR at the band edges)"""
    if window == "hann": return torch.from_numpy(np.load(fft_path(d))["fft"].squeeze(0))
    s, fps, lo, hi = np.load(d / "vibration/02_raw_shifts.npy"), float(meta["fps"]), float(meta["min_freq"]), float(meta["max_freq"])  # (L,T,2)
    f = np.fft.rfftfreq(s.shape[1], 1 / fps)
    X = np.fft.rfft(sosfiltfilt(butter(5, [lo, hi], fs=fps, btype="band", output="sos"), s, axis=1) * tukey(s.shape[1], 0.02)[None, :, None], axis=1)
    return torch.from_numpy(X[:, (f >= lo) & (f <= hi)].astype(np.complex64))  # the same bins as 04_ffts

def collect_samples(exp_dir: Path) -> dict[Path, dict]:
    samples = {}
    for d in sorted((exp_dir / "samples").iterdir()):
        if (d / "metadata.jsonl").exists() and (d / "image/03_smask.png").exists() and any((d / f).exists() for f in FFT_FILES):
            samples[d] = dict(load_meta(d), experiment=exp_dir.name, sample_id=d.name)
    print(f"{exp_dir.name}: {len(samples)} samples")
    return samples

#*** 2 x, y

def process_x(samples: dict[Path, dict], window: str):
    """|fft| / its std over the whole sample, (L,F,C) float32; lazy, the FFTs are 10 GB+"""
    for d, meta in tqdm(samples.items(), desc="processing x"):
        x = load_fft(d, meta, window).abs()
        yield (x / x.std()).numpy()

def downsample(a, h, w): return np.clip(np.asarray(Image.fromarray(a).resize((w, h), Image.BOX), dtype=np.float32), 0, 1)

def process_y(samples: dict[Path, dict], objects: list[str], boxes: list[str], max_objects: int, h: int, w: int) -> list[dict]:
    """mask (h,w), each object's own mask (max_objects,h,w), object index, box index, per-object (row,col) coms as box fractions"""
    where = {(m["experiment"], m["position_id"], m["speaker"]): d for d, m in samples.items()}
    ys = []
    for d, meta in tqdm(samples.items(), desc="processing y"):
        if "without_lid_position_id" in meta:  # a with-lid photo shows the lid: take the targets of its without-lid twin
            d = where[meta["experiment"], meta["without_lid_position_id"], meta["speaker"]]
            meta = samples[d]
        y = downsample(np.asarray(Image.open(d / "image/03_smask.png").convert("F")) / 255.0, h, w)
        k = "+".join(sorted(meta["objects"]))
        H, W = meta["crop_overhead_shape"][:2]
        coms = [[c[0][0] / H, c[0][1] / W] for c in meta["coms"]] + [[0.0, 0.0]] * max_objects
        inst = [downsample((np.load(f) > 0).astype(np.float32), h, w) for f in sorted((d / "image/smasks").glob("*.npy"))]
        inst = np.stack(inst + [np.zeros((h, w), np.float32)] * max_objects)[:max_objects]
        ys.append(dict(y=y, inst=inst, object=objects.index(k if k in objects else "unseen"), box=boxes.index(meta["box"]),
                       coms=np.array(coms[:max_objects], np.float32)))
    return ys

#*** 3 splits

def split(layouts, eval_frac, speakers, keep=lambda p: True): return dict(layouts=layouts, eval_frac=eval_frac, speakers=speakers, keep=keep)

def gastronorm(t, spk):
    UNSEEN_SPK = {8, 17, 18, 20, 21, 27, 49, 52, 54, 57, 62, 76, 84, 86, 106, 118, 119, 120, 122, 127,
                  133, 143, 148, 150, 180, 182, 185, 193, 204, 224, 226, 239, 245, 269, 291}  # gastro cube positions for the unseen speaker 4
    cube = ["red-cube-grid1", "red-cube-grid2", "red-cube-grid3_test"]
    s = {
        "empty-box":                    split(["empty-box-1", "empty-box-2", "empty-box-3", "empty-box-4"], 0.0, spk),
        "vase":                         split(["vase-grid-1"], t, spk),
        "candle":                       split(["candle"], t, spk),
        "cube":                         split(cube, t, spk, lambda p: p not in UNSEEN_SPK),
        "cube-unseen-speaker-baseline": split(cube, 1.0, spk, lambda p: p in UNSEEN_SPK),
        "cube-unseen-speaker-unseen":   split(cube, 1.0, (4,), lambda p: p in UNSEEN_SPK),
        "soap-dispenser":               split(["soap-dispenser-grid-1", "soap-dispenser-grid-2", "soap-dispenser-grid-3"], t, spk),
        "cube-vase":                    split(["cube-vase"], t, spk),
        "two-cubes":                    split(["two-cubes-grid-1", "two-cubes-grid-2"], t, spk),
        "cube-with-lid":                split(["red-cube-with-lid"], 1.0, spk),
        "cube-without-lid":             split(["red-cube-without-lid"], 1.0, spk),
        "cylinder-1000g":               split(["cylinder-1000g"], 1.0, spk),
        "cylinder-500g":                split(["cylinder-500g"], 1.0, spk),
        "cylinder-100g":                split(["cylinder-100g"], 1.0, spk),
        "coffee-pot":                   split(["coffee-pot"], 1.0, spk),
        "soap-dispenser-sideways":      split(["soap-dispenser-sideways"], 1.0, spk),
    }
    return {f"gastro/{k}": dict(v, experiment="2026_09_27_gastronorm_four_objs") for k, v in s.items()}

def plastic(t, spk):
    s = {
        "empty-box":                      split(["empty-box-grid-1", "empty-box-2", "empty-box-3", "empty-box-4", "empty-box-5"], 0.0, spk),
        "soap-dispenser":                 split([f"soap-dispenser-grid-{i}" for i in range(1, 6)], t, spk),
        "vase":                           split(["vase-grid-1", "vase-grid-2", "vase-grid-3"], t, spk),
        "candle":                         split(["candle-grid-1", "candle-grid-2", "candle-grid-3"], t, spk),
        "cube":                           split(["cube-box-grid-1", "cube-box-grid-2", "cube-box-grid-3"], t, spk),
        "coffee-pot":                     split(["coffee-pot-grid-1", "coffee-pot-grid-2"], 1.0, spk),
        "two-cubes":                      split(["two-cubes-grid-1", "two-cubes-grid-2"], t, spk),
        "cube-vase":                      split(["cube-vase"], t, spk),
        "cube-unseen-speaker-baseline":   split(["cube-5-speakers"], 1.0, spk),
        "cube-unseen-speaker-unseen":     split(["cube-5-speakers"], 1.0, (4,)),
        "cube-with-lid":                  split(["red-cube-with-lid"], 1.0, spk),
        "cube-without-lid":               split(["red-cube-without-lid"], 1.0, spk),
        "portable-charger":               split(["portable-charger"], 1.0, spk),
        "cylinder-200g":                  split(["cylinder-200g"], 1.0, spk),
        "soap-dispenser-sideways":        split(["soap-dispenser-sideways"], 1.0, spk),
        "cylinder-100g":                  split(["cylinder-100g"], 1.0, spk),
        "cylinder-500g":                  split(["cylinder-500g"], 1.0, spk),
        "cylinder-1000g":                 split(["cylinder-1000g"], 1.0, spk),
        "cube-shifted-lasers-1":          split(["cube-shifted-lasers-1"], 1.0, spk),
        "cube-shifted-lasers-2":          split(["cube-shifted-lasers-2"], 1.0, spk),
        "cube-shifted-lasers-3-backside": split(["cube-shifted-lasers-3-backside"], 1.0, spk),
    }
    return {f"plastic/{k}": dict(v, experiment="2026_10_05_green_plastic_four_objs") for k, v in s.items()}

# cardboard rotated-soap positions with the mask axis at 30-60 deg, held out entirely: an unseen angle
UNSEEN_ANGLE = {4502, 4513, 4514, 4540, 4541, 4556, 4557, 4558, 4559, 4604, 4605, 4606, 4607, 4610, 4613, 4616, 4620, 4625, 4626,
                4628, 4633, 4634, 4635, 4636, 4641, 4642, 4643}

def cardboard(t, spk):
    s = {
        "empty-box":                    split(["empty-box-3", "empty-box-4", "empty-box-5", "empty-box-6", "empty-box-grid-1", "empty-box-grid-2"], 0.0, spk),
        "biscoff":                      split(["biscoff"], 1.0, spk),
        "candle":                       split(["candle-grid-1", "candle-grid-2", "candle-grid-3"], t, spk),
        "vase":                         split(["vase-grid-1", "vase-grid-2", "vase-grid-3"], t, spk),
        "vase-laser-shifted-2":         split(["vase-laser-shifted-2"], 1.0, spk),
        "vase-laser-shifted-3":         split(["vase-laser-shifted-3"], 1.0, spk),
        "vase-laser-shifted-4":         split(["vase-laser-shifted-4", "empty-laser-shifted-4"], 1.0, spk),
        "vase-with-lid":                split(["vase-with-lid"], 1.0, spk),
        "vase-without-lid":             split(["vase-without-lid"], 1.0, spk),
        "vase-unseen-speaker-baseline": split(["vase-unseeen-speaker4"], 1.0, spk),
        "vase-unseen-speaker-unseen":   split(["vase-unseeen-speaker4"], 1.0, (4,)),
        "cube-vase":                    split(["cube-vase"], t, spk),
        "soap-dispenser-rotated":       split(["soap-dispenser-rotated", "soap-dispenser-rotated-2", "soap-dispenser-sideways"], t, spk),
        "coffee-pot":                   split(["coffee-pot"], 1.0, spk),
        "cube-rotated":                 split(["cube-different-rotations"], t, spk),
        "cube":                         split(["cube-grid-1", "cube-grid-2", "cube-random_poses", "cube-random-pos2"], t, spk),
        "cylinder-1000g":               split(["cylinder-1000g", "cylinder-1000g-grid-1"], t, spk),
        "cylinder-500g":                split(["cylinder-500g-grid-1", "cylinder-500g-grid-2", "cylinder-500g-grid-2."], t, spk),
        "cylinder-100g":                split(["cylinder-100g-grid-1"], t, spk),
        "cylinder-200g":                split(["cylinder-200g-grid-1", "cylinder-200g-grid-2"], t, spk),
        "glass-cup":                    split(["glass-cup"], t, spk),
        "glass-cup-broken":             split(["glass-cup-broken"], t, spk),
        "portable-charger":             split(["portable-charger"], 1.0, spk),
        "portable-charger-sideways":    split(["portable-charger-sideways"], 1.0, spk),
        "soap-dispenser":               split(["soap-dispenser-grid-1", "soap-dispenser-grid-2", "soap-dispenser-grid-3"], t, spk),
        "tomato-sauce":                 split(["tomato-sauce"], 1.0, spk),
        "two-cubes":                    split(["two-cubes"], t, spk),
    }
    # the wedge: eval positions drawn as without it (shuffle_keep), then it's taken out, so the other eval positions don't move
    rot = s["soap-dispenser-rotated"]
    s["soap-dispenser-rotated"] = dict(rot, keep=lambda p: p not in UNSEEN_ANGLE, shuffle_keep=rot["keep"])
    s["soap-dispenser-unseen-angle"] = dict(rot, eval_frac=1.0, keep=lambda p: p in UNSEEN_ANGLE)
    return {f"cardboard/{k}": dict(v, experiment="2026_10_11_cardboard_four_objs") for k, v in s.items()}

def footprint(d: Path, meta: dict, objects=REGION_OBJECTS) -> np.ndarray | None:
    """full-res bool footprint of the sample's REGION_OBJECTS: their own instance masks (image/smasks/<object><i>.npy), else the scene's
    03_smask; None if none of them is in the scene"""
    if not set(meta.get("objects") or ()) & set(objects): return None
    inst = [np.load(f) > 0 for f in (d / "image/smasks").glob("*.npy") if f.stem.rstrip("0123456789") in objects]
    return np.any(inst, 0) if inst else np.asarray(Image.open(d / "image/03_smask.png").convert("L")) > 127

def region_overlap(exp: str, region=REGION, objects=REGION_OBJECTS) -> dict[int, float]:
    """{position_id: fraction of the REGION_OBJECTS footprint inside region}, positions without them left out; one sample per position. Cached"""
    f = CACHE / f"{exp}_{'_'.join(map(str, region))}_{'+'.join(objects)}.json"
    if f.exists(): return {int(k): v for k, v in json.loads(f.read_text()).items()}
    r0, r1, c0, c1, out, seen = *region, {}, set()
    for d in tqdm(sorted((EXPERIMENTS / exp / "samples").iterdir()), desc=f"footprints {exp}"):
        if not (d / "metadata.jsonl").exists() or (m := load_meta(d)).get("position_id") in seen: continue
        seen.add(m["position_id"])
        if (mk := footprint(d, m, objects)) is None or not mk.any(): continue
        H, W = mk.shape
        out[m["position_id"]] = float(mk[round(r0 * H):round(r1 * H), round(c0 * W):round(c1 * W)].sum() / mk.sum())
    f.parent.mkdir(parents=True, exist_ok=True); f.write_text(json.dumps(out))
    return out

def get_splits(test_size=0.2, speakers=(1, 3, 5, 7)) -> dict[str, dict]:
    """the three boxes, the cardboard 30-60 deg wedge held out, and REGION held out for REGION_OBJECTS only: every trained split loses the
    positions where one of them has a pixel in it (eval positions still drawn as in the wedge split, then dropped, so every other position
    lands where the wedge split put it); <box>/unseen-position = their single-object positions >= REGION_EVAL inside"""
    splits = gastronorm(test_size, speakers) | plastic(test_size, speakers) | cardboard(test_size, speakers)
    for box in ("gastro", "plastic", "cardboard"):
        exp = splits[f"{box}/candle"]["experiment"]
        frac = region_overlap(exp)
        for k, s in splits.items():
            if k.startswith(box + "/") and s["eval_frac"] < 1.0:
                splits[k] = dict(s, keep=lambda q, keep=s["keep"], frac=frac: keep(q) and frac.get(q, 0.0) == 0.0, shuffle_keep=s.get("shuffle_keep", s["keep"]))
        layouts = [l for o in REGION_OBJECTS for l in splits[f"{box}/{o}"]["layouts"]]  # single-object layouts
        splits[f"{box}/unseen-position"] = dict(split(layouts, 1.0, speakers, lambda q, frac=frac: frac.get(q, 0.0) >= REGION_EVAL), experiment=exp)
    return splits

# eval splits per box and category (EvalSummary, score.py); ood = mean over the categories not in NOT_OOD
CATEGORIES = {
    "gastro": {"id": ["candle", "cube", "soap-dispenser", "vase"], "cylinder": ["cylinder-1000g", "cylinder-500g", "cylinder-100g"],
               "multiple": ["two-cubes", "cube-vase"], "rotated": ["soap-dispenser-sideways"], "lid": ["cube-with-lid"], "unseen-object": ["coffee-pot"],
               "unseen-speaker": ["cube-unseen-speaker-unseen"], "unseen-position": ["unseen-position"], "control": ["cube-without-lid", "cube-unseen-speaker-baseline"]},
    "plastic": {"id": ["candle", "cube", "soap-dispenser", "vase"], "cylinder": ["cylinder-1000g", "cylinder-500g", "cylinder-200g", "cylinder-100g"],
                "multiple": ["two-cubes", "cube-vase"], "rotated": ["soap-dispenser-sideways"], "lid": ["cube-with-lid"], "unseen-object": ["coffee-pot", "portable-charger"],
                "unseen-speaker": ["cube-unseen-speaker-unseen"], "unseen-laser": ["cube-shifted-lasers-1", "cube-shifted-lasers-2", "cube-shifted-lasers-3-backside"],
                "unseen-position": ["unseen-position"], "control": ["cube-without-lid", "cube-unseen-speaker-baseline"]},
    "cardboard": {"id": ["candle", "cube", "soap-dispenser", "vase"], "cylinder": ["cylinder-1000g", "cylinder-500g", "cylinder-200g", "cylinder-100g"],
                  "multiple": ["two-cubes", "cube-vase"], "rotated": ["soap-dispenser-rotated", "cube-rotated"], "lid": ["vase-with-lid"], "broken": ["glass-cup-broken"],
                  "unseen-object": ["biscoff", "tomato-sauce", "coffee-pot", "portable-charger"], "unseen-rotated-object": ["portable-charger-sideways"],
                  "unseen-angle": ["soap-dispenser-unseen-angle"], "unseen-speaker": ["vase-unseen-speaker-unseen"],
                  "unseen-laser": ["vase-laser-shifted-2", "vase-laser-shifted-3", "vase-laser-shifted-4"], "unseen-position": ["unseen-position"],
                  "control": ["vase-without-lid", "glass-cup", "vase-unseen-speaker-baseline"]},
}
NOT_OOD = ("id", "control", "unseen-angle", "unseen-position")

def key(meta: dict) -> tuple[str, int]: return meta["experiment"], meta["position_id"]  # positions are numbered per capture

def rows(s: dict, metas: list[dict]) -> list[int]:
    return [i for i, m in enumerate(metas) if m["experiment"] == s["experiment"] and m["layout"] in s["layouts"]
            and m["speaker"] in s["speakers"] and s["keep"](m["position_id"])]

def build_splits(splits: dict, metas: list[dict], seed: int = 42) -> dict[str, list[int]]:
    """{"train": idxs, "eval/<name>": idxs}: each split's positions shuffled with the same seed, the first eval_frac held out"""
    idxs = {"train": []}
    for name, s in splits.items():
        rs = rows(s, metas)
        pool = rows(dict(s, keep=s["shuffle_keep"]), metas) if "shuffle_keep" in s else rs
        positions = sorted({metas[i]["position_id"] for i in pool})
        random.Random(seed).shuffle(positions)  # by position: every speaker at a position lands on the same side
        held = set(positions[:round(s["eval_frac"] * len(positions))])
        idxs["train"] += [i for i in rs if metas[i]["position_id"] not in held]
        if held: idxs[f"eval/{name}"] = [i for i in rs if metas[i]["position_id"] in held]
    train = {key(metas[i]) for i in idxs["train"]}
    for name, ids in idxs.items():  # leak check: no position in train and eval
        if name != "train": assert not train & {key(metas[i]) for i in ids}, f"{name} shares positions with train"
    return {name: sorted(ids) for name, ids in idxs.items()}

def load_splits(test_size=0.2, speakers=(1, 3, 5, 7)):
    """(samples {dir: meta}, idxs) without touching the spectra"""
    splits = get_splits(test_size, speakers)
    samples = {d: m for e in sorted({s["experiment"] for s in splits.values()}) for d, m in collect_samples(EXPERIMENTS / e).items()}
    idxs = build_splits(splits, list(samples.values()))
    bad = {(r["experiment"], r["sample_id"]) for r in csv.DictReader(open(EXCLUDE))}  # capture glitches: dropped from train AND eval
    metas = list(samples.values())
    idxs = {name: [i for i in ids if (metas[i]["experiment"], metas[i]["sample_id"]) not in bad] for name, ids in idxs.items()}
    n = lambda ids: len({key(list(samples.values())[i]) for i in ids})
    print(tabulate([[name, len(ids), n(ids)] for name, ids in idxs.items()], ["split", "samples", "positions"]))
    return samples, idxs

#*** 4 dataloaders

class VibrationDataset(StreamingDataset):
    def __init__(self, mds_dir: Path, metas: list[dict], banned: set):
        super().__init__(local=str(mds_dir), shuffle=False)
        self.metas, self.banned = metas, banned  # banned: the other side's positions (train vs eval)

    def __getitem__(self, i):
        row, m = super().__getitem__(i), self.metas[i]
        assert (row["experiment"], row["position_id"]) == key(m) and key(m) not in self.banned, f"bad row {i}: {key(m)}"
        t = lambda k: torch.from_numpy(row[k].copy())
        return dict(fft=t("X"), mask_true=t("y"), inst=t("inst"), object=row["object"], box=row["box"], coms=t("coms"),
                    experiment=m["experiment"], position_id=m["position_id"], speaker=m["speaker"], sample_id=m["sample_id"],
                    n_objects=m["n_objects"], layout=m["layout"], crop_h=m["crop_overhead_shape"][0], crop_w=m["crop_overhead_shape"][1])

def build_dataset(window="tukey", batch_size=1024, eval_batch_size=108, num_workers=4):
    samples, idxs = load_splits()
    metas = list(samples.values())
    L, F, C = load_fft(next(iter(samples)), metas[0], "hann").shape
    info = dict(n_laser_rows=metas[0]["n_rows"], n_laser_cols=metas[0]["n_cols"], n_freqs=F, n_channels=C,
                freq_range=(float(metas[0]["min_freq"]), float(metas[0]["max_freq"])),
                objects=sorted({"+".join(sorted(metas[i]["objects"])) for i in idxs["train"]}) + ["unseen"],  # combos seen in train + "unseen"
                n_objects=sorted({m["n_objects"] for m in metas}), boxes=sorted({m["box"] for m in metas}))
    ys = process_y(samples, info["objects"], info["boxes"], max(info["n_objects"]), 64, 64)

    MDS.mkdir(exist_ok=True)
    mds = Path(tempfile.mkdtemp(prefix=f"{window}_", dir=MDS))
    atexit.register(shutil.rmtree, mds, ignore_errors=True)
    cols = {"X": "ndarray:float32", "y": "ndarray:float32", "inst": "ndarray:float32", "object": "int", "box": "int", "coms": "ndarray:float32", "experiment": "str", "position_id": "int"}
    with MDSWriter(out=str(mds), columns=cols, size_limit="200MB") as w:
        for m, X, y in zip(metas, process_x(samples, window), ys): w.write({"X": X, **y, "experiment": m["experiment"], "position_id": int(m["position_id"])})

    train_pos = {key(metas[i]) for i in idxs["train"]}
    eval_pos = {key(metas[i]) for n, ids in idxs.items() if n != "train" for i in ids}
    train_ds, eval_ds = VibrationDataset(mds, metas, eval_pos), VibrationDataset(mds, metas, train_pos)
    # eval: no workers (~70 eval splits x persistent workers ran the machine out of files and RAM)
    train_dl = DataLoader(Subset(train_ds, idxs["train"]), batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers,
                          pin_memory=True, persistent_workers=num_workers > 0)
    eval_dl = [Evaluator(label=n, dataloader=DataLoader(Subset(eval_ds, ids), batch_size=eval_batch_size)) for n, ids in idxs.items() if n != "train"]
    train_eval = Evaluator(label="train", dataloader=DataLoader(Subset(train_ds, idxs["train"]), batch_size=eval_batch_size))
    return train_dl, eval_dl, train_eval, info
