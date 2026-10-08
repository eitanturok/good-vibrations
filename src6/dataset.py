import atexit, json, random, shutil, tempfile, textwrap
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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

def process_x(samples: dict[Path, dict], signal: str, norm: str, patch_size: int | None):
    """fft -> (L,P,patch_size,C) tokens, or (L,F,C) unpatched if patch_size is None"""
    # process lazily since FFTs are large (10GB+ for all samples)
    for d in tqdm(samples, desc="processing x"):
        x = load_fft(d).abs()                                                 # (L,F,C) complex -> (L,F,C) |fft|
        x = torch.log(x + 1e-3) if signal == "log_magnitude" else x           # (L,F,C); 1e-3 floors the dead bins (notebook 65)
        x = (x - x.mean()) / x.std() if norm == "z" else x / x.std()          # (L,F,C), stats over the whole sample
        if patch_size is None: yield x.numpy(); continue                      # (L,F,C), no patching
        (L, F_, C), P = x.shape, -(-x.shape[1] // patch_size)                 # P = ceil(F / patch_size)
        x = F.pad(x, (0, 0, 0, P * patch_size - F_))                          # (L,P*patch_size,C); zero-pad, truncating drops the top freqs
        yield x.reshape(L, P, patch_size, C).numpy()                          # (L,P,patch_size,C)

#*** 3 process y

def process_y(samples: dict[Path, dict], objects: list[str], max_objects: int, out_h: int, out_w: int, rgb: bool) -> list[dict]:
    """downsample mask image, map object to index, map coms to tensor"""
    ys = []
    for d, meta in tqdm(samples.items(), desc="processing y"):
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
        ys.append(dict(y=y, object=obj, coms=np.array(coms + [[0.0, 0.0]] * (max_objects - len(coms)), dtype=np.float32)))
    return ys

#*** 4 datasets

def split(layouts, eval_frac, speakers=None, keep=lambda position_id: True):  # speakers=None -> all speakers
    return dict(layouts=layouts, eval_frac=eval_frac, speakers=speakers, keep=keep)

# speakers=None trains and evals on every speaker and holds none out: the unseen-speaker splits go, and their
# positions become one more held-out-positions split, so every other split keeps the same eval positions
def gastronorm(test_size=0.2, speakers=(1, 3, 7)) -> dict[str, dict]:
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
        "cube-lid":                     split(["red-cube-lid"], 1.0, speakers),
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

def plastic(test_size=0.2, speakers=(1, 3, 7)) -> dict[str, dict]:
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
        "cube-lid":                       split(["red-cube-lid"], 1.0, speakers),
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

def gastronorm_plastic(test_size=0.2, speakers=(1, 3, 7)) -> dict[str, dict]: return gastronorm(test_size, speakers) | plastic(test_size, speakers)

DATASETS = {"gastronorm": gastronorm, "plastic": plastic, "gastronorm_plastic": gastronorm_plastic}

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
            table.append([len(table), name, n, n - n_eval, n_eval, round(100 * split["eval_frac"]), split["speakers"] or "all", wrap(split["layouts"])])
        table.append(total("total", lambda i: metas[i]["experiment"] == exp))
        print(f"\n{exp}\n" + tabulate(table, headers, tablefmt="simple_grid"))
    table = [[k, *total(exp, lambda i, exp=exp: metas[i]["experiment"] == exp)[1:]] for k, exp in enumerate(exps)]
    table += [total("total", lambda i: True)]
    print("\nall experiments\n" + tabulate(table, ["#", "experiment", *headers[2:-1]], tablefmt="simple_grid"))

#*** 6 cast to mds

def write_mds(mds_dir: Path, metas: list[dict], Xs, ys: list[dict]) -> None:
    """Xs, ys: each sample's processed input and targets, in the same order as metas; row i of the MDS is metas[i]"""
    shutil.rmtree(mds_dir, ignore_errors=True)
    columns = {"X": "ndarray:float32", "y": "ndarray:float32", "object": "int", "coms": "ndarray:float32", "experiment": "str", "position_id": "int"}
    with MDSWriter(out=str(mds_dir), columns=columns, size_limit="200MB") as w:
        for meta, X, y in zip(metas, Xs, ys):
            w.write({"X": X, **y, "experiment": meta["experiment"], "position_id": int(meta["position_id"])})
    print(f"wrote {len(metas)} samples to {mds_dir} ({sum(f.stat().st_size for f in mds_dir.rglob('*')) / 1e9:.1f} GB)")

#*** 7 data info

def get_data_info(samples: dict[Path, dict], train_idxs: list[int]) -> dict:
    """what the model and losses need to know about the data"""
    metas = list(samples.values())
    _, n_freqs, n_channels = load_fft(next(iter(samples))).shape  # (L,F,C); n_freqs = real bins, before tokenize pads to whole patches
    return dict(n_laser_rows=metas[0]["n_rows"], n_laser_cols=metas[0]["n_cols"], n_freqs=n_freqs, n_channels=n_channels,
                objects=sorted({"+".join(sorted(metas[i]["objects"])) for i in train_idxs}) + ["unseen"],  # object combos seen in train, + "unseen" for any other
                n_objects=sorted({meta["n_objects"] for meta in metas}))                     # every object count in the data, train and eval

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

        return dict(fft=torch.from_numpy(row["X"].copy()), mask_true=torch.from_numpy(row["y"].copy()),
                    object=row["object"], coms=torch.from_numpy(row["coms"].copy()),
                    experiment=meta["experiment"], position_id=meta["position_id"], speaker=meta["speaker"],
                    sample_id=meta["sample_id"], n_objects=meta["n_objects"], layout=meta["layout"],
                    crop_h=meta["crop_overhead_shape"][0], crop_w=meta["crop_overhead_shape"][1])  # full-res box size, so viz can draw the real aspect

#*** 9 build dataset

def build_dataset(dataset: str = "gastronorm_plastic", test_size: float = 0.2, speakers=(1, 3, 7),
                  signal: str = "magnitude", norm: str = "std", patch_size: int | None = None,
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
    Xs = process_x(samples, signal, norm, patch_size)

    # process outputs: downsample mask, map object to index, map coms to tensor
    data_info = get_data_info(samples, idxs["train"])
    ys = process_y(samples, data_info["objects"], max(data_info["n_objects"]), out_h, out_w, rgb)

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
