# supress warnings
import warnings, logging
warnings.filterwarnings("ignore", message=r"The pynvml package is deprecated.*", category=FutureWarning)
# only log errors from this file in order to suppress the warning "Redirects are currently not supported in Windows or MacOs."
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import argparse, os, json, math
import torch
import modal
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn import functional as F
from datasets import load_dataset
from huggingface_hub import snapshot_download
from composer.utils.reproducibility import seed_all
from composer.models import ComposerModel
from torchmetrics.regression import MeanSquaredError
from composer.core import State, Time, TimeUnit
from composer import Trainer, Callback, Logger
from composer.profiler import JSONTraceHandler, cyclic_schedule
from composer.profiler.profiler import Profiler
from composer.loggers import WandBLogger
from composer.callbacks import RuntimeEstimator, SpeedMonitor, OOMObserver, NaNMonitor, SystemMetricsMonitor
from icecream import install; install()
import wandb


# **** Modal ****

HF_CACHE_VOL = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"

image = (
    modal.Image.debian_slim()
    .apt_install("git")
    .env({"HF_HUB_CACHE": HF_CACHE_PATH, "HF_XET_HIGH_PERFORMANCE": "1"})
    # .uv_sync()
    .uv_pip_install([
        'ipykernel', 'pip', 'ipywidgets', # for notebooks
        'datasets', 'Pillow', 'torchcodec', 'torch>2.10', 'scikit-learn', 'icecream', 'wandb', 'modal', 'pynvml',
        'psutil', # for wandb to properly log the systems pannels (gpu utilization, gpu memory, etc.)
        'mosaicml-streaming', # for streaming dataset
        ])
    .uv_pip_install("git+https://github.com/eitanturok/composer.git@18440c7")  # the lastest commit on my `hf-object-store` branch from my composer fork
    .add_local_dir("src2", remote_path="/root")
)

app = modal.App(
    image=image,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
)

# ***** Callbacks *****

def _make_input_images(inputs: torch.Tensor, num_images: int):
    if inputs.shape[0] < num_images:
        num_images = inputs.shape[0]
    return inputs[:num_images].unsqueeze(-1).detach().cpu().numpy()

class MaskVisualizer(Callback):
    def __init__(self, num_images, train_interval):
        self.num_images = num_images
        self.train_interval = Time.from_input(train_interval, TimeUnit.EPOCH)
        self.last_train_time_value_logged = -1
        self.last_eval_step_logged = -1
    def _log_image(self, state: State, logger: Logger, data_name: str):
        mask_pred, mask_true = state.outputs['mask_pred'], state.batch['mask_true']
        image = _make_input_images(torch.cat([mask_pred, mask_true], dim=2), self.num_images)
        logger.log_images(image, name=data_name, channels_last=True, use_table=False)
    def before_loss(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.train_interval.unit).value
        if current_time_value % self.train_interval.value == 0 and current_time_value != self.last_train_time_value_logged:
            self.last_train_time_value_logged = current_time_value
            self._log_image(state, logger, 'Images/train')
    def eval_after_forward(self, state: State, logger: Logger):
        eval_batch = state.eval_timestamp.get(TimeUnit.BATCH).value
        train_step = state.timestamp.batch.value
        if eval_batch == 0 and train_step != self.last_eval_step_logged:
            self.last_eval_step_logged = train_step
            self._log_image(state, logger, 'Images/eval')

#***** Dataset *****

class VibrationDataset(Dataset):
    def __init__(self, repo_id:str, patch_size:int, out_h:int, out_w:int, speakers:list[int,str]|list[int]|list[str]|str|None=None, n_objects:list[int]|int|None=None, n_samples:int=None, num_proc:int=8):
        print(f'Downloading dataset {repo_id}...')
        self.ds = load_dataset(repo_id, split="train", num_proc=num_proc) # this is `data/metadata.jsonl`
        self.ds = self.ds.remove_columns(['segmented_overhead_file_name', 'speckle_vibrations_file_name', 'speckle_shifts_ifft_audio_file_name', 'audio_file_name', 'mask_file_name'])
        print(f"Loaded dataset with {len(self.ds)} samples\n")

        # Filter the dataset
        print('Filtering dataset...')
        if speakers is not None:
            self.ds = self.ds.filter(lambda row: row["speakers"] in (str(speakers) if isinstance(speakers, list) else [speakers]), num_proc=num_proc)
            print(f"Filtered dataset to {len(self.ds)} samples with speakers={speakers}")
        if n_objects is not None:
            self.ds = self.ds.filter(lambda row: row["n_objects"] in (n_objects if isinstance(n_objects, list) else [n_objects]), num_proc=num_proc)
            print(f"Filtered dataset to {len(self.ds)} samples with n_objects={n_objects}")
        if n_samples is not None:
            n_samples_ = min(n_samples, len(self.ds))
            self.ds = self.ds.select(range(n_samples_))
            print(f"Selected first {n_samples_} samples from the dataset")
        print(f"Final dataset contains {len(self.ds)} samples\n")

        # Download the segmentation masks and speckle shift FFTs for the filtered dataset
        print('Downloading masks and FFTs...')
        def get_path(paths): return [json.loads(manifest)['artifacts'][paths] for manifest in list(self.ds['manifest'])]
        mask_paths, fft_paths = get_path('mask_npz'), get_path('speckle_shifts_fft')
        print(f"Mask paths: {mask_paths[:5]}...\nFFT paths: {fft_paths[:5]}...")
        snapshot_dir = snapshot_download(repo_id, repo_type="dataset", allow_patterns=set(mask_paths+fft_paths)) # might be duplicate paths for masks
        print(f"Downloaded snapshot to {snapshot_dir}\n")

        # Load the masks and FFTs
        print('Loading masks and FFTs...')
        def load_sample(paths, key): return torch.stack([torch.from_numpy(np.load(os.path.join(snapshot_dir, path))[key]) for path in paths])
        self.masks, self.fft = load_sample(mask_paths, 'mask'), load_sample(fft_paths, 'fft')
        print(f"masks.shape={self.masks.shape}\tmasks.dtype={self.masks.dtype}\nfft.shape={self.fft.shape}\tfft.dtype={self.fft.dtype}\n")

        # discretize masks and cast to float
        print('Discretizing masks...')
        self.masks = F.adaptive_avg_pool2d(self.masks[:, None].float(), (out_h, out_w)).squeeze()
        print(f"masks.shape={self.masks.shape}\tmasks.dtype={self.masks.dtype}\n")

        # normalize and patchify FFTs
        print('Normalizing and patchifying FFTs...')
        # self.fft = self.fft.abs() # take magnitude of FFTs
        # drops entries that do not fully fit into patch_size
        self.fft = self.fft.unfold(2, patch_size, patch_size) # (B,L,F,2) -> (B,L,P,2,PS)
        print(f'fft.shape={self.fft.shape}\t{self.fft.dtype=}\n')

    def __len__(self): return len(self.ds)
    def __getitem__(self, idx): return dict(mask_true=self.masks[idx], fft=self.fft[idx])

def build_dataset(repo_id, patch_size, out_h, out_w, batch_size, eval_batch_size, seed, generator, test_size, num_workers, speakers, n_objects, n_samples):
    dataset = VibrationDataset(repo_id, patch_size, out_h, out_w, speakers, n_objects, n_samples)

    train_indices, eval_indices = train_test_split(np.arange(len(dataset)), test_size=test_size, random_state=seed, shuffle=True)
    # drop_last=True does not seem to speed things up
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=generator, pin_memory=False, persistent_workers=num_workers>0, prefetch_factor=4 if num_workers>0 else None)
    eval_loader = DataLoader(Subset(dataset, eval_indices), batch_size=eval_batch_size, shuffle=False, num_workers=num_workers, generator=generator, pin_memory=False, persistent_workers=num_workers>0, prefetch_factor=4 if num_workers>0 else None)
    print(f"Train dataloader: batch_size={batch_size}, batches={len(train_loader)}, n_samples={len(train_indices)}")
    print(f"Eval dataloader: batch_size={eval_batch_size}, batches={len(eval_loader)}, n_samples={len(eval_indices)}")

    data_info = dict(out_h=dataset.masks.shape[1], out_w=dataset.masks.shape[2], n_freqs=dataset.fft.shape[2] * dataset.fft.shape[4],
                    n_laser_rows=int(math.sqrt(dataset.fft.shape[1])), n_laser_cols=int(math.sqrt(dataset.fft.shape[1])), patch_size=patch_size)
    print(f'{data_info=}')
    return train_loader, eval_loader, data_info

#***** Model *****

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)] / dim))
    freqs = torch.arange(end).unsqueeze(dim=1) * freqs.unsqueeze(dim=0)
    return torch.cat([freqs.cos(), freqs.sin()], dim=-1)

def precompute_freqs_cis_2d(dim: int, h: int, w: int, theta: float = 10000.0) -> torch.Tensor:
    freqs_h, freqs_w = precompute_freqs_cis(dim // 2, h, theta), precompute_freqs_cis(dim // 2, w, theta),
    freqs_h, freqs_w = freqs_h.reshape(h, 1, -1).repeat(1, w, 1), freqs_w.reshape(1, w, -1).repeat(h, 1, 1)
    return torch.cat([freqs_h, freqs_w], dim=-1).reshape(h * w, dim)

def apply_rope(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    assert x.shape[-1] % 2 == 0
    shp = [1] * (x.ndim - 2) + [x.shape[1], -1]  # works with 1D + 2D rope
    cos, sin = freqs_cis.reshape(*shp).chunk(2, dim=-1)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

class MLPDecoder(nn.Module):
    def __init__(self, d_model, out_h, out_w):
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.net = nn.Sequential(nn.Linear(d_model, 256), nn.ReLU(), nn.Linear(256, out_h * out_w))
    def forward(self, cls): return self.net(cls).view(-1, self.out_h, self.out_w)

raw_to_tokens = {"magnitude": lambda t: t.abs(), "complex": lambda t: torch.cat([t.real, t.imag], dim=-1), "mag_phase": lambda t: torch.cat([t.abs(), t.angle()], dim=-1)}

class LaserEncoder(nn.Module):
    def __init__(self, patch_size, d_model, num_heads, num_layers, signal_length, signal):
        super().__init__()
        self.embed = nn.Linear(2 * patch_size, d_model)
        self.layers = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, batch_first=True), num_layers=num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.register_buffer("freqs_cis", precompute_freqs_cis(d_model, signal_length // patch_size))
        self.raw_to_tokens = raw_to_tokens[signal]

    def forward(self, x):
        # x.shape = (B_L,P,C,_PS) = (batch_size * n_lasers, n_patches, n_coords, patch_size)
        B_L, P, _, _ = x.shape
        x = self.raw_to_tokens(x).float()  # (B_L,P,C,_PS) -> (B_L,P,C,PS) where PS=_PS or 2*_PS
        x = self.embed(x.reshape(B_L, P, -1))  # (B_L,P,C,PS) -> (B_L,P,D)
        x = apply_rope(x, self.freqs_cis) # (B_L,P,D) -> (B_L,P,D)
        x = torch.cat((self.cls_token.expand(B_L, -1, -1), x), dim=1) # (B_L,P,D) -> (B_L,P+1,D)
        output = self.layers(x)  # (B_L,P+1,D) -> (B_L,P+1,D)
        return output[:, 0, :]  # (B_L,P+1,D) -> (B_L,D)

def create_metrics(): return {"mse": MeanSquaredError()}

class VibrationTransformer(ComposerModel):
    def __init__(self, d_model, pnt_num_heads, pnt_num_layers, seq_num_heads, seq_num_layers, data_info, signal='magnitude'):
        super().__init__()

        # encoder
        self.laser_encoder = LaserEncoder(data_info['patch_size'], d_model, pnt_num_heads, pnt_num_layers, data_info["n_freqs"], signal)
        self.box_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer( d_model=d_model, nhead=seq_num_heads, batch_first=True), num_layers=seq_num_layers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)  # Initialize to small random values
        self.register_buffer("freqs_laser", precompute_freqs_cis_2d(d_model, data_info['n_laser_rows'], data_info['n_laser_cols'])) # for laser grid

        # decoder
        self.decoder = MLPDecoder(d_model, data_info['out_h'], data_info['out_w'])

        # metrics
        self.train_metrics, self.val_metrics = create_metrics(), create_metrics()

    def forward(self, batch):
        # B=batch size, L=n_lasers, C=n_coordinates=2, PS=patch_size, D=d_model
        x = batch['fft']
        B, L, _, _, _ = x.shape

        # LaserEncoder learns patterns between all frequencies from a single laser
        # flatten so LaserEncoder processes all lasers AND all batches in parallel
        x = self.laser_encoder(x.flatten(0, 1)).reshape(B, L, -1)  # (B,L,P,C,PS) -> (B,L,D)

        # BoxEncoder learns patterns between ALL the lasers shining on the box
        x = apply_rope(x, self.freqs_laser) # (B,L,D) -> (B,L,D)
        x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)  # (B,L,D) (1,1,D) -> (B,L+1,D)
        output = self.box_encoder(x)  # (B,L+1,D) -> (B,L+1,D)
        cls_embedding = output[:, 0, :]  # (B,L+1,D) -> (B,D)

        # Predict segmentation mask
        mask_logits = self.decoder(cls_embedding) # (B,D) -> (B,H,W)
        mask_pred = mask_logits.sigmoid()
        return dict(mask_pred=mask_pred)

    def loss(self, outputs, batch):
        # mse is averaged over (B,H,W) so the error is independent of the height and width, making it stable across different out_h / out_w
        return F.mse_loss(outputs['mask_pred'], batch['mask_true'])

    def get_metrics(self, is_train=False):
        return self.train_metrics if is_train else self.val_metrics

    def update_metric(self, batch, outputs, metric):
        if isinstance(metric, MeanSquaredError):
            metric.update(outputs['mask_pred'], batch['mask_true'])

    def eval_forward(self, batch, outputs=None):
        return outputs if outputs is not None else self.forward(batch)

def get_parser():
    parser = argparse.ArgumentParser()
    # system
    parser.add_argument("--seed",                   type=int,   default=42)
    parser.add_argument("--num-workers",            type=int,   default=4)
    parser.add_argument("--debug",                  type=int,   default=0)
    # data
    parser.add_argument("--n-samples",              type=int,   default=None)
    parser.add_argument("--repo-id",                type=str,   default="eturok-weizmann/laser-vibrations")
    parser.add_argument("--test-size",              type=float, default=0.2)
    parser.add_argument("--out-h",                  type=int,   default=40)
    parser.add_argument("--out-w",                  type=int,   default=20)
    parser.add_argument("--patch-size",             type=int,   default=256)
    parser.add_argument("--speakers",               type=int,   default=None)
    parser.add_argument("--n-objects",              type=int,   default=None)
    # model
    parser.add_argument("--d-model",                type=int,   default=128)
    parser.add_argument("--pnt-num-heads",          type=int,   default=2)
    parser.add_argument("--seq-num-heads",          type=int,   default=2)
    parser.add_argument("--pnt-num-layers",         type=int,   default=2)
    parser.add_argument("--seq-num-layers",         type=int,   default=2)
    # train
    parser.add_argument("--batch-size",             type=int,   default=128)
    parser.add_argument("--lr",                     type=float, default=1e-4)
    parser.add_argument("--max-duration",           type=str,   default="20ep")
    # eval
    parser.add_argument("--eval-batch-size",        type=int,   default=128)
    parser.add_argument("--eval-interval",          type=str,   default="5ep")
    # run
    parser.add_argument("--run-name",               type=str,   default=None)
    # logging
    parser.add_argument("--num-masks-logged",       type=str,   default=8)
    parser.add_argument("--mask-logging-interval",  type=str,   default=None, help="Interval to log masks. If not set, defaults to eval_interval.")
    return parser

@app.function(
    gpu="A10",
    timeout=86_400,  # maximum timeout is 24 hours or 86_400 seconds; see https://modal.com/docs/guide/timeouts#timeouts
    retries=1,
)
def train(**kwargs):

    # parse args
    args = get_parser().parse_args()  # get defaults
    args.__dict__.update(kwargs)  # apply overrides from cli
    # torch.backends.cudnn.benchmark = torch.backends.cuda.benchmark = True # find faster convolution kernels

    # set device
    device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using {device=}')

    # set seeds for reproducibility before initializing model + dataloader
    seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    print(f"Set random seed to {args.seed} for reproducibility")

    # make dataset, model, optimizer
    train_loader, eval_loader, data_info = build_dataset(args.repo_id, args.patch_size, args.out_h, args.out_w, args.batch_size, args.eval_batch_size, args.seed,
                                                         generator, args.test_size, args.num_workers, args.speakers, args.n_objects, args.n_samples)
    model = VibrationTransformer(args.d_model, args.pnt_num_heads, args.pnt_num_layers, args.seq_num_heads, args.seq_num_layers, data_info)
    optimizer = torch.optim.Adam(model.parameters(), args.lr, fused=True)

    # make trainer
    config = data_info | args.__dict__ | dict(gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", num_parameters=sum([p_.numel() for p_ in model.parameters()]))
    logger = WandBLogger("laser-vibrations", group="speed", name=args.run_name, init_kwargs={"settings": wandb.Settings(x_disable_stats=False), "config": config, "save_code": True, "id": args.run_name, "resume": "allow"})
    profiler = Profiler(
        trace_handlers=[JSONTraceHandler(folder=f"runs/{{run_name}}/composer_profiler",
                                         merged_trace_filename=f"runs/{{run_name}}/composer_profiler/merged_trace_node{{node_rank}}.json",
                                        #  remote_file_name=f"hf://{args.repo_id}/runs/{{run_name}}/composer_profiler/ep{{epoch}}-ba{{batch}}-rank{{rank}}.json",
                                        #  merged_trace_remote_file_name=f"hf://{args.repo_id}/runs/{{run_name}}/composer_profiler/merged_trace_node{{node_rank}}.json",
                                         overwrite=True)],
        schedule=cyclic_schedule(wait=0, warmup=0, active=1, repeat=1),
        torch_prof_folder=f"runs/{{run_name}}/torch_profiler", torch_prof_overwrite=True, torch_prof_memory_filename=None,
        # torch_prof_remote_file_name=f"hf://{args.repo_id}/runs/{{run_name}}/torch_profiler/rank{{rank}}.{{batch}}.pt.trace.json")
    callbacks=[SpeedMonitor(1), OOMObserver(), NaNMonitor(), RuntimeEstimator(time_unit="minutes"), SystemMetricsMonitor(), MaskVisualizer(args.num_masks_logged, args.mask_logging_interval or args.eval_interval)]
    trainer = Trainer(run_name=args.run_name, model=model, optimizers=optimizer, train_dataloader=train_loader, auto_log_hparams=False,
                    eval_dataloader=eval_loader, max_duration=args.max_duration, seed=args.seed, eval_interval=args.eval_interval,
                    device=device, save_metrics=True, log_to_console=True, progress_bar=False,
                    loggers=logger, callbacks=callbacks, profiler=profiler if args.debug > 0 else None)

    # train da model!!!
    trainer.fit()
    trainer.close()


@app.local_entrypoint()
def main(*args):
    train.remote(**vars(get_parser().parse_args(args)))  # runs on Modal GPU

if __name__ == "__main__":
    train.local(**vars(get_parser().parse_args()))
