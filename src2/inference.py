# supress warnings
import warnings, logging
warnings.filterwarnings("ignore", message=r"The pynvml package is deprecated.*", category=FutureWarning)
# only log errors from this file in order to suppress the warning "Redirects are currently not supported in Windows or MacOs."
logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)

import os

import modal
import torch 
import wandb
from composer.utils.reproducibility import seed_all
from composer.callbacks import RuntimeEstimator, SpeedMonitor, OOMObserver, NaNMonitor, SystemMetricsMonitor, CheckpointSaver
from huggingface_hub import HfApi
from composer import Trainer
from composer.loggers import WandBLogger

from train import get_parser
from callbacks import MaskVisualizer, OutputSaver
from dataset import build_dataset
from model import VibrationTransformer

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
    .uv_pip_install("git+https://github.com/eitanturok/composer.git@4519dd2")  # the lastest commit on my `hf-object-store` branch from my composer fork
    .add_local_dir("src2", remote_path="/root")
)

app = modal.App(
    image=image,
    volumes={HF_CACHE_PATH: HF_CACHE_VOL},
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
)


@app.function(
    gpu="A10",
    timeout=86_400,  # maximum timeout is 24 hours or 86_400 seconds; see https://modal.com/docs/guide/timeouts#timeouts
    retries=0,
)
def inference(**kwargs):

    # parse args
    parser = get_parser()
    parser.add_argument("--load-path", type=str, default="runs/1777479603-unique-guan/checkpoints/ep10-ba10-rank0.pt")
    args = parser.parse_args()
    args.__dict__.update(kwargs)  # apply overrides from cli

    # apply eval overrides
    args.max_duration = "1ep"
    args.eval_interval = "1ep"
    args.save_output_interval = "1ep"
    args.checkpoint_inveral = "9999999999ep"  # effectively disable checkpointing during inference

    # download model checkpoint from HuggingFace Hub
    print(f"Downloading model checkpoint from {args.load_path}...")
    api = HfApi()
    local_dir = os.path.expanduser("~/good-vibrations")
    snapshot_dir = api.snapshot_download(args.remote_checkpoint_folder, repo_type='dataset', max_workers=8, local_dir=local_dir, allow_patterns=args.load_path)
    load_path = os.path.join(snapshot_dir, args.load_path)
    print(f"Downloaded checkpoint to {load_path}")

    # set device
    device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using {device=}')

    # set seeds for reproducibility before initializing model + dataloader
    seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    print(f"Set random seed to {args.seed} for reproducibility")

    # make dataset, model, optimizer
    train_loader, eval_loader, data_info = build_dataset(args.repo_id, args.patch_size, args.out_h, args.out_w, args.batch_size, args.eval_batch_size, args.seed,
                                                         generator, args.test_size, args.num_workers, args.speakers, args.n_objects, args.n_samples, args.num_proc, args.dry_run)
    model = VibrationTransformer(args.d_model, args.pnt_num_heads, args.pnt_num_layers, args.seq_num_heads, args.seq_num_layers, data_info)
    optimizer = torch.optim.Adam(model.parameters(), 1e-16, fused=True)

    # make trainer
    config = data_info | args.__dict__ | dict(gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", num_parameters=sum([p_.numel() for p_ in model.parameters()]))
    logger = WandBLogger("laser-vibrations", group="speed", name=args.run_name, init_kwargs={"settings": wandb.Settings(x_disable_stats=False), "config": config, "save_code": True, "id": args.run_name, "resume": "allow"})
    callbacks=[SpeedMonitor(1), OOMObserver(), NaNMonitor(), RuntimeEstimator(time_unit="minutes"), SystemMetricsMonitor(),
               MaskVisualizer(args.num_masks_logged, args.mask_logging_interval or args.eval_interval),
            #    ExportForInferenceCallback(save_format='torchscript',save_path='runs/{{run_name}}/model.pth'),
               CheckpointSaver(folder=f"runs/{{run_name}}/runs/{{run_name}}/checkpoints", weights_only=True, overwrite=True, save_interval=args.checkpoint_interval),
               OutputSaver(folder=f"runs/{{run_name}}/runs/{{run_name}}/forward_outputs", overwrite=True, shard_size=64, save_interval=args.save_output_interval)
               ]
    trainer = Trainer(run_name=args.run_name, model=model, optimizers=optimizer, train_dataloader=train_loader, auto_log_hparams=False,
                    eval_dataloader=eval_loader, max_duration=args.max_duration, seed=args.seed, eval_interval=args.eval_interval,
                    device=device, save_metrics=True, log_to_console=True, progress_bar=False,
                    loggers=logger, callbacks=callbacks, load_path=load_path, load_weights_only=True)

    # train da model!!!
    trainer.fit()

    # close up shop!!
    trainer.close()

    # upload results
    api.create_repo(args.remote_checkpoint_folder, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(folder_path=f'runs/{trainer.state.run_name}', repo_id=args.remote_checkpoint_folder, repo_type="dataset", num_workers=8)


@app.local_entrypoint()
def main(*args):
    inference.remote(**vars(get_parser().parse_args(args)))  # runs on Modal GPU

if __name__ == "__main__":
    inference.local(**vars(get_parser().parse_args()))
