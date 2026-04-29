import argparse

import modal
import torch
import wandb
from composer import Trainer
from composer.utils.reproducibility import seed_all
from composer.callbacks import RuntimeEstimator, SpeedMonitor, OOMObserver, NaNMonitor, SystemMetricsMonitor, CheckpointSaver
from composer.loggers import WandBLogger
from huggingface_hub import HfApi

from model import VibrationTransformer
from dataset import build_dataset
from callbacks import MaskVisualizer



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

def get_parser():
    parser = argparse.ArgumentParser()
    # system
    parser.add_argument("--seed",                   type=int,   default=42)
    parser.add_argument("--num-workers",            type=int,   default=4)
    parser.add_argument("--debug",                  type=int,   default=0)
    # data
    parser.add_argument("--n-samples",              type=int,   default=None)
    parser.add_argument("--repo-id",                type=str,   default="eturok-weizmann/laser-vibrations")
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
    parser.add_argument("--load-path",              type=str,   default=None)
    # train
    parser.add_argument("--batch-size",             type=int,   default=128)
    parser.add_argument("--lr",                     type=float, default=1e-4)
    # eval
    parser.add_argument("--eval-batch-size",        type=int,   default=128)
    parser.add_argument("--eval-interval",          type=str,   default="1ep")
    # run
    parser.add_argument("--run-name",               type=str,   default=None)
    # logging
    parser.add_argument("--num-masks-logged",       type=str,   default=16)
    parser.add_argument("--mask-logging-interval",  type=str,   default=None, help="Interval to log masks. If not set, defaults to eval_interval.")
    return parser

@app.function(
    gpu="A10",
    timeout=86_400,  # maximum timeout is 24 hours or 86_400 seconds; see https://modal.com/docs/guide/timeouts#timeouts
    retries=1,
)
def inference(**kwargs):
    # parse args
    args = get_parser().parse_args()  # get defaults
    args.__dict__.update(kwargs)  # apply overrides from cli

    # set device
    device = 'gpu' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f'Using {device=}')

    # set seeds for reproducibility before initializing model + dataloader
    seed_all(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    print(f"Set random seed to {args.seed} for reproducibility")

    # make dataset model
    eval_loader, _, data_info = build_dataset(args.repo_id, args.patch_size, args.out_h, args.out_w, args.batch_size, args.eval_batch_size, args.seed,
                                                         generator, 1, args.num_workers, args.speakers, args.n_objects, args.n_samples)
    model = VibrationTransformer(args.d_model, args.pnt_num_heads, args.pnt_num_layers, args.seq_num_heads, args.seq_num_layers, data_info)

    # make trainer
    config = data_info | args.__dict__ | dict(gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", num_parameters=sum([p_.numel() for p_ in model.parameters()]))
    logger = WandBLogger("laser-vibrations", group="speed", name=args.run_name, init_kwargs={"settings": wandb.Settings(x_disable_stats=False), "config": config, "save_code": True, "id": args.run_name, "resume": "allow"})
    callbacks=[SpeedMonitor(1), OOMObserver(), NaNMonitor(), RuntimeEstimator(time_unit="minutes"), SystemMetricsMonitor(),
               MaskVisualizer(args.num_masks_logged, args.mask_logging_interval or args.eval_interval),
               ]
    trainer = Trainer(run_name=args.run_name, model=model, auto_log_hparams=False,
                    eval_dataloader=eval_loader, max_duration="1ep", seed=args.seed, eval_interval=args.eval_interval,
                    device=device, save_metrics=True, log_to_console=True, progress_bar=False,
                    loggers=logger, callbacks=callbacks, load_path=args.load_path)

    # train da model!!!
    trainer.fit()
    trainer.close()

    # upload results
    api = HfApi()
    api.create_repo(args.repo_id, repo_type="dataset", exist_ok=True)
    api.upload_large_folder(folder_path=f'runs/{trainer.state.run_name}', repo_id=args.repo_id, repo_type="dataset", num_workers=8)


@app.local_entrypoint()
def main(*args):
    inference.remote(**vars(get_parser().parse_args(args)))  # runs on Modal GPU

if __name__ == "__main__":
    inference.local(**vars(get_parser().parse_args()))
