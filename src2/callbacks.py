import os, re

import torch
import numpy as np
import matplotlib.pyplot as plt
from composer.core import State, Time, TimeUnit
from composer import Callback, Logger
from torch.utils.data import Subset
from composer.utils import format_name_with_dist

# ***** MaskVizualizer *****

def _make_input_images(inputs: torch.Tensor, num_images: int):
    if inputs.shape[0] < num_images:
        num_images = inputs.shape[0]
    return inputs[:num_images].unsqueeze(-1).detach().cpu().numpy()

class MaskVisualizer(Callback):
    def __init__(self, num_images, train_interval):
        self.num_images = num_images
        self.train_interval = Time.from_input(train_interval, TimeUnit.EPOCH)
        self.last_train_time_value_logged = -1
        self.last_eval_step_logged = {}
    def _log_image(self, state: State, logger: Logger, data_name: str, pad_width: int=1):
        mask_pred, mask_true = state.outputs['mask_pred'], state.batch['mask_true']
        # add white padding between pred and true masks for easier visualization
        padding = torch.ones(mask_pred.shape[0], mask_pred.shape[1], pad_width, device=mask_pred.device)
        image = _make_input_images(torch.cat([mask_pred, padding, mask_true], dim=2), self.num_images)
        logger.log_images(image, name=data_name, channels_last=True, use_table=False)
    def before_loss(self, state: State, logger: Logger):
        current_time_value = state.timestamp.get(self.train_interval.unit).value
        if current_time_value % self.train_interval.value == 0 and current_time_value != self.last_train_time_value_logged:
            self.last_train_time_value_logged = current_time_value
            self._log_image(state, logger, 'Images/train')
    def eval_after_forward(self, state: State, logger: Logger):
        eval_batch = state.eval_timestamp.get(TimeUnit.BATCH).value
        train_step = state.timestamp.batch.value
        evaluator_label = state.dataloader_label or 'eval'
        if eval_batch == 0 and train_step != self.last_eval_step_logged.get(evaluator_label):
            self.last_eval_step_logged[evaluator_label] = train_step
            self._log_image(state, logger, f'Images/{evaluator_label}')

# ***** OutputSaver *****

def _to_cpu(x:torch.Tensor): return x.detach().to('cpu', copy=True)

class OutputSaver(Callback):
    def __init__(self, save_interval, folder, filename='ep{epoch:04d}-ba{batch:06d}.pt'):
        self.save_interval, self.folder, self.filename = Time.from_input(save_interval, TimeUnit.EPOCH), folder, filename
    def init(self, state: State, logger: Logger):
        del logger
        self.folder = format_name_with_dist(self.folder, state.run_name)
        os.makedirs(self.folder, exist_ok=True)

    def save_outputs(self, state: State, logger: Logger, data_name: str):
        current_time_value = state.timestamp.get(self.save_interval.unit).value
        if current_time_value % self.save_interval.value == 0:
            data_name = re.sub(r'[^a-zA-Z0-9_.-]+', '_', data_name)
            path = os.path.join(self.folder, data_name, self.filename.format(epoch=state.timestamp.epoch.value, batch=state.timestamp.batch.value))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            outputs = {'fft': _to_cpu(state.batch['fft']), 'mask_pred': _to_cpu(state.outputs['mask_pred']), 'mask_true': _to_cpu(state.batch['mask_true']), 'info': state.batch['info']}
            logger.log_metrics({os.path.join(data_name, k): v for k, v in outputs.items()})

    def after_forward(self, state, logger): self.save_outputs(state, logger, state.dataloader_label)
    def eval_after_forward(self, state, logger): self.save_outputs(state, logger, f'{state.dataloader_label or "eval"}')


# ***** DataDistribution *****

def compute_distributions(dataset, indices):
    base_dataset = dataset.dataset if isinstance(dataset, Subset) else dataset

    n_x_positions, n_y_positions = len(base_dataset.x_pos_encoder.classes_), len(base_dataset.y_pos_encoder.classes_)
    position_distribution = np.zeros((n_x_positions, n_y_positions), dtype=np.uint16)
    x_positions = base_dataset.x_pos_encoder.transform([base_dataset.ds[idx]["x_position"] for idx in indices])
    y_positions = base_dataset.y_pos_encoder.transform([base_dataset.ds[idx]["y_position"] for idx in indices])
    for x, y in zip(x_positions, y_positions): position_distribution[int(x), int(y)] += 1

    out_h, out_w = base_dataset.masks.shape[1:]
    mask_distribution = np.zeros((out_h, out_w), dtype=np.uint16)
    for idx in indices: mask_distribution += base_dataset.masks[idx].cpu().numpy().astype(np.uint16)

    return position_distribution, mask_distribution

def plot_position_distribution(position_distribution, folder: str, split_name: str, vmax: int, n_samples: int, total_samples: int):
    plt.figure()
    plt.imshow(position_distribution, origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(label="count")
    plt.title(f"Box Position Distribution [{split_name}, n={n_samples}/{total_samples}]")
    for i, row in enumerate(position_distribution):
        for j, value in enumerate(row):
            if value: plt.text(j, i, int(value), ha="center", va="center", color="white")

    path = os.path.join(folder, f"position_distribution_{split_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

def plot_mask_distribution(mask_distribution, folder: str, split_name: str, vmax: int, n_samples: int, total_samples: int):
    plt.figure()
    plt.imshow(mask_distribution, origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    plt.colorbar(label="count")
    plt.title(f"Box Mask Distribution [{split_name}, n={n_samples}/{total_samples}]")

    path = os.path.join(folder, f"mask_distribution_{split_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()

class DataDistribution(Callback):
    def __init__(self, folder): self.folder = folder
    def init(self, state: State, logger: Logger):
        del logger
        self.folder = format_name_with_dist(self.folder, state.run_name)
        os.makedirs(self.folder, exist_ok=True)

    def fit_start(self, state:State, logger:Logger):
        del logger # unused
        dataloaders = {'train': state.train_dataloader} | {evaluator.label.replace('/', '_'): evaluator.dataloader.dataloader for evaluator in state.evaluators}

        # count the number of samples in each position in the box and the number of samples with a mask in each pixel, for each dataloader
        distributions = {}
        for label, dataloader in dataloaders.items():
            subset = dataloader.dataset
            pos_dist, mask_dist = compute_distributions(subset, subset.indices)
            distributions[label] = (pos_dist, mask_dist)

        # compute max values across all distributions for consistent color scaling in the plots
        position_max = max([dist[0].max() for dist in distributions.values()])
        mask_max = max([dist[1].max() for dist in distributions.values()])

        # plot distributions
        total_samples = sum([len(dataloader.dataset.indices) for dataloader in dataloaders.values()])
        for label, dataloader in dataloaders.items():
            subset = dataloader.dataset
            pos_dist, mask_dist = distributions[label]
            n_samples = len(subset.indices)
            plot_position_distribution(pos_dist, self.folder, label, position_max, n_samples, total_samples)
            plot_mask_distribution(mask_dist, self.folder, label, mask_max, n_samples, total_samples)
