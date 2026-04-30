import os, re
from collections import defaultdict

import torch
from composer.core import Event, State, Time, TimeUnit
from composer import Callback, Logger
from composer.utils import dist
from composer.utils import ensure_folder_is_empty, format_name_with_dist

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

def load_forward_outputs(path: str):
    paths = [path] if os.path.isfile(path) else sorted(os.path.join(root, name) for root, _, files in os.walk(path) for name in files if name.endswith('.pt'))
    rows = [row for p in paths for row in torch.load(p, map_location='cpu', weights_only=False)]
    return sorted(rows, key=lambda row: (row['split'], row['label'], row['timestamp']['epoch'], row['timestamp']['batch'], row['forward_idx']))

def get_forward_outputs(path: str, split=None, label=None, epoch=None, batch=None, eval_batch=None, forward_idx=None):
    rows = load_forward_outputs(path)
    if split is not None: rows = [row for row in rows if row['split'] == split]
    if label is not None: rows = [row for row in rows if row['label'] == label]
    if epoch is not None: rows = [row for row in rows if row['timestamp']['epoch'] == epoch]
    if batch is not None: rows = [row for row in rows if row['timestamp']['batch'] == batch]
    if eval_batch is not None: rows = [row for row in rows if row['eval_timestamp']['batch'] == eval_batch]
    if forward_idx is not None: rows = [row for row in rows if row['forward_idx'] == forward_idx]
    return rows

# def _shape(x):
#     if isinstance(x, torch.Tensor): return tuple(x.shape)
#     if isinstance(x, dict): return {k: _shape(v) for k, v in x.items()}
#     if isinstance(x, tuple): return tuple(_shape(v) for v in x)
#     if isinstance(x, list): return [_shape(v) for v in x]
#     return None

# class OutputSaver(Callback):
#     def __init__(self, folder='runs/{run_name}/forward_outputs', filename='ep{epoch:04d}-ba{batch:06d}/{split}.{label}.shard{shard_idx:05d}-rank{rank}.pt', shard_size=32, save_interval='1ep', save_train=True, save_eval=True, save_predict=False, overwrite=False):
#         self.folder = folder
#         self.filename = filename
#         self.shard_size = shard_size
#         self.save_interval = Time.from_input(save_interval, TimeUnit.EPOCH)
#         self.splits = {Event.AFTER_FORWARD: save_train, Event.EVAL_AFTER_FORWARD: save_eval, Event.PREDICT_AFTER_FORWARD: save_predict}
#         self.overwrite = overwrite
#         self.path = None
#         self.buffers = defaultdict(list)
#         self.forward_idxs = defaultdict(int)
#         self.shard_idxs = defaultdict(int)
#         self.last_saved = defaultdict(lambda: -1)
#         self.enabled = defaultdict(bool)

#     def init(self, state: State, logger: Logger):
#         del logger
#         self.path = format_name_with_dist(self.folder, state.run_name)
#         os.makedirs(self.path, exist_ok=True)
#         if not self.overwrite:
#             ensure_folder_is_empty(self.path)

#     def run_event(self, event: Event, state: State, logger: Logger):
#         if event == Event.INIT: return self.init(state, logger)
#         if event == Event.BATCH_START: return self._start(state, 'train', 'train')
#         if event == Event.EVAL_START: return self._start(state, 'eval', state.dataloader_label or 'eval')
#         if event == Event.PREDICT_START: return self._start(state, 'predict', 'predict')
#         split = {Event.AFTER_FORWARD: 'train', Event.EVAL_AFTER_FORWARD: 'eval', Event.PREDICT_AFTER_FORWARD: 'predict'}.get(event)
#         label = state.dataloader_label or split
#         if split is not None and self.splits[event] and self.enabled[label]:
#             self._record(state, split, label)
#         elif event == Event.EVAL_END:
#             self._flush(f"eval:{state.dataloader_label or 'eval'}")
#         elif event == Event.PREDICT_END:
#             self._flush('predict:predict')
#         elif event == Event.FIT_END:
#             self._flush('train:train')
#         elif event == Event.EVAL_STANDALONE_END:
#             for key in list(self.buffers):
#                 self._flush(key)

#     def state_dict(self):
#         return {'forward_idxs': dict(self.forward_idxs), 'shard_idxs': dict(self.shard_idxs)}

#     def load_state_dict(self, state):
#         self.forward_idxs.update(state.get('forward_idxs', {}))
#         self.shard_idxs.update(state.get('shard_idxs', {}))

#     def _start(self, state: State, split: str, label: str):
#         t = state.timestamp.get(self.save_interval.unit).value
#         if self.save_interval.unit == TimeUnit.EPOCH and split == 'train': t += 1
#         self.enabled[label] = t % self.save_interval.value == 0 and t != self.last_saved[label]
#         if self.enabled[label]:
#             self._flush(f'{split}:{label}')
#             self.last_saved[label] = t

#     def _record(self, state: State, split: str, label: str):
#         key = f'{split}:{label}'
#         self.forward_idxs[key] += 1
#         self.buffers[key].append({
#             'split': split,
#             'label': label,
#             'forward_idx': self.forward_idxs[key],
#             'rank': dist.get_global_rank(),
#             'timestamp': state.timestamp.state_dict(),
#             'eval_timestamp': state.eval_timestamp.state_dict(),
#             'output_shape': _shape(state.outputs),
#             'outputs': self._to_cpu(state.outputs),
#         })
#         if len(self.buffers[key]) >= self.shard_size:
#             self._flush(key)

#     def _flush(self, key):
#         assert self.path is not None, 'OutputSaver.init() must run before saving outputs'
#         if not self.buffers[key]:
#             return
#         split, label = key.split(':', 1)
#         record = self.buffers[key][-1]
#         timestamp = record['timestamp']
#         label = re.sub(r'[^a-zA-Z0-9_.-]+', '_', label)
#         path = os.path.join(self.path, self.filename.format(split=split, label=label, shard_idx=self.shard_idxs[key], rank=dist.get_global_rank(), epoch=timestamp['epoch'], batch=timestamp['batch']))
#         os.makedirs(os.path.dirname(path), exist_ok=True)
#         torch.save(self.buffers[key], path)
#         self.shard_idxs[key] += 1
#         self.buffers[key].clear()

#     def _to_cpu(self, x):
#         if isinstance(x, torch.Tensor): return x.detach().to('cpu', copy=True)
#         if isinstance(x, dict): return {k: self._to_cpu(v) for k, v in x.items()}
#         if isinstance(x, tuple): return tuple(self._to_cpu(v) for v in x)
#         if isinstance(x, list): return [self._to_cpu(v) for v in x]
#         return x


def _to_cpu(x:torch.Tensor): return x.detach().to('cpu', copy=True)

class OutputSaver(Callback):
    def __init__(self, save_interval, folder, filename='ep{epoch:04d}-ba{batch:06d}.pt'):
        self.save_interval, self.folder, self.filename = Time.from_input(save_interval, TimeUnit.EPOCH), folder, filename
    def init(self, state: State, logger: Logger):
        del logger
        self.folder = format_name_with_dist(self.folder, state.run_name)
        os.makedirs(self.folder, exist_ok=True)

    def save_outputs(self, state: State, logger: Logger, data_name: str):
        del logger
        if state.timestamp.epoch.value % self.save_interval == 0:
            data_name = re.sub(r'[^a-zA-Z0-9_.-]+', '_', data_name)
            path = os.path.join(self.folder, data_name, self.filename.format(epoch=state.timestamp.epoch.value, batch=state.timestamp.batch.value))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            outputs = {'fft': _to_cpu(state.batch['fft']), 'outputs': _to_cpu(state.outputs), 'mask_true': _to_cpu(state.batch['mask_true']), 'info': state.batch['info']}
            torch.save(outputs, path)

    def after_forward(self, state, logger): self.save_outputs(state, logger, 'train')
    def eval_after_forward(self, state, logger): self.save_outputs(state, logger, f'eval.{state.dataloader_label or "eval"}')
