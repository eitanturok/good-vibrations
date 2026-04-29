import torch
from composer.core import State, Time, TimeUnit
from composer import Callback, Logger

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
        padding = torch.ones(mask_pred.shape[0], mask_pred.shape[1], 5, device=mask_pred.device)  # add white padding between pred and true masks
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
        if eval_batch == 0 and train_step != self.last_eval_step_logged:
            self.last_eval_step_logged = train_step
            self._log_image(state, logger, 'Images/eval')
