import math
import pytorch_lightning as pl
import torch
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR


class PaintingGan(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 1e-3) -> None:   
        super().__init__()
        self.save_hyperparameters() # instead of self.num_classes etc.

    def configure_optimizers(self):
        gen_adam = Adam(lr=self.hparams.learning_rate, betas= (0.5, 0.999))
        dis_adam = Adam(lr=self.hparams.learning_rate, betas= (0.5, 0.999))
        return [gen_adam, dis_adam] , [get_cosine_schedule_with_warmup(gen_adam, 1000, 100_000), get_cosine_schedule_with_warmup(dis_adam, 1000, 100_000)]
    
    def training_step(self, batch, batch_idx):
        return 


def get_cosine_schedule_with_warmup(
    optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: float = 0.5, last_epoch: int = -1
):
    """
    Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
            The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
            following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training.

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)