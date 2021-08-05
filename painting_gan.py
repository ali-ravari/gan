import math
from pickle import TUPLE
from typing import Tuple
import pytorch_lightning as pl
import torch
from torch.functional import Tensor
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from generator import Generator
from torchvision.models import resnet50
from torch import nn
import torch.nn.functional as F


class PaintingGan(pl.LightningModule):
    def __init__(self, num_classes: int, learning_rate: float = 1e-3, latent_dim: int= 100) -> None:   
        super().__init__()
        self.save_hyperparameters() # instead of self.num_classes etc.
        self.generator = Generator(latent_dim)
        self.discriminator = resnet50(pretrained=True)
        self.discriminator.fc = nn.Linear(self.discriminator.fc.in_features, 1)


    def configure_optimizers(self):
        gen_adam = Adam(self.generator.parameters(), lr=self.hparams.learning_rate, betas= (0.5, 0.999))
        dis_adam = Adam(self.discriminator.parameters(), lr=self.hparams.learning_rate, betas= (0.5, 0.999))
        return [gen_adam, dis_adam] , [get_cosine_schedule_with_warmup(gen_adam, 1000, 100_000), get_cosine_schedule_with_warmup(dis_adam, 1000, 100_000)]
    
    def get_progress_bar_dict(self):
        default = super().get_progress_bar_dict()
        del default['loss']
        return default

    def shared_step(self, batch: Tuple[Tensor, Tensor], optimizer_idx: int, log_prefix: str):
        reals, _ = batch
        # sample noise
        z = torch.randn(reals.shape[0], self.hparams.latent_dim, 1, 1, device=self.device)
        fakes = self.generator(z)
        if optimizer_idx == 1: # training discriminator
            fakes = fakes.detach()

        discr_input = torch.cat([reals, fakes])
        discr_labels = torch.cat([torch.ones(reals.shape[0], device=self.device), torch.zeros(fakes.shape[0], device=self.device)])
        discr_preds = self.discriminator(discr_input)

        discr_loss = F.binary_cross_entropy_with_logits(discr_preds.squeeze(), discr_labels)
        gen_loss = F.binary_cross_entropy_with_logits(discr_preds.squeeze(), 1 - discr_labels)

        self.log(log_prefix + 'discr_loss', discr_loss, prog_bar=True)
        self.log(log_prefix + 'gen_loss', gen_loss, prog_bar=True)
        loss = gen_loss if optimizer_idx == 0 else discr_loss
        return loss
        #return {"loss":loss, "progress_bar": {"g_loss":-discr_loss}}
    
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, optimizer_idx: int):
        return self.shared_step(batch, optimizer_idx, log_prefix='train_')
    
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int):
        self.shared_step(batch, optimizer_idx=0, log_prefix='val_')

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