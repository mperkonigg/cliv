from typing import Callable

import lightning as L
import torch


class ClivLightningModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | dict,
        loss: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        metrics: list[torch.nn.Module | Callable] = [],
        seed: int=-1,
    ):
        """Lightning module for cliv

        Args:
            model (torch.nn.Module): model
            optimizer (torch.optim.Optimizer | dict): optimizer used for training
            loss (torch.nn.Module): _description_
            scheduler (torch.optim.lr_scheduler.LRScheduler, optional): _description_. Defaults to None.
            metrics (list[torch.nn.Module, Callable], optional): _description_. Defaults to [].
            seed (int, optional): _description_. Defaults to -1.
        """
        super().__init__()

        self.model = model
        self.loss = loss
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.metrics = metrics

        if seed!=-1:
            torch.manual_seed(seed)

    def forward(self, x):
        return self.model(x)

    def step(self, batch, step_name: str):
        x, y, _, ann_ids = batch

        loss, y_hat = self.model.train_step(x, y, self.loss, ann_ids)

        self.log(f"{step_name}_loss", loss)

        for metric in self.metrics:
            metric_name = metric.__class__.__name__
            result = metric(y_hat, y)
            self.log(f"{step_name}_{metric_name}", result)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, step_name="train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, step_name="val")

    def test_step(self, batch, batch_idx):
        return self.step(batch, step_name="test")

    def configure_optimizers(self):
        if type(self.optimizer) is dict:
            if self.optimizer['opt_class'] == 'adam':
                opt_params = [
                    {'params': self.model.unet.parameters()},
                    {'params': self.model.head.parameters()},
                    {'params': self.model.z.parameters(
                    ), 'lr': self.optimizer['z_learning_rate']}
                ]

                self.optimizer = torch.optim.AdamW(
                    opt_params, lr=self.optimizer['lr'], weight_decay=0.1)

        if self.scheduler:
            return [self.optimizer], [self.scheduler]
        return self.optimizer
