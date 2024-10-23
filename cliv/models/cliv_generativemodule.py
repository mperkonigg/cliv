from typing import Callable

import lightning as L
import torch
import random


class ClivGenerativeModule(L.LightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        gen_model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | dict,
        loss: torch.nn.Module,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        metrics: list[torch.nn.Module | Callable] = [],
        train_annotators: list[str] = None,
        fix_prev_zvalues: bool = True,
        replay_samples: int = -1,
        seed: int=-1,
        lambda_gen: int=1,
    ):
        """Lightning module for the generative model

        Args:
            model (torch.nn.Module): Model
            gen_model (torch.nn.Module): Generative model (usually model from the previous round)
            optimizer (torch.optim.Optimizer | dict): optimizer for training
            loss (torch.nn.Module): loss for training
            scheduler (torch.optim.lr_scheduler.LRScheduler, optional): _description_. Defaults to None.
            metrics (list[torch.nn.Module | Callable], optional): _description_. Defaults to [].
            train_annotators (list[str], optional): Annotators that are used for training. Defaults to None.
            fix_prev_zvalues (bool, optional): wheter or not the fix the annotation style parameter of previous readers. Defaults to True.
            replay_samples (int, optional): number of samples to replay If -1 replay whole batch for all prev readers. Defaults to -1.
            seed (int, optional): seed of the run. Defaults to -1.
        """
        super().__init__()

        self.model = model

        if fix_prev_zvalues:  # fix the posteriors of previous distributions to not be influenced by generative replay
            for par in self.model.z.z_vecs[:-1]:
                par.requires_grad = False

        self.gen_model = gen_model
        self.gen_model.eval()
        self.loss = loss
        self.scheduler = scheduler
        self.optimizer = optimizer
        self.metrics = metrics
        self.train_annotators = train_annotators
        self.replay_samples = replay_samples
        self.lambda_gen = lambda_gen

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

    def replay_step(self, batch, step_name: str):
        x, y, _, ann_ids = batch
        loss, y_hat = self.model.train_step(
            x, y, self.loss, ann_ids, replay_step=True)

        self.log(f"{step_name}_loss", loss)

        return loss

    def training_step(self, batch, batch_idx):
        # manipulate batch to add generative input
        x, y, ids, ann_ids = batch

        samples = []
        imgs = []
        ann_ids_gen = []

        ann_ids = self.model.map_annotators_to_correct_id(
            ann_ids, self.train_annotators)

        # batch update after ann_ids updated
        loss_step = self.step((x, y, ids, ann_ids), step_name="train")

        if self.replay_samples == -1:
            for a, ann in enumerate(self.gen_model.annotators):
                annotator_ids = torch.ones(x.shape[0]).to(self.device) * a
                sample = self.gen_model.forward(x, annotator_ids)
                imgs.append(x)
                samples.append(torch.argmax(sample, dim=1).unsqueeze(1))
                ann_ids_gen.append(annotator_ids)
        else:
            for i in range(self.replay_samples):
                annotator_ids = torch.tensor(random.choices(range(len(
                    self.gen_model.annotators)), k=x.shape[0]))  # range(len()) to map to annotator ids
                sample = self.gen_model.forward(annotator_ids)
                imgs.append(x)
                samples.append(torch.argmax(sample, dim=1).unsqueeze(1))
                ann_ids_gen.append(annotator_ids)

        img_tens = torch.concat(imgs)
        sample_tens = torch.concat(samples)
        ann_ids_gen = torch.concat(ann_ids_gen)

        new_batch = (
            img_tens,
            sample_tens,
            ids*len(self.gen_model.annotators),
            ann_ids_gen,
        )

        loss_replay_step = self.replay_step(
            new_batch, step_name="replay_train")

        loss = loss_step + (self.lambda_gen * loss_replay_step)
        self.log(f"cum_train_loss", loss)

        return loss

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
