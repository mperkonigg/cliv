import os
import shutil
from glob import glob
from typing import Any

import lightning as L
import mlflow
import torch
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import Logger, MLFlowLogger, TensorBoardLogger
from ruamel.yaml import YAML

from cliv.framework.callbacks import DictPersistJSON, StatusCallback


def replace_symbols_dict(config_dict):
    """Creates a dict without the + in the keys"""
    config_data = {}
    for k in config_dict:
        if type(config_dict[k]) is dict:
            nested_dict = replace_symbols_dict(config_dict[k])
            config_data[k.replace("+", "")] = nested_dict
        else:
            config_data[k.replace("+", "")] = config_dict[k]
    return config_data


class ClivTrainer:
    def __init__(
        self,
        datamodule: L.LightningDataModule,
        lightningmodule: L.LightningModule,
        output_path: str = None,
        seed: int = 0,
        config_hash: str = None,
        recompute: bool = False,
        trainer_args: dict = {},
        config_dict: dict = {},
        logger: dict = None,
        matmul_precision: str = "highest",
    ):
        self.datamodule = datamodule
        self.lightningmodule = lightningmodule
        self.output_path = output_path
        self.seed = seed
        self.config_hash = config_hash
        self.recompute = recompute
        self.trainer_args = trainer_args
        self.config_dict = config_dict
        self.logger = logger

        torch.set_float32_matmul_precision(matmul_precision)

        self.init_stuff()

    def init_stuff(self):
        print("output path before setting", self.output_path)
        self.output_path = os.path.join(
            self.output_path, self.config_hash
        )  # we add the config hash to the output path
        print("output path set to", self.output_path)
        if os.path.exists(self.output_path) and self.recompute:
            shutil.rmtree(self.output_path)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
            YAML().dump(self.config_dict, open(self.output_path + "/config.yaml", "w"))

        self.checkpoint_path = os.path.join(self.output_path, "checkpoints")
        checkpoint_args: dict[str, Any] = {
            "dirpath": self.checkpoint_path,
            "save_last": True,
            **self.config_dict.get("checkpoint", {}),
        }
        self.checkpoint_callback = ModelCheckpoint(**checkpoint_args)

        self.status_path = os.path.join(self.output_path, "status.json")

        self.status = DictPersistJSON(self.status_path)
        self.status_callback = StatusCallback(self.status)

        seed_everything(self.seed)

    def get_best_checkpoint(self):
        if self.checkpoint_callback.best_model_path == "":
            return glob(self.checkpoint_callback.dirpath + "/epoch*.ckpt")[0]
        else:
            return self.checkpoint_callback.best_model_path

    def get_last_checkpoint(self):
        if self.checkpoint_callback.last_model_path == "":
            if os.path.exists(self.checkpoint_callback.dirpath + "/last.ckpt"):
                return self.checkpoint_callback.dirpath + "/last.ckpt"
            else:
                return None
        else:
            return self.checkpoint_callback.last_model_path

    def create_logger(self) -> Logger:
        """Creates the logger of the experiment.

        Returns:
            Logger: experiment logger
        """

        if self.logger is not None:
            if self.logger["type"] == "mlflow":
                mlflow.enable_system_metrics_logging()
                if "tracking_uri" in self.logger:
                    tracking_uri = self.logger["tracking_uri"]
                else:
                    tracking_uri = "http://127.0.0.1:5000"
                mlflow_logger = MLFlowLogger(
                    experiment_name=self.config_dict["experiment_name"],
                    tracking_uri=tracking_uri,
                    run_name=self.config_hash,
                )

                conf_dict_replaced = replace_symbols_dict(self.config_dict)
                mlflow_logger.log_hyperparams(conf_dict_replaced)
                return mlflow_logger
            else:
                raise NotImplementedError(f"Logger {self.logger['type']} not implemented")

    def create_trainer(self):
        trainer_args = {
            **self.trainer_args,
            "callbacks": [self.checkpoint_callback, self.status_callback],
            "logger": [self.create_logger()],
        }

        trainer = L.Trainer(**trainer_args)
        return trainer

    def fit(self):
        if self.status["fit"]:
            print(
                "Training is already finished!"
            )
            return self

        last_checkpoint = self.get_last_checkpoint()
        if last_checkpoint == "":
            last_checkpoint = None

        if last_checkpoint is not None:
            print(f"Resuming training from {last_checkpoint}")

        trainer = self.create_trainer()
        trainer.fit(self.lightningmodule, self.datamodule, ckpt_path=last_checkpoint)
        return self

    def load_model(self, best_checkpoint=False, eval=True, incomplete=False):
        if not self.status["fit"] and not incomplete:
            raise Exception(
                "\nTraining has not finished."
            )

        if best_checkpoint:
            load_ckpt = self.get_best_checkpoint()
        else:
            load_ckpt = self.get_last_checkpoint()

        print(f"Load model from: {load_ckpt}")
        self.lightningmodule.load_state_dict(
            torch.load(load_ckpt, map_location="cpu")["state_dict"]
        )

        model = self.lightningmodule.model

        if eval:
            model.eval()

        return model