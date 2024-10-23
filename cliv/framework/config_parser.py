import hashlib
import importlib
from typing import Any

import lightning as L
from ruamel.yaml import YAML

from .cliv_trainer import ClivTrainer


def hashable_dict(input_dat) -> str:
    values = []

    if isinstance(input_dat, dict):
        keys = list(input_dat.keys())
        keys.sort()  # sort keys because dict is not always in same order
        for k in keys:
            values.append(k)
            values.append(hashable_dict(input_dat[k]))
    elif isinstance(input_dat, list):
        for list_elm in input_dat:
            values.append(hashable_dict(list_elm))
    else:
        values.append(str(input_dat))

    return "".join(values)


def load_class_from_str(class_str: str):
    mn, cn = class_str.rsplit(".", 1)
    m = importlib.import_module(mn)
    c = getattr(m, cn)
    return c


def check_required_entries(
    config: dict[Any, Any] | str, required_entries: list[str]
):
    for entry in required_entries:
        assert entry in config, f"{entry} missing from config file"


def parse(config: dict[Any, Any] | list | str):
    if isinstance(config, (tuple, list)):
        return [parse(param) for param in config]
    if isinstance(config, dict):
        if "+class" in config:
            class_conf = config.copy()
            cls_name = str(class_conf.pop("+class"))
            conf_cls = load_class_from_str(cls_name)
            class_config = {k: parse(v) for k, v in class_conf.items()}
            return conf_cls(**class_config)
        else:
            config = {k: parse(v) for k, v in config.items()}

    return config

def parse_trainer_class(config: dict[Any, Any] | list | str):
    class_conf = config.copy()
    cls_name = str(class_conf.pop("+class"))
    conf_cls = load_class_from_str(cls_name)
    return conf_cls
        
def parse_experiment(config: dict):
    check_required_entries(config, ["model", "optimizer", "loss"])
    model = parse(config["model"])

    
    optimizer_kwargs = {"params": model.parameters(), **config["optimizer"]}
    optimizer = parse(optimizer_kwargs)

    if "scheduler" in config:
        scheduler_kwargs = {"optimizer": optimizer, **config["scheduler"]}
        scheduler = parse(scheduler_kwargs)
    else:
        scheduler = None

    experiment_kwargs: dict[str, Any] = {
        **config,
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }

    return parse(experiment_kwargs)


def parse_config(config: dict[str, Any] | str, replace_in_conf=None):
    if isinstance(config, str):
        assert ".yaml" in config, "only .yaml config files are supported"
        yaml = YAML(typ="safe")
        with open(config) as f:
            config_dict = yaml.load(f)
    else:
        config_dict = config
    
    if replace_in_conf is not None:
        config_dict.update(replace_in_conf)

    check_required_entries(config_dict, ["datamodule", "experiment"])

    datamodule = parse(config_dict["datamodule"])
    experiment = parse_experiment(config_dict["experiment"])

    assert isinstance(
        datamodule, L.LightningDataModule
    ), "Datamodule has to be a LightningDataModule"
    assert isinstance(
        experiment, L.LightningModule
    ), "Experiment has to be a LightningModule"

    output_path = config_dict.get("output_path", "models/")
    seed = config_dict.get("seed", 0)

    config_hash = hashlib.sha1(hashable_dict(config_dict).encode()).hexdigest()

    print("trainer_args", config_dict.get("trainer", {}))

    trainer = ClivTrainer(
        datamodule,
        experiment,
        seed=seed,
        config_hash=config_hash,
        output_path=output_path,
        trainer_args=config_dict.get("trainer", {}),
        config_dict=config_dict,
        logger=config_dict.get("logger", None),
        matmul_precision=config_dict.get("matmul_precision", "highest"),
    )

    return trainer
