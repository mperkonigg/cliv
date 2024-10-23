from collections.abc import Mapping
from msig_tools.ml_framework import parse_config
from ruamel.yaml import YAML

def update_deep(d, u):
    for k, v in u.items():
        # this condition handles the problem
        if not isinstance(d, Mapping):
            d = u
        elif isinstance(v, Mapping):
            r = update_deep(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]

    return d


yaml = YAML(typ="safe")
with open('./train_configs/reproduce/reproduce_config_monai.yaml') as f:
#with open('./train_configs/naive_cont/annot_2_naive_cont.yaml') as f:
    config_dict = yaml.load(f)

for crossval in ["Crossval0", "Crossval1", "Crossval2", "Crossval3"]:
    u = {"datamodule": 
            {"train_path": f"{crossval}/train", "val_path": f"{crossval}/train", "test_path": f"{crossval}/val"}}
    
    update_deep(config_dict, u)
    trainer = parse_config(config_dict)
    trainer.fit()