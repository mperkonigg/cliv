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

readers_seq = ["Maps2_T", "Maps3_T", "Maps4_T", "Maps5_T", "Maps6_T"]

yaml = YAML(typ="safe")
with open('./train_configs/single_reader_configs/annot_1_only.yaml') as f:
#with open('./train_configs/naive_cont/annot_2_naive_cont.yaml') as f:
    config_dict = yaml.load(f)

trainer = parse_config(config_dict)
trainer.fit()

for i in range(len(readers_seq)):
    u = {"datamodule": 
        {"train_masks": [f"Maps/{readers_seq[i]}"]}, 
        "experiment": 
        {"model": 
            {"annotators": [readers_seq[i]],
                "pretrain_path": pre_train_path,}}}
    
    update_deep(config_dict, u)

    print(config_dict)
    print("______________________________________________________________")
    #pre_train_path += "next"

    trainer = parse_config(config_dict)
    trainer.fit()
    pre_train_path = trainer.get_last_checkpoint()
    del trainer