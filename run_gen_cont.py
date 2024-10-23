from collections.abc import Mapping
from cliv.framework import parse_config
from ruamel.yaml import YAML

def update_deep(d, u):
    for k, v in u.items():
        if not isinstance(d, Mapping):
            d = u
        elif isinstance(v, Mapping):
            r = update_deep(d.get(k, {}), v)
            d[k] = r
        else:
            d[k] = u[k]

    return d

train_config_path = "/home/fs72359/matthias_p/projects/pico/train_configs/"

for crossval in ["Crossval0", "Crossval1", "Crossval2", "Crossval3"]:

    readers_seq = ["Maps1_T", "Maps2_T", "Maps3_T", "Maps4_T", "Maps5_T", "Maps6_T"]

    yaml = YAML(typ="safe")
    with open(f'{train_config_path}/single_reader_configs/annot_1_only_monai.yaml') as f:
        config_dict = yaml.load(f)
    u = {"datamodule": {"train_path": f"{crossval}/train", "val_path": f"{crossval}/train", "test_path": f"{crossval}/val"}}
    update_deep(config_dict, u)

    trainer = parse_config(config_dict)

    print(crossval, readers_seq[0])

    trainer.fit()
    pre_train_path = trainer.get_last_checkpoint()

    yaml = YAML(typ="safe")
    with open(f'{train_config_path}/naive_cont/annot_2_naive_cont.yaml') as f:
        config_dict = yaml.load(f)

    for i in range(len(readers_seq)-1):
        u = {"datamodule": 
                {"train_masks": [f"Maps/{readers_seq[i+1]}"],
                "train_path": f"{crossval}/train", "val_path": f"{crossval}/train", "test_path": f"{crossval}/val"},
                "experiment": 
                    {
                    "model": 
                        {"annotators": readers_seq[:i+2],
                        "pretrain_path": pre_train_path,
                        "train_annotators": [f"{readers_seq[i+1]}"]}}} # for naive!
        
        
        update_deep(config_dict, u)

        print(crossval, readers_seq[i+1], "naive")

        trainer = parse_config(config_dict)
        trainer.fit()
        pre_train_path = trainer.get_last_checkpoint()
        del trainer

    readers_seq = ["Maps1_T", "Maps2_T", "Maps3_T", "Maps4_T", "Maps5_T", "Maps6_T"]

    yaml = YAML(typ="safe")
    with open(f'{train_config_path}/single_reader_configs/annot_1_only_monai.yaml') as f:
        config_dict = yaml.load(f)
    u = {"datamodule": {"train_path": f"{crossval}/train", "val_path": f"{crossval}/train", "test_path": f"{crossval}/val"}}
    update_deep(config_dict, u)

    trainer = parse_config(config_dict)

    trainer.fit()
    pre_train_path = trainer.get_last_checkpoint()

    yaml = YAML(typ="safe")
    with open(f'{train_config_path}/gen_cont/annot_2_gen_cont_fix_replay_sample.yaml') as f:
        config_dict = yaml.load(f)

    for i in range(len(readers_seq)-1):
        u = {"datamodule": 
            {"train_masks": [f"Maps/{readers_seq[i+1]}"],
            "train_path": f"{crossval}/train", "val_path": f"{crossval}/train", "test_path": f"{crossval}/val"},
            "experiment": 
            {"train_annotators": [f"{readers_seq[i+1]}"],
                "model": 
                {"annotators": readers_seq[:i+2],
                    "pretrain_path": pre_train_path,}, # for naive!
                "gen_model":
                {"annotators": readers_seq[:i+1],
                    "pretrain_path": pre_train_path}}}
        
        update_deep(config_dict, u)

        print(crossval, readers_seq[i+1], "gen")

        trainer = parse_config(config_dict)
        trainer.fit()
        pre_train_path = trainer.get_last_checkpoint()
        del trainer

    readers_seq = ["Maps2_T", "Maps3_T", "Maps4_T", "Maps5_T", "Maps6_T"]

    yaml = YAML(typ="safe")
    with open(f'{train_config_path}/single_reader_configs/annot_1_only_monai.yaml') as f:
        config_dict = yaml.load(f)
    u = {"datamodule": {"train_path": f"{crossval}/train", "val_path": f"{crossval}/train", "test_path": f"{crossval}/val"}}
    update_deep(config_dict, u)

    trainer = parse_config(config_dict)

    trainer.fit()
    pre_train_path = trainer.get_last_checkpoint()

    yaml = YAML(typ="safe")
    with open(f'{train_config_path}/naive_cont/annot_2_naive_cont_one_dist.yaml') as f:
        config_dict = yaml.load(f)

    for i in range(len(readers_seq)):
        u = {"datamodule": 
            {"train_masks": [f"Maps/{readers_seq[i]}"],
            "train_path": f"{crossval}/train", "val_path": f"{crossval}/train", "test_path": f"{crossval}/val"},
            "experiment": 
            {"model": 
                {"annotators": [readers_seq[i]],
                    "pretrain_path": pre_train_path,}}}
        
        update_deep(config_dict, u)

        print(crossval, readers_seq[i], "naive one")


        trainer = parse_config(config_dict)
        trainer.fit()
        pre_train_path = trainer.get_last_checkpoint()
        del trainer


# Joint model!

yaml = YAML(typ="safe")
with open(f'{train_config_path}/reproduce/reproduce_config_monai.yaml') as f:
    config_dict = yaml.load(f)

for crossval in ["Crossval0", "Crossval1", "Crossval2", "Crossval3"]:
    u = {"datamodule": 
            {"train_path": f"{crossval}/train", "val_path": f"{crossval}/train", "test_path": f"{crossval}/val"}}
    
    update_deep(config_dict, u)
    trainer = parse_config(config_dict)
    trainer.fit()


#single reader models

yaml = YAML(typ="safe")
with open(f'{train_config_path}/single_reader_configs/annot_1_only_monai.yaml') as f:
    config_dict = yaml.load(f)


for crossval in ["Crossval0", "Crossval1", "Crossval2", "Crossval3"]:
    for reader in readers_seq:

        u = {"datamodule": {"train_masks": [f"Maps/{reader}"], 
                            "train_path": f"{crossval}/train", "val_path": f"{crossval}/train", "test_path": f"{crossval}/val"},
                             "experiment": 
                    {"model": 
                    {"annotators": [reader],
                     "train_annotators": [reader]}}}
        update_deep(config_dict, u)

        trainer = parse_config(config_dict)
        trainer.fit()