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



# for seed in [-1, 1, 2, 3, 4]:#[1, 2, 3, 4]:

#     readers_seq = ["label_a2", "label_a3", "label_a4"]

#     yaml = YAML(typ="safe")
#     with open('./train_configs/single_reader_configs/mmis_label_a1_ewc.yaml') as f:
#         config_dict = yaml.load(f)
#     if seed!=-1:
#         u = {"experiment": {"seed": seed}}
#         update_deep(config_dict, u)

#     trainer = parse_config(config_dict)

#     trainer.fit()
#     pre_train_path = trainer.get_last_checkpoint()

#     yaml = YAML(typ="safe")
#     with open('./train_configs/ewc_cont/mmis_label_a2_ewc.yaml') as f:
#     #with open('./train_configs/naive_cont/annot_2_naive_cont.yaml') as f:
#         config_dict = yaml.load(f)

#     for i in range(len(readers_seq)):
#         if seed==-1:
#             u = {"datamodule": 
#                 {"train_masks": [f"{readers_seq[i]}"]},
#                     #"seed": seed},
#                 "experiment": 
#                 {#"seed": seed,
#                     "model": 
#                     {"annotators": [readers_seq[i]],
#                         "pretrain_path": pre_train_path,}}}
#         else:
#             u = {"datamodule": 
#                 {"train_masks": [f"{readers_seq[i]}"]},
#                     #"seed": seed},
#                 "experiment": 
#                 {"seed": seed,
#                     "model": 
#                     {"annotators": [readers_seq[i]],
#                         "pretrain_path": pre_train_path,}}}

#         update_deep(config_dict, u)

#         print(config_dict)
#         print("______________________________________________________________")

#         trainer = parse_config(config_dict)
#         trainer.fit()
#         pre_train_path = trainer.get_last_checkpoint()


for seed in [-1, 1, 2, 3, 4]:
    for annotator_overlap in [0.0, 0.50, 0.75]:


    # # single reader models

    # # yaml = YAML(typ="safe")
    # # with open('./train_configs/single_reader_configs/mmis_label_a1_overlap_sampling.yaml') as f:
    # #      config_dict = yaml.load(f)

    # # for reader in ["label_a1"]:
    # #     u = {"datamodule":
    # #          {"seed": seed, "train_masks": [reader],
    # #           "val_masks": [reader],
    # #           "test_masks": [reader]},
    # #           "experiment": {"model":
    # #                          {"annotators": [reader]}}}

    # #     update_deep(config_dict, u)
    # #     trainer = parse_config(config_dict)
    # #     trainer.fit()
    # #     del trainer

        readers_seq = ["label_a1", "label_a2", "label_a3", "label_a4"]

        yaml = YAML(typ="safe")
        with open('./train_configs/single_reader_configs/mmis_label_a1_overlap_sampling.yaml') as f:
            config_dict = yaml.load(f)
        u = {"datamodule": {"seed": seed, "annotator_overlap": annotator_overlap}, "experiment": {"seed": seed}}
        update_deep(config_dict, u)

        trainer = parse_config(config_dict)

        trainer.fit()
        pre_train_path = trainer.get_last_checkpoint()

        yaml = YAML(typ="safe")
        with open('./train_configs/gen_cont/mmis_label_a2_gen_cont_fix_overlap_sampling.yaml') as f:
        #with open('./train_configs/naive_cont/annot_2_naive_cont.yaml') as f:
            config_dict = yaml.load(f)

        for i in range(len(readers_seq)-1):
            u = {"datamodule": 
                {"seed": seed, "train_masks": [f"{readers_seq[i+1]}"]},
                "experiment": 
                {   "seed": seed,
                    "lambda_gen": lambda_gen,
                    "train_annotators": [f"{readers_seq[i+1]}"],
                    "model": 
                    {"annotators": readers_seq[:i+2],
                        "pretrain_path": pre_train_path,}, # for naive!
                    "gen_model":
                    {"annotators": readers_seq[:i+1],
                        "pretrain_path": pre_train_path}}}
            
            update_deep(config_dict, u)

            print(config_dict)
            print("______________________________________________________________")
            #pre_train_path += "next"

            trainer = parse_config(config_dict)
            trainer.fit()
            pre_train_path = trainer.get_last_checkpoint()
            del trainer

    #     yaml = YAML(typ="safe")
    #     with open('./train_configs/reproduce/mmis_reproduce_config_monai_overlap_sampling.yaml') as f:
    #         config_dict = yaml.load(f)


    #     u = {"datamodule": {"seed": seed, "annotator_overlap": annotator_overlap}, "experiment": {"seed": seed}}
    #     update_deep(config_dict, u)
    #     trainer = parse_config(config_dict)
    #     trainer.fit()
    #     del trainer



    #     yaml = YAML(typ="safe")
    #     with open('./train_configs/single_reader_configs/mmis_label_a1_overlap_sampling.yaml') as f:
    #         config_dict = yaml.load(f)
    #     u = {"datamodule": {"seed": seed, "annotator_overlap": annotator_overlap}, "experiment": {"seed": seed}}
    #     update_deep(config_dict, u)

    #     trainer = parse_config(config_dict)

    #     trainer.fit()
    #     pre_train_path = trainer.get_last_checkpoint()

    #     yaml = YAML(typ="safe")
    #     with open('./train_configs/naive_cont/mmis_label_a2_naive_cont_overlap_sampling.yaml') as f:
    #         config_dict = yaml.load(f)

    #     for i in range(len(readers_seq)-1):
    #         u = {"datamodule": 
    #                 {"seed": seed, "annotator_overlap": annotator_overlap, "train_masks": [f"{readers_seq[i+1]}"]},
    #                 "experiment": 
    #                     {
    #                     "seed": seed,
    #                     "model": 
    #                         {"annotators": readers_seq[:i+2],
    #                         "pretrain_path": pre_train_path,
    #                         "train_annotators": [f"{readers_seq[i+1]}"]}}} # for naive!
            
            
    #         update_deep(config_dict, u)

    #         print(config_dict)
    #         print("______________________________________________________________")
    #         #pre_train_path += "next"

    #         trainer = parse_config(config_dict)
    #         trainer.fit()
    #         pre_train_path = trainer.get_last_checkpoint()
    #         del trainer




#     readers_seq = ["label_a2", "label_a3", "label_a4"]

#     yaml = YAML(typ="safe")
#     with open('./train_configs/single_reader_configs/mmis_label_a1_overlap_sampling.yaml') as f:
#         config_dict = yaml.load(f)
#     u = {"datamodule": {"seed": seed, "annotator_overlap": annotator_overlap}, "experiment": {"seed": seed}}
#     update_deep(config_dict, u)

#     trainer = parse_config(config_dict)

#     trainer.fit()
#     pre_train_path = trainer.get_last_checkpoint()

#     yaml = YAML(typ="safe")
#     with open('./train_configs/naive_cont/mmis_label_a2_naive_cont_one_dist_overlap_sampling.yaml') as f:
#     #with open('./train_configs/naive_cont/annot_2_naive_cont.yaml') as f:
#         config_dict = yaml.load(f)

#     for i in range(len(readers_seq)):
#         u = {"datamodule": 
#             {"seed": seed, "annotator_overlap": annotator_overlap, "train_masks": [f"{readers_seq[i]}"]},
#                 #"seed": seed},
#             "experiment": 
#             {"seed": seed,
#                 "model": 
#                 {"annotators": [readers_seq[i]],
#                     "pretrain_path": pre_train_path,}}}
        
#         update_deep(config_dict, u)

#         print(config_dict)
#         print("______________________________________________________________")
#         #pre_train_path += "next"

#         trainer = parse_config(config_dict)
#         trainer.fit()
#         pre_train_path = trainer.get_last_checkpoint()
#         del trainer