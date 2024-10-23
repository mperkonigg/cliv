from monai.metrics import DiceMetric, ConfusionMatrixMetric, compute_iou
from itertools import combinations
from sklearn.metrics import cohen_kappa_score
import torch
import numpy as np
import SimpleITK as sitk
from scipy import stats
from collections.abc import Mapping
from ruamel.yaml import YAML
from cliv.framework import parse_config
import pandas as pd
from cliv.datasets import ClivDSTypes
from torch.utils.data import DataLoader

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


def evaluate_cliv_model(dataloader: DataLoader, 
                        model: torch.nn.Module, 
                        evaluate_readers: list[str] = ['Maps1_T', 'Maps2_T', 'Maps3_T', 'Maps4_T', 'Maps5_T', 'Maps6_T'], 
                        num_classes=5) -> dict:
    """Evaluates a model for cliv

    Args:
        dataloader (DataLoader): Dataloader holding the dataset to evaluate on
        model (torch.nn.Module): Model that should be evaluated
        evaluate_readers (list[str], optional): readers annotations that should be evaluated. Defaults to ['Maps1_T', 'Maps2_T', 'Maps3_T', 'Maps4_T', 'Maps5_T', 'Maps6_T'].
        num_classes (int, optional): Number of classes in the annotations of the dataset. Defaults to 5.

    Returns:
        dict: Dictonary with dice, accuracy and cohen kappa for the readers in evaluate_readers
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dice_metrices = dict()
    acc_metrices = dict()
    ck_scores = dict()

    iter_dl = iter(dataloader)

    for er in evaluate_readers:
        dice_metrices[er] = DiceMetric()
        acc_metrices[er] = ConfusionMatrixMetric(metric_name='accuracy')
        ck_scores[er] = []

    for batch in iter_dl:
        imgs = batch[0].to(device)
        model.forward(imgs)

        for er in evaluate_readers:
            pred = model.forward(imgs, torch.zeros(len(batch[0])), [er])
            pred_max = torch.nn.functional.one_hot(torch.argmax(
                pred, dim=1), num_classes=num_classes).moveaxis(-1, 1)
            dice_metrices[er](pred_max.cpu(), batch[1])

            acc_metrices[er](pred_max.cpu(), torch.nn.functional.one_hot(
                batch[1].squeeze()).moveaxis(-1, 1))

            ck_scores[er].append(cohen_kappa_score(torch.argmax(pred, dim=1).cpu().numpy(
            ).flatten(), batch[1].squeeze().numpy().flatten()))  # not pretty at all

    results = {}
    results['dice'] = {}
    results['acc'] = {}
    results['cohen_kappa'] = {}

    for er in evaluate_readers:
        results['dice'][er] = dice_metrices[er].aggregate()[0].numpy()
        results['acc'][er] = acc_metrices[er].aggregate()[0].numpy()[0]
        results['cohen_kappa'][er] = np.array(ck_scores[er]).mean()

    return results



def get_results_staple(dataloader: DataLoader, 
                       model: torch.nn.Module, 
                       evaluate_readers: list[str]=['Maps1_T', 'Maps2_T', 'Maps3_T', 'Maps4_T', 'Maps5_T', 'Maps6_T']) -> dict:
    """Calculate the results when merging with the STAPLE algorithm

    Args:
        dataloader (DataLoader): Dataloader holding the dataset to evaluate
        model (torch.nn.Module): Model for evaluations
        evaluate_readers (list[str], optional): readers that should be considered. Defaults to ['Maps1_T', 'Maps2_T', 'Maps3_T', 'Maps4_T', 'Maps5_T', 'Maps6_T'].

    Returns:
        Dict: dice, accuracy and cohen kappa for STAPLE merging
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    iter_dl = iter(dataloader)

    dice_metric = DiceMetric()
    acc_metric = ConfusionMatrixMetric(metric_name='accuracy')
    ck_scores = []

    for batch in iter_dl:
        imgs = batch[0].to(device)

        pred_imgs = []
        for er in evaluate_readers:
            pred = model.forward(imgs, torch.zeros(len(batch[0])), [er])
            pred_max = torch.argmax(pred, dim=1)

            pred_imgs.append(pred_max.cpu().detach())

        staple_predictions = []
        for i in range(len(batch[0])):
            masks = [m[i] for m in pred_imgs]
            masks_sitk_format = [sitk.GetImageFromArray(
                mask.astype(np.uint8)) for mask in masks]
            vote_masks_sitk_format = sitk.MultiLabelSTAPLE(masks_sitk_format)
            vote_masks = sitk.GetArrayFromImage(vote_masks_sitk_format)

            if np.any(vote_masks < 0) or np.any(vote_masks > 4) or np.any(np.mod(vote_masks, 1) != 0):
                # fall back to majority voting to avoid unannotated regions
                vote_masks = stats.mode(masks, axis=0)[0]

            staple_predictions.append(vote_masks)

        one_hot_pred = torch.nn.functional.one_hot(torch.tensor(
            staple_predictions).long(), num_classes=5).moveaxis(-1, 1)
        dice_metric(one_hot_pred.cpu(), batch[1])
        acc_metric(one_hot_pred.cpu(), torch.nn.functional.one_hot(
            batch[1].squeeze()).moveaxis(-1, 1))
        ck_scores.append(cohen_kappa_score(np.array(staple_predictions).flatten(
        ), batch[1].squeeze().numpy().flatten()))  # not pretty at all

    return {'dice': dice_metric.aggregate().numpy()[0],
            'acc': acc_metric.aggregate()[0].numpy()[0],
            'cohen_kappa': np.array(ck_scores).mean()}

def get_bwt_fwt(df_res: pd.DataFrame, 
                prefix_annotator: str = "Maps/", 
                readers_seq: list[str]=["Maps1_T", "Maps2_T", "Maps3_T", "Maps4_T", "Maps5_T", "Maps6_T"], 
                performance_metric: str='dice', 
                mode: str ="NAIVE") -> tuple:
    """Calculates the BWT/FWT values based on a results dataframe

    Args:
        df_res (pd.DataFrame): Dataframe containing the results of the model
        prefix_annotator (str, optional): Optional prefix for the reader names (needed for gleason). Defaults to "Maps/".
        readers_seq (list[str], optional): Sequence of readers to calculate BWT/FWT. Defaults to ["Maps1_T", "Maps2_T", "Maps3_T", "Maps4_T", "Maps5_T", "Maps6_T"].
        performance_metric (str, optional): Performance metric to calculate BWT/FWT on, has to be a column in df_res. Defaults to 'dice'.
        mode (str, optional): Mode of the model "NAIVE", "GEN" or "NAIVE_ONE". Defaults to "NAIVE".

    Returns:
        tuple: values for bwt and fwt
    """
    
    bwt = 0.0
    for reader in readers_seq[:-1]:
        df_annotator = df_res.loc[df_res.annotator_gt ==
                                  f'{prefix_annotator}{reader}']
        base = df_annotator.loc[(df_annotator.train_phase == reader) & (
            df_annotator.inference_dist == reader)][performance_metric].values[0]
        if mode == "NAIVE_ONE":
            final = df_annotator.loc[(df_annotator.train_phase == readers_seq[-1]) & (
                df_annotator.inference_dist == readers_seq[-1])][performance_metric].values[0]
        else:
            final = df_annotator.loc[(df_annotator.train_phase == readers_seq[-1]) & (
                df_annotator.inference_dist == reader)][performance_metric].values[0]
        bwt += final-base

    bwt = bwt/(len(readers_seq)-1)

    fwt = 0.0
    for i, reader in enumerate(readers_seq[2:]):
        df_annotator = df_res.loc[df_res.annotator_gt ==
                                  f'{prefix_annotator}{reader}']
        base = df_annotator.loc[(
            df_annotator.train_phase == readers_seq[0])].dice.values[0]
        performance = df_annotator.loc[(df_annotator.train_phase == readers_seq[-1]) & (
            df_annotator.inference_dist == readers_seq[-1])][performance_metric].values[0]
        fwt += performance - base

    fwt = fwt/(len(readers_seq)-2)

    return bwt, fwt

def _get_update_dict(mode: str, 
                     reader_idx: int, 
                     pre_train_path: str, 
                     dataset: ClivDSTypes=ClivDSTypes.GLEASON19, 
                     readers_seq: list[str]=["Maps1_T", "Maps2_T", "Maps3_T", "Maps4_T", "Maps5_T", "Maps6_T"], 
                     crossval: str="Crossval0", 
                     seed: int=0, 
                     exp_seed: int=-1,
                     lambda_gen: float=-1.0) -> dict:
    """Helper function to generate an update dictonary to manipulate the configs

    Args:
        mode (str): Model mode "NAIVE", "GEN" or "NAIVE_ONE" (NAIVE==CLIV_add, GEN=CLIV_pseudo, NAIVE_ONE=naive)
        reader_idx (int): index in the readers sequence
        pre_train_path (str): path to a pretrained model
        dataset (ClivDSTypes, optional): Which dataset is used GLEASON19/MMIS. Defaults to ClivDSTypes.GLEASON19.
        readers_seq (list[str], optional): Sequence of readers. Defaults to ["Maps1_T", "Maps2_T", "Maps3_T", "Maps4_T", "Maps5_T", "Maps6_T"].
        crossval (str, optional): Crossval str for GLEASON19. Defaults to "Crossval0".
        seed (int, optional): seed for dataset splits. Defaults to 0.
        exp_seed (int, optional): seed for experiments for multiple rounds. Defaults to -1.

    Returns:
        Dict: Update dictonary
    """
    if dataset == ClivDSTypes.GLEASON19:

        u = {"datamodule":
                 {"train_masks": [f"Maps/{readers_seq[reader_idx+1]}"],
                  "train_path": f"{crossval}/train", "val_path": f"{crossval}/train", "test_path": f"{crossval}/val"}}
        
        if mode == "NAIVE":
            u2 = {"experiment":
            {
                "model":
                {"annotators": readers_seq[:reader_idx+2],
                "pretrain_path": pre_train_path,
                "train_annotators": [f"{readers_seq[reader_idx+1]}"]}}} 
        elif mode == "NAIVE_ONE":
            u2 = {"experiment":
                 {"model":
                  {"annotators": [readers_seq[reader_idx+1]],
                   "pretrain_path": pre_train_path, }}}
        elif mode == "GENERATIVE":
            u = {
                 "experiment":
                 {"train_annotators": [f"{readers_seq[reader_idx+1]}"],
                  "model":
                  {"annotators": readers_seq[:reader_idx+2],
                   "pretrain_path": pre_train_path, },
                  "gen_model":
                  {"annotators": readers_seq[:reader_idx+1],
                   "pretrain_path": pre_train_path}}}
        else:
            raise Exception(f"mode {mode} is not defined!")        
    elif dataset == ClivDSTypes.MMIS:
        u = {"datamodule":
                 {"train_masks": [f"{readers_seq[reader_idx+1]}"],
                  "seed": seed}}
        if mode == "NAIVE":
            u2 = {"experiment":
                 {
                     "seed": exp_seed,
                     "model":
                     {"annotators": readers_seq[:reader_idx+2],
                         "pretrain_path": pre_train_path,
                         "train_annotators": [f"{readers_seq[reader_idx+1]}"]}}} 
        elif mode == "NAIVE_ONE":
            u = {"experiment":
                 {"seed": exp_seed,
                     "model":
                  {"annotators": [readers_seq[reader_idx+1]],
                   "pretrain_path": pre_train_path, }}}

        elif mode == "GENERATIVE":
            u = {"experiment":
                 {"seed": exp_seed, 
                  "lambda_gen": lambda_gen,
                  "train_annotators": [f"{readers_seq[reader_idx+1]}"],
                  "model":
                  {"annotators": readers_seq[:reader_idx+2],
                   "pretrain_path": pre_train_path, },
                  "gen_model":
                  {"annotators": readers_seq[:reader_idx+1],
                   "pretrain_path": pre_train_path}}}
        else:
            raise Exception(f"mode {mode} is not defined!")
        

        u = update_deep(u, u2)
        if seed == -1:
            u['datamodule'].pop('seed')  # backward comp
        if exp_seed == -1:
            u['experiment'].pop('seed')  # backward comp
        if lambda_gen == -1.0 and 'lambda_gen' in u['experiment']:
            u['experiment'].pop('lambda_gen')

    return u


def full_benchmark_cliv(base_config: str, 
                        cont_config: str, 
                        test_data_loaders: list[DataLoader], 
                        dataset: ClivDSTypes=ClivDSTypes.GLEASON19, 
                        crossval: str='Crossval0', 
                        readers_seq: list[str]=["Maps1_T", "Maps2_T", "Maps3_T", "Maps4_T", "Maps5_T", "Maps6_T"], 
                        mode: str="NAIVE", 
                        seed: int=0, 
                        exp_seed: int=-1,
                        lambda_gen: float=-1.0) -> dict:
    """Run a full benchmark for a continual learning run of cliv

    Args:
        base_config (str): Config of the base model (first reader model)
        cont_config (str): Config for the continual runs
        test_data_loaders (list[DataLoader]): Test dataloaders to evaluate on
        dataset (ClivDSTypes, optional): Type of dataset GLEASON19/MMIS. Defaults to ClivDSTypes.GLEASON19.
        crossval (str, optional): crossval for GLEASON19. Defaults to 'Crossval0'.
        readers_seq (list[str], optional): sequence of readers to evaluate. Defaults to ["Maps1_T", "Maps2_T", "Maps3_T", "Maps4_T", "Maps5_T", "Maps6_T"].
        mode (str, optional): mode of training "NAIVE", "GEN" or "NAIVE_ONE". Defaults to "NAIVE".
        seed (int, optional): seed for datasefull_bet splits, ignored if -1. Defaults to 0.
        exp_seed (int, optional): seed for the experiment run, ignored if -1. Defaults to -1.

    Returns:
        dict: Results for full benchmark: bwt/fwt, reader results, (staple results is applicable)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    yaml = YAML(typ="safe")
    with open(base_config) as f:
        config_dict = yaml.load(f)
    if dataset == ClivDSTypes.GLEASON19:
        u = {"datamodule": {"train_path": f"{crossval}/train",
                            "val_path": f"{crossval}/train", "test_path": f"{crossval}/val"}}
        update_deep(config_dict, u)
        num_classes = 5
    elif dataset == ClivDSTypes.MMIS:
        if seed != -1:
            u = {"datamodule": {"seed": seed}}
            update_deep(config_dict, u)
        if exp_seed != -1:
            u = {"experiment": {"seed": exp_seed}}
            update_deep(config_dict, u)
        num_classes = 2

    trainer = parse_config(config_dict)

    model = trainer.load_model()
    model.to(device)
    model.eval()

    eval_res = []

    for m, dl in test_data_loaders.items():
        if m != 'Maps/STAPLE':
            iter_dl = iter(dl)
            df_res = pd.DataFrame(evaluate_cliv_model(
                iter_dl, model, evaluate_readers=readers_seq[:1], num_classes=num_classes))
            df_res["annotator_gt"] = m
            df_res["train_phase"] = readers_seq[0]
            eval_res.append(df_res)

    pre_train_path = trainer.get_last_checkpoint()
    yaml = YAML(typ="safe")
    with open(cont_config) as f:
        config_dict = yaml.load(f)

    for i in range(len(readers_seq)-1):
        u = _get_update_dict(mode, i, pre_train_path, dataset=dataset,
                             readers_seq=readers_seq, crossval=crossval, seed=seed, exp_seed=exp_seed, lambda_gen=lambda_gen)
        update_deep(config_dict, u)

        trainer = parse_config(config_dict)
        model = trainer.load_model()
        model.to(device)
        model.eval()

        for m, dl in test_data_loaders.items():
            if m != 'Maps/STAPLE':
                iter_dl = iter(dl)
                if mode == "NAIVE_ONE":
                    df_res = pd.DataFrame(evaluate_cliv_model(iter_dl, model, evaluate_readers=[
                                          readers_seq[i+1]], num_classes=num_classes))
                else:
                    df_res = pd.DataFrame(evaluate_cliv_model(
                        iter_dl, model, evaluate_readers=readers_seq[:i+2], num_classes=num_classes))
                df_res["annotator_gt"] = m
                df_res["train_phase"] = readers_seq[i+1]
                eval_res.append(df_res)

        pre_train_path = trainer.get_last_checkpoint()

    df_res = pd.concat(eval_res).reset_index(
        names="inference_dist")  # res table of single things
    df_res.pivot_table(index=['annotator_gt', 'train_phase'], columns=[
                       'inference_dist'], values=['dice', 'acc', 'cohen_kappa'])

    if dataset == ClivDSTypes.GLEASON19:
        bwt, fwt = get_bwt_fwt(df_res, readers_seq=readers_seq, mode=mode)
    else:
        bwt, fwt = get_bwt_fwt(df_res, prefix_annotator="",
                               readers_seq=readers_seq, mode=mode)

    if 'Maps/STAPLE' in test_data_loaders:
        if mode == "NAIVE_ONE":
            dict_staple = get_results_staple(
                test_data_loaders['Maps/STAPLE'], model, evaluate_readers=[readers_seq[-1]])
        else:
            dict_staple = get_results_staple(
                test_data_loaders['Maps/STAPLE'], model, evaluate_readers=readers_seq)
    else:
        dict_staple = {}

    return {"single_reader_res": df_res, "bwt_fwt": {"bwt": bwt, "fwt": fwt}, "staple_res": dict_staple}


def get_final_model(base_config: str, 
                    cont_config: str, 
                    dataset: ClivDSTypes=ClivDSTypes.GLEASON19, 
                    crossval: str='Crossval0', 
                    readers_seq: list[str]=["Maps1_T", "Maps2_T", "Maps3_T", "Maps4_T", "Maps5_T", "Maps6_T"], 
                    mode: str="NAIVE", 
                    seed: int=-1, 
                    exp_seed: int=-1)->torch.nn.Module:
    """Helper function to get the final model of a continual learning run

    Args:
        base_config (str): Config of the base model (first reader model)
        cont_config (str): Config for the continual runs
        test_data_loaders (list[DataLoader]): Test dataloaders to evaluate on
        dataset (ClivDSTypes, optional): Type of dataset GLEASON19/MMIS. Defaults to ClivDSTypes.GLEASON19.
        crossval (str, optional): crossval for GLEASON19. Defaults to 'Crossval0'.
        readers_seq (list[str], optional): sequence of readers to evaluate. Defaults to ["Maps1_T", "Maps2_T", "Maps3_T", "Maps4_T", "Maps5_T", "Maps6_T"].
        mode (str, optional): mode of training "NAIVE", "GEN" or "NAIVE_ONE". Defaults to "NAIVE".
        seed (int, optional): seed for dataset splits, ignored if -1. Defaults to 0.
        exp_seed (int, optional): seed for the experiment run, ignored if -1. Defaults to -1.

    Returns:
        torch.nn.Module: final model
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    yaml = YAML(typ="safe")
    with open(base_config) as f:
        config_dict = yaml.load(f)
    if dataset == ClivDSTypes.GLEASON19:
        u = {"datamodule": {"train_path": f"{crossval}/train",
                            "val_path": f"{crossval}/train", "test_path": f"{crossval}/val"}}
        update_deep(config_dict, u)
    elif dataset == ClivDSTypes.MMIS:
        if seed != -1:
            u = {"datamodule": {"seed": seed}}
            update_deep(config_dict, u)

        if exp_seed != -1:
            u = {"experiment": {"seed": exp_seed}}
            update_deep(config_dict, u)

    trainer = parse_config(config_dict)

    pre_train_path = trainer.get_last_checkpoint()

    yaml = YAML(typ="safe")
    with open(cont_config) as f:
        config_dict = yaml.load(f)

    for i in range(len(readers_seq)-1):
        u = _get_update_dict(mode, i, pre_train_path, dataset=dataset,
                             readers_seq=readers_seq, crossval=crossval, seed=seed, exp_seed=exp_seed)
        update_deep(config_dict, u)

        trainer = parse_config(config_dict)
        pre_train_path = trainer.get_last_checkpoint()

    model = trainer.load_model()
    model.to(device)
    model.eval()

    return model


def ged_full(test_data_loaders: list[DataLoader], 
             model: torch.nn.Module,
             num_classes: int=2, 
             readers_seq: list[str]=['label_a1', 'label_a2', 'label_a3', 'label_a4']) -> float:
    """Generalized energy distance following https://arxiv.org/pdf/2403.13417 and https://arxiv.org/pdf/1806.05034 


    Args:
        test_data_loaders (list[DataLoader]): dataloaders for the individual readers
        model (torch.nn.Module): model to calculate ged on
        num_classes (int, optional): number of classes in the annoations. Defaults to 2.
        readers_seq (list[str], optional): readers to consider. Defaults to ['label_a1', 'label_a2', 'label_a3', 'label_a4'].

    Returns:
        float: GED value
    """
    preds = {r: [] for r in readers_seq}
    dl_iterators = {d: iter(dl) for d, dl in test_data_loaders.items()}

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for batch in dl_iterators[readers_seq[0]]:

        for reader in readers_seq:
            pred = model.forward(batch[0].to(device), torch.zeros(len(batch[0])), [reader])
            pred = torch.nn.functional.one_hot(torch.argmax(
                pred, dim=1), num_classes=num_classes).moveaxis(-1, 1).detach().cpu()

            preds[reader].append(pred)

    dl_iterators = {d: iter(dl) for d, dl in test_data_loaders.items()}
    gt = {r: [] for r in readers_seq}
    for r in readers_seq:
        for batch in dl_iterators[r]:
            gt[r].append(torch.nn.functional.one_hot(
                batch[1].squeeze()).moveaxis(-1, 1))

    preds = {d: torch.concat(pred) for d, pred in preds.items()}
    gt = {d: torch.concat(g) for d, g in gt.items()}

    iou_readers = 0.0
    iou_preds_readers = 0.0
    iou_predictions = 0.0

    reader_combs = list(combinations(readers_seq, 2))
    for tup in reader_combs:
        iou_predictions += 1 - \
            compute_iou(preds[tup[0]], preds[tup[1]])[:, 1].mean()
        iou_readers += 1-compute_iou(gt[tup[0]], gt[tup[1]])[:, 1].mean()
    iou_predictions /= len(reader_combs)
    iou_readers /= len(reader_combs)

    for r in readers_seq:
        iou_preds_readers += 1-compute_iou(preds[r], gt[r])[:, 1].mean()

    iou_preds_readers /= len(readers_seq)

    return float(2*iou_preds_readers - iou_predictions - iou_readers)
