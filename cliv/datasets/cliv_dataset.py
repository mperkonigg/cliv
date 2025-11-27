import os
import torch
import numpy as np
import random
import h5py
from typing import Callable, Union
from enum import Enum
from glob import glob
import SimpleITK as sitk

class ClivDSTypes(Enum):
    MMIS = 1
    QUBIQ = 2


class ClivMMISDataset(torch.utils.data.Dataset):
    """Dataset using the MMIS dataset (https://mmis2024.com/#dataset)

        Args:
            data_path (str): path to the data of MMIS
            transforms (Callable, optional): transforms to apply. Defaults to None.
            xkeys (list, optional): keys to return as x of batch. Defaults to ['img'].
            annotators (list, optional): annotator labels to use. Defaults to ['label_a1', 'label_a2', 'label_a3', 'label_a4'].
    """

    def __init__(
            self,
            data_path: str,
            transforms: Callable = None,
            xkeys: list[str] = ['img'],
            annotators: list[str] = ['label_a1',
                                     'label_a2', 'label_a3', 'label_a4'],
            annotator_overlap: float = None,
            available_annotators: list[str] = ['label_a1',
                                     'label_a2', 'label_a3', 'label_a4'],
            seed: int = 0,
    ):
        self.h5_files = [f for f in os.listdir(data_path) if '.h5' in f]

        self.annotator_overlap = annotator_overlap

        if annotator_overlap is not None:
            random.seed(seed)
            random.shuffle(self.h5_files)

            overlap_ratio = int(len(self.h5_files)*annotator_overlap)

            self.split_ratio = int((len(self.h5_files)-overlap_ratio)/len(available_annotators))
            self.annotator_labels = {}

            h5_list = []
            annotator_list = []
            for i, a in enumerate(available_annotators):
                if a in annotators:
                    h5_list.extend(self.h5_files[:overlap_ratio])
                    annotator_list.extend([a]*overlap_ratio)
                    h5_list.extend(self.h5_files[overlap_ratio+(i*self.split_ratio): overlap_ratio+((i+1)*self.split_ratio)])
                    annotator_list.extend([a]*self.split_ratio)

            self.h5_files = h5_list
            self.annotator_list = annotator_list


        self.transforms = transforms
        self.xkeys = xkeys
        self.annotators = annotators
        self.data_path = data_path

    def __getitem__(self, i):
        # read data
        if self.annotator_overlap is None:
            data = h5py.File(os.path.join(self.data_path, self.h5_files[i]), "r")
            # choose one of the annotators at random
            id = random.randint(0, len(self.annotators)-1)
            annotator_id = self.annotators[id]
        else:
            data = h5py.File(os.path.join(self.data_path, self.h5_files[i]), "r")
            annotator_id = self.annotator_list[i]
            id = self.annotators.index(annotator_id)


        unpacked_data = {}
        for d in data:
            # unpack for performance reasons, otherwise transform to tensor is extremley slow in monai
            unpacked_data[d] = data[d][:]

        unpacked_data["label"] = unpacked_data[annotator_id]

        if self.transforms is not None:
            dataentry = self.transforms(unpacked_data)

        x = [dataentry[k] for k in self.xkeys]
        x = x[0] if len(x) == 1 else x

        return x, dataentry["label"], annotator_id, id

    def __len__(self):
        return len(self.h5_files)

class ClivQUIBDataset(torch.utils.data.Dataset):
    """Dataset using the QUIB Datasets

        Args:
            data_path (_type_): path to the data of QUIB
            transforms (_type_, optional): transforms to apply. Defaults to None.
            annotators (list, optional): annotator labels to use. Defaults to ['seg01', 'seg02', 'seg03', 'seg04', 'seg05', 'seg06'] prostate use case.
    """

    def __init__(
            self,
            data_path: str,
            transforms: Callable = None,
            task_id: Union[list[str], str]="task01", #some QUIB datasets have more than one tasks
            annotators: list[str] = ['seg01', 'seg02', 'seg03', 'seg04', 'seg05', 'seg06'],
            annotator_overlap: float = None,
            seed: int = 0,
            recode_classes: dict = None,
            recoded_channels: int=12,
    ):
        self.case_dirs = [d for d in glob(f"{data_path}/*") if os.path.isdir(d)]

        # we have to get rid of case_dirs for which no annotator has a segmentation
        if type(task_id) is str:
            task_id = [task_id]
        
        for cd in self.case_dirs:
            remove_cd = False
            for tid in task_id:
                if not np.any([os.path.exists(os.path.join(cd, f"{tid}_{annot}.nii.gz")) for annot in annotators]):
                    remove_cd = True
            if remove_cd:
                self.case_dirs.remove(cd)

        self.task_id = task_id
        self.annotator_overlap = annotator_overlap
        self.recode_classes = recode_classes
        self.recoded_channels = recoded_channels

        if annotator_overlap is not None:
            raise NotImplementedError #TODO: later after base experiments are running

        self.transforms = transforms
        self.annotators = annotators
        self.data_path = data_path


    def __getitem__(self, i):
        # read data
        if self.annotator_overlap is None:
            seg_paths = None
            while seg_paths is None:
                data = self.case_dirs[i]
                # choose one of the annotators at random
                id = random.randint(0, len(self.annotators)-1)
                annotator_id = self.annotators[id]
                if np.all([os.path.exists(os.path.join(data, f"{tid}_{annotator_id}.nii.gz")) for tid in self.task_id]):
                    seg_paths = [os.path.join(data, f"{tid}_{annotator_id}.nii.gz") for tid in self.task_id]
        else:
            raise NotImplementedError 
        
        img = torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data, "image.nii.gz"))), dtype=torch.long)
        seg = torch.tensor(np.concatenate([sitk.GetArrayFromImage(sitk.ReadImage(sp)) for sp in seg_paths]), dtype=torch.long)

        dataentry = {"img": img, "label": seg}

        if self.recode_classes is not None:
            recoded_labels = torch.zeros((self.recoded_channels, dataentry["label"].shape[1], dataentry["label"].shape[2]))

            for k, v in self.recode_classes.items():
                recoded_labels[v] = dataentry["label"][k]
            dataentry["label"] = recoded_labels
        
        if self.transforms is not None:
            dataentry = self.transforms(dataentry)

        x = dataentry["img"]

        return x, dataentry["label"], annotator_id, id

    def __len__(self):
        return len(self.case_dirs)
