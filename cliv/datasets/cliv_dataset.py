import os
import torch
import numpy as np
import pandas as pd
import cv2
import random
import h5py
from typing import Callable
from enum import Enum


class ClivDSTypes(Enum):
    GLEASON19 = 1
    MMIS = 2


class ClivDataset(torch.utils.data.Dataset):
    """Crowdsourced_Dataset Dataset. Read images, apply augmentation and preprocessing transformations.
    Args:
        image_path (str): path to images folder
        masks_dir (str): path to segmentation masks folder
        class_values (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
    """

    def __init__(
            self,
            data_path,
            image_path,
            masks_dirs,
            augmentation=None,
            annotator_ids='auto',
    ):

        image_path = os.path.join(data_path, image_path)

        mask_paths = [os.path.join(data_path, m) for m in masks_dirs]
        self.annotators = [x.split('/')[-1] for x in masks_dirs]
        self.mask_paths = mask_paths

        self.ids = self.get_valid_ids(os.listdir(image_path), mask_paths)
        self.images_fps = [os.path.join(image_path, image_id)
                           for image_id in self.ids]

        self.annotators_no = len(self.annotators)
        self.augmentation = augmentation
        self.annotator_ids = annotator_ids

    def __getitem__(self, i):
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        indexes = np.random.permutation(self.annotators_no)
        mask_found = False
        for ann_index in indexes:
            ann_path = self.mask_paths[ann_index]
            mask_path = os.path.join(ann_path, self.ids[i])
            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, 0)
                id = self.mask_paths.index(ann_path)
                if self.annotator_ids == 'auto':
                    annotator_id = id
                else:
                    annotator_id = self.annotator_ids[id]
                mask_found = True
                break
            else:
                continue
        if not mask_found:
            raise Exception('No mask was found for image: ' +
                            self.images_fps[i])
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation({"image": image, "mask": mask})
            image = sample['image']
            mask = sample['mask']

        return image, mask, self.ids[i], annotator_id

    def __len__(self):
        return len(self.ids)

    def get_valid_ids(self, image_ids, mask_paths):
        """
        Returns all image ids that have at least one corresponding annotated mask
        """
        all_masks = []
        for p in range(len(mask_paths)):
            mask_ids = os.listdir(mask_paths[p])
            for m in mask_ids:
                all_masks.append(m)
        all_unique_masks = np.unique(all_masks)
        valid_ids = np.intersect1d(image_ids, all_unique_masks)

        return valid_ids


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
