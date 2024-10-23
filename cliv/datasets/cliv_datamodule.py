from typing import Optional

from lightning import LightningDataModule
from monai.transforms.compose import MapTransform
from torch.utils.data import DataLoader

from .cliv_dataset import ClivDataset, ClivMMISDataset, ClivDSTypes


class ClivDataModule(LightningDataModule):
    def __init__(
        self,
        data_path: str,
        train_path: str,
        train_masks: list[str],
        val_path: str,
        val_masks: list[str],
        test_path: str,
        test_masks: list[str],
        batch_size: int = 32,
        class_no: int = 0,
        traintransforms: Optional[MapTransform] = None,
        valtransforms: Optional[MapTransform] = None,
        testtransforms: Optional[MapTransform] = None,
        num_workers: int = 8,
        dataset: ClivDSTypes | str = ClivDSTypes.GLEASON19,
        annotator_overlap: float = None,
        seed: int = 0,
    ):
        """Datamodule for cliv

        Args:
            data_path (str): path to the data folder
            train_path (str): path to the train folder (relative to the data folder)
            train_masks (list[str]): masks to use during training
            val_path (str): path to the validation folder (relative to the data folder)
            val_masks (list[str]):  masks to use during validation
            test_path (str): path to the test folder (relative to the data folder)
            test_masks (list[str]):  masks to use during testing
            batch_size (int, optional): _description_. Defaults to 32.
            class_no (int, optional): number of classes for segmentation task. Defaults to 0.
            traintransforms (Optional[MapTransform], optional): data transforms used during training. Defaults to None.
            valtransforms (Optional[MapTransform], optional): data transforms used during validation. Defaults to None.
            testtransforms (Optional[MapTransform], optional): data transforms used during testing.. Defaults to None.
            num_workers (int, optional): _description_. Defaults to 8.
            dataset (ClivDSTypes, optional): datasettype used, possible values ClivDSTypes.GLEASON19, ClivDSTypes.MMIS. Defaults to ClivDSTypes.GLEASON.
            annotator_overlap (float, optional): overlap ratio between readers. Defaults to None.
            seed (int, optional): seed for dataset overlap splits. Defaults to 0.
        """
        super().__init__()

        self.data_path = data_path
        self.train_path = train_path
        self.train_masks = train_masks

        self.val_path = val_path
        self.val_masks = val_masks
        self.test_path = test_path
        self.test_masks = test_masks
        self.traintransforms = traintransforms
        self.valtransforms = valtransforms
        self.testtransforms = testtransforms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_no = class_no
        if type(dataset) == str:
            self.dataset = ClivDSTypes[dataset]
        else:
            self.dataset = dataset
        print("dataset is", self.dataset)
        self.annotator_overlap = annotator_overlap
        self.seed = seed

    def setup(self, stage: str):
        if self.dataset == ClivDSTypes.GLEASON19:
            if stage == "fit":
                self.train_data = ClivDataset(
                    self.data_path,
                    self.train_path,
                    self.train_masks,
                    self.traintransforms,
                    class_no=self.class_no
                )
                self.val_data = ClivDataset(
                    self.data_path,
                    self.val_path,
                    self.val_masks,
                    self.valtransforms,
                    class_no=self.class_no
                )
            elif stage == "test":
                self.test_data = ClivDataset(
                    self.data_path,
                    self.test_path,
                    self.test_masks,
                    self.testtransforms,
                    class_no=self.class_no
                )
        elif self.dataset == ClivDSTypes.MMIS:
            if stage == "fit":
                self.train_data = ClivMMISDataset(
                    self.data_path + self.train_path,
                    self.traintransforms,
                    xkeys=['img'],
                    annotators=self.train_masks,
                    annotator_overlap=self.annotator_overlap,
                    seed=self.seed,
                )
                self.val_data = ClivMMISDataset(
                    self.data_path + self.val_path,
                    self.valtransforms,
                    xkeys=['img'],
                    annotators=self.val_masks,
                    seed=self.seed,
                )
            elif stage == "test":
                self.test_data = ClivMMISDataset(
                    self.data_path + self.test_path,
                    self.testtransforms,
                    xkeys=['img'],
                    annotators=self.test_masks,
                )

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, num_workers=self.num_workers, drop_last=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )