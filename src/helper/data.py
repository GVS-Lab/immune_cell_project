import logging
from typing import Iterable, List

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data.datasets import TorchTransformableSubset, TorchImageDataset


class BaseDataHandler(object):
    def __init__(
        self,
        dataset: TorchImageDataset,
        batch_size: int = 64,
        num_workers: int = 10,
        transformation_dicts: List[dict] = None,
        random_state: int = 42,
        drop_last_batch: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transformation_dicts = transformation_dicts
        self.random_state = random_state
        self.drop_last_batch = drop_last_batch


class DataHandler(BaseDataHandler):
    def __init__(
        self,
        dataset: TorchImageDataset,
        batch_size: int = 64,
        num_workers: int = 10,
        transformation_dicts: List[dict] = None,
        random_state: int = 42,
        drop_last_batch: bool = True,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            transformation_dicts=transformation_dicts,
            random_state=random_state,
            drop_last_batch=drop_last_batch,
        )
        self.train_val_test_datasets_dict = None
        self.data_loader_dict = None

    def stratified_train_val_test_split(self, splits: Iterable) -> None:
        train_portion, val_portion, test_portion = splits[0], splits[1], splits[2]

        indices = np.array(list(range(len(self.dataset))))
        labels = np.array(self.dataset.labels)

        train_and_val_idc, test_idc = train_test_split(
            indices,
            test_size=test_portion,
            stratify=labels,
            random_state=self.random_state,
        )

        train_idc, val_idc = train_test_split(
            train_and_val_idc,
            test_size=val_portion / (1 - test_portion),
            stratify=labels[train_and_val_idc],
            random_state=self.random_state,
        )
        logging.debug(
            "Data split on FoV image level ( {} training images, {} validation images,"
            " {} test images).".format(len(train_idc), len(val_idc), len(test_idc))
        )

        train_dataset = TorchTransformableSubset(
            dataset=self.dataset, indices=train_idc
        )
        val_dataset = TorchTransformableSubset(dataset=self.dataset, indices=val_idc)
        test_dataset = TorchTransformableSubset(dataset=self.dataset, indices=test_idc)

        self.train_val_test_datasets_dict = {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
        logging.debug(
            "Training samples: {}, validation samples: {}, test samples: {}.".format(
                len(train_idc), len(val_idc), len(test_idc)
            )
        )

    def get_data_loader_dict(self, shuffle: bool = True,) -> None:
        if self.transformation_dicts is not None:
            if len(self.transformation_dicts) > 0:
                for k in self.transformation_dicts[0].keys():
                    self.train_val_test_datasets_dict[k].set_transform_pipeline(
                        [
                            self.transformation_dicts[i][k]
                            for i in range(len(self.transformation_dicts))
                        ]
                    )
        data_loader_dict = {}
        for k, dataset in self.train_val_test_datasets_dict.items():
            data_loader_dict[k] = DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=shuffle and k == "train",
                num_workers=self.num_workers,
                drop_last=self.drop_last_batch,
            )

        self.data_loader_dict = data_loader_dict
