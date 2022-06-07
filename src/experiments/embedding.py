import os
from typing import List
import pandas as pd
import torch

from src.data.datasets import TorchTransformableSuperset
from src.experiments.base import BaseExperiment
from src.helper.data import DataHandler
from src.utils.basic.visualization import plot_confusion_matrices
from src.utils.torch.data import init_image_dataset, init_profile_dataset
from src.utils.torch.evaluation import (
    get_preds_labels,
    get_confusion_matrices,
    save_latents_to_hdf,
)
from src.utils.torch.exp import train_val_test_loop
from src.utils.torch.general import get_device
from src.utils.torch.model import (
    get_exp_configuration,
    get_nuclei_image_transformations_dict, get_nuclei_padding_transformation_dict,
)


class BaseEmbeddingExperiment:
    def __init__(
        self,
        data_config: dict,
        model_config: dict,
        save_freq: int = -1,
        pseudo_rgb: bool = False,
    ):

        self.data_transform_pipeline_dicts = []
        self.data_loader_dict = None
        self.data_set = None
        self.data_key = None
        self.label_key = None
        self.extra_feature_key = None
        self.index_key = None
        self.exp_config = None
        self.data_config = data_config
        self.model_config = model_config
        self.save_freq = save_freq
        self.pseudo_rgb = pseudo_rgb

    def initialize_profile_data_set(self):
        self.data_key = self.data_config.pop("data_key")
        self.label_key = self.data_config.pop("label_key")
        if "index_key" in self.data_config:
            self.index_key = self.data_config.pop("index_key")
        if "extra_feature_key" in self.data_config:
            self.extra_feature_key = self.data_config.pop("extra_feature_key")

        self.data_set = init_profile_dataset(**self.data_config)

    def initialize_image_data_set(self):
        self.data_key = self.data_config.pop("data_key")
        self.label_key = self.data_config.pop("label_key")
        if (
            "extra_features" in self.data_config
            and len(self.data_config["extra_features"]) > 0
        ):
            self.extra_feature_key = "extra_features"
        if "index_key" in self.data_config:
            self.index_key = self.data_config.pop("index_key")

        self.data_set = init_image_dataset(**self.data_config)

    def initialize_data_transform_pipeline(
        self, data_transform_pipelines: List[str] = None
    ):
        for data_transform_pipeline in data_transform_pipelines:
            if data_transform_pipeline is None:
                self.data_transform_pipeline_dicts.append(None)
            elif data_transform_pipeline == "nuclei_image":
                self.data_transform_pipeline_dicts.append(
                    get_nuclei_image_transformations_dict(224)
                )
            elif data_transform_pipeline == "padding_only":
                self.data_transform_pipeline_dicts.append(get_nuclei_padding_transformation_dict())
            else:
                raise NotImplementedError

    def initialize_exp_config(self):
        model_config = self.model_config["model_config"]
        optimizer_config = self.model_config["optimizer_config"]
        loss_config = self.model_config["loss_config"]
        if self.label_weights is not None:
            loss_config["weight"] = self.label_weights
            loss_config["weight"] = self.label_weights

        self.exp_config = get_exp_configuration(
            model_dict=model_config,
            data_loader_dict=None,
            data_key=self.data_key,
            label_key=self.label_key,
            index_key=self.index_key,
            extra_feature_key=self.extra_feature_key,
            optimizer_dict=optimizer_config,
            loss_fct_dict=loss_config,
        )

    def load_model(self, weights_fname):
        weights = torch.load(weights_fname)
        self.exp_config.model_config.model.load_state_dict(weights)

    def extract_and_save_latents(self, output_dir):
        device = get_device()
        for dataset_type in ["train", "val", "test"]:
            save_path = os.path.join(
                output_dir, "{}_latents.h5".format(str(dataset_type))
            )
            save_latents_to_hdf(
                exp_config=self.exp_config,
                data_loader_dict=self.data_loader_dict,
                save_path=save_path,
                dataset_type=dataset_type,
                device=device,
            )

    def plot_confusion_matrices(self, normalize=None):
        self.exp_config.data_loader_dict = self.data_loader_dict
        confusion_matrices = get_confusion_matrices(
            exp_config=self.exp_config,
            dataset_types=["train", "val", "test"],
            normalize=normalize,
        )
        plot_confusion_matrices(
            confusion_matrices,
            output_dir=self.output_dir,
            # display_labels=sorted(self.target_list),
        )


class EmbeddingExperiment(BaseExperiment, BaseEmbeddingExperiment):
    def __init__(
        self,
        output_dir: str,
        data_config: dict,
        model_config: dict,
        train_val_test_split: List[float] = [0.7, 0.2, 0.1],
        batch_size: int = 64,
        num_epochs: int = 64,
        early_stopping: int = -1,
        random_state: int = 42,
        save_freq: int = -1,
        pseudo_rgb: bool = False,
    ):
        BaseExperiment.__init__(
            self,
            output_dir=output_dir,
            train_val_test_split=train_val_test_split,
            batch_size=batch_size,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            random_state=random_state,
        )
        BaseEmbeddingExperiment.__init__(
            self,
            data_config=data_config,
            model_config=model_config,
            save_freq=save_freq,
            pseudo_rgb=pseudo_rgb,
        )

        self.trained_model = None
        self.loss_dict = None
        self.label_weights = None

    def initialize_image_data_set(self,):
        super().initialize_image_data_set()

    def initialize_data_transform_pipeline(self, data_transform_pipelines: str = None):
        super().initialize_data_transform_pipeline(
            data_transform_pipelines=data_transform_pipelines
        )

    def initialize_data_loader_dict(self, drop_last_batch: bool = True):

        dh = DataHandler(
            dataset=self.data_set,
            batch_size=self.batch_size,
            num_workers=15,
            random_state=self.random_state,
            transformation_dicts=self.data_transform_pipeline_dicts,
            drop_last_batch=drop_last_batch,
        )
        dh.stratified_train_val_test_split(splits=self.train_val_test_split)
        dh.get_data_loader_dict(shuffle=True)
        self.data_loader_dict = dh.data_loader_dict
        self.label_weights = dh.dataset.label_weights
        self.data_set = dh.dataset

    def initialize_exp_config(self):
        super().initialize_exp_config()

    def train_models(self):
        self.exp_config.data_loader_dict = self.data_loader_dict
        (
            self.trained_model,
            self.loss_dict,
            self.best_loss_dict,
        ) = train_val_test_loop(
            output_dir=self.output_dir,
            exp_config=self.exp_config,
            num_epochs=self.num_epochs,
            early_stopping=self.early_stopping,
            device=self.device,
            save_freq=self.save_freq,
        )

    def load_model(self, weights_fname):
        super().load_model(weights_fname=weights_fname)

    def extract_and_save_latents(self, output_dir=None):
        super().extract_and_save_latents(output_dir=self.output_dir)

    def visualize_loss_evolution(self):
        super().visualize_loss_evolution()

    def evaluate_test_performance(self):
        super().evaluate_test_performance()

    def save_preds_labels(self, dataset_types=None):
        if dataset_types is None:
            dataset_types = ["train", "val", "test"]
        for dataset_type in dataset_types:
            preds, labels, idc = get_preds_labels(
                exp_config=self.exp_config, dataset_type=dataset_type
            )
            pred_label_df = pd.DataFrame(preds, columns=["prediction"], index=idc)
            pred_label_df["labels"] = labels
            pred_label_df.to_csv(
                os.path.join(self.output_dir, "pred_label_{}.csv".format(dataset_type))
            )


class EmbeddingCustomSplitExperiment(EmbeddingExperiment):
    def __init__(
        self,
        output_dir: str,
        data_config: dict,
        model_config: dict,
        batch_size: int = 64,
        num_epochs: int = 64,
        early_stopping: int = -1,
        random_state: int = 42,
        save_freq: int = -1,
        pseudo_rgb: bool = False,
    ):
        super().__init__(
            output_dir=output_dir,
            data_config=data_config,
            model_config=model_config,
            train_val_test_split=None,
            batch_size=batch_size,
            num_epochs=num_epochs,
            early_stopping=early_stopping,
            random_state=random_state,
            save_freq=save_freq,
            pseudo_rgb=pseudo_rgb,
        )

        self.test_nuclei_metadata_file = None
        self.val_nuclei_metadata_file = None
        self.train_nuclei_metadata_file = None
        self.test_data_set = None
        self.val_data_set = None
        self.train_data_set = None

    def initialize_image_data_set(self):
        self.data_key = self.data_config.pop("data_key")
        self.label_key = self.data_config.pop("label_key")
        if (
            "extra_features" in self.data_config
            and len(self.data_config["extra_features"]) > 0
        ):
            self.extra_feature_key = "extra_features"
        if "index_key" in self.data_config:
            self.index_key = self.data_config.pop("index_key")

        self.train_nuclei_metadata_file = self.data_config.pop(
            "train_nuclei_metadata_file"
        )
        self.val_nuclei_metadata_file = self.data_config.pop("val_nuclei_metadata_file")
        self.test_nuclei_metadata_file = self.data_config.pop(
            "test_nuclei_metadata_file"
        )

        self.data_config["nuclei_metadata_file"] = self.train_nuclei_metadata_file
        self.train_data_set = init_image_dataset(**self.data_config)

        self.data_config["nuclei_metadata_file"] = self.val_nuclei_metadata_file
        self.val_data_set = init_image_dataset(**self.data_config)

        self.data_config["nuclei_metadata_file"] = self.test_nuclei_metadata_file
        self.test_data_set = init_image_dataset(**self.data_config)

        self.data_set = TorchTransformableSuperset(
            datasets=[self.train_data_set, self.val_data_set, self.test_data_set]
        )

    def initialize_data_transform_pipeline(self, data_transform_pipelines: str = None):
        super().initialize_data_transform_pipeline(
            data_transform_pipelines=data_transform_pipelines
        )

    def initialize_data_loader_dict(self, drop_last_batch: bool = True):
        dh = DataHandler(
            dataset=self.data_set,
            batch_size=self.batch_size,
            num_workers=15,
            random_state=self.random_state,
            transformation_dicts=self.data_transform_pipeline_dicts,
            drop_last_batch=drop_last_batch,
        )

        dh.train_val_test_datasets_dict = {
            "train": self.train_data_set,
            "val": self.val_data_set,
            "test": self.test_data_set,
        }
        dh.get_data_loader_dict(shuffle=True)
        self.data_loader_dict = dh.data_loader_dict
        self.label_weights = dh.dataset.label_weights
        self.data_set = dh.dataset

    def initialize_exp_config(self):
        super().initialize_exp_config()
        self.exp_config.data_loader_dict = self.data_loader_dict

    def train_models(self):
        super().train_models()

    def load_model(self, weights_fname):
        super().load_model(weights_fname=weights_fname)

    def extract_and_save_latents(self):
        super().extract_and_save_latents()

    def visualize_loss_evolution(self):
        super().visualize_loss_evolution()

    def evaluate_test_performance(self):
        super().evaluate_test_performance()

    def save_preds_labels(self, dataset_types=None):
        super().save_preds_labels(dataset_types=dataset_types)
