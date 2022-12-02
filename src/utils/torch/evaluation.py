import os
from typing import List

import imageio
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.helper.models import ModelConfig, ExpConfig
from src.utils.basic.export import dict_to_hdf
from src.utils.torch.general import get_device


def get_latent_representations_for_model(
    model: Module,
    dataset: Dataset,
    data_key: str = "seq_data",
    label_key: str = "label",
    extra_feature_key: str = None,
    index_key: str = "id",
    device: str = "cuda:0",
) -> dict:
    # create Dataloader
    dataloader = DataLoader(
        dataset=dataset, batch_size=32, shuffle=False, num_workers=15
    )

    latent_representations = []
    labels = []
    index = []
    model.eval().to(device)

    for (idx, sample) in enumerate(
        tqdm(dataloader, desc="Compute latents for the evaluation")
    ):
        input = sample[data_key]
        if label_key is not None:
            labels.extend(sample[label_key].detach().cpu().numpy())
        if index_key in sample:
            index.extend(sample[index_key])

        if extra_feature_key is not None:
            extra_features = sample[extra_feature_key].float().to(device)
        else:
            extra_features = None

        output = model(input, extra_features)
        latents = output["latents"]
        latent_representations.extend(latents.detach().cpu().numpy())

    latent_representations = np.array(latent_representations).squeeze()
    labels = np.array(labels).squeeze()

    latent_dict = {"latents": latent_representations}

    if len(labels) != 0:
        latent_dict["labels"] = labels
    if len(index) != 0:
        latent_dict["index"] = index

    return latent_dict


def save_latents_to_hdf(
    exp_config: ExpConfig,
    save_path: str,
    data_loader_dict: dict,
    dataset_type: str = "val",
    dataset: Dataset = None,
    device: str = "cuda:0",
):
    model = exp_config.model_config.model
    if dataset is None:
        try:
            dataset = data_loader_dict[dataset_type].dataset
        except KeyError:
            raise RuntimeError(
                "Unknown dataset_type: {}, expected one of the following: train, val,"
                " test".format(dataset_type)
            )
    save_latents_and_labels_to_hdf(
        model=model,
        dataset=dataset,
        save_path=save_path,
        data_key=exp_config.data_key,
        label_key=exp_config.label_key,
        index_key=exp_config.index_key,
        extra_feature_key=exp_config.extra_feature_key,
        device=device,
    )


def save_latents_and_labels_to_hdf(
    model: Module,
    dataset: Dataset,
    save_path: str,
    data_key: str = "image",
    label_key: str = "label",
    index_key: str = "id",
    extra_feature_key: str = None,
    device: str = "cuda:0",
):
    data = get_latent_representations_for_model(
        model=model,
        dataset=dataset,
        data_key=data_key,
        label_key=label_key,
        index_key=index_key,
        extra_feature_key=extra_feature_key,
        device=device,
    )

    expanded_data = {}
    if "latents" in data:
        latents = data["latents"]
        for i in range(latents.shape[1]):
            expanded_data["zs_{}".format(i)] = latents[:, i]
    if "index" in data:
        index = data["index"]
    else:
        index = None
    if "labels" in data:
        expanded_data["labels"] = data["labels"]

    dict_to_hdf(data=expanded_data, save_path=save_path, index=index)


def save_latents_from_model(
    output_dir: str,
    exp_config: ExpConfig,
    dataset_types: List[str] = None,
    device: str = "cuda:0",
):
    os.makedirs(output_dir, exist_ok=True)
    if dataset_types is None:
        dataset_types = ["train", "val"]

    for dataset_type in dataset_types:
        save_latents_to_hdf(
            exp_config=exp_config,
            save_path=output_dir
            + "/latent_representations_{}.csv.gz".format(dataset_type),
            dataset_type=dataset_type,
            device=device,
        )


def visualize_image_ae_performance(
    domain_model_config: ModelConfig,
    epoch: int,
    output_dir: str,
    phase: str,
    device: str = "cuda:0",
):
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    image_ae = domain_model_config.model.to(device)
    image_inputs = domain_model_config.inputs.to(device)
    size = image_inputs.size(2), image_inputs.size(3)

    recon_images = image_ae(image_inputs)["recons"]

    for i in range(image_inputs.size()[0]):
        imageio.imwrite(
            os.path.join(image_dir, "%s_epoch_%s_inputs_%s.jpg" % (phase, epoch, i)),
            np.uint8(image_inputs[i].cpu().data.view(size).numpy() * 255),
        )
        imageio.imwrite(
            os.path.join(image_dir, "%s_epoch_%s_recons_%s.jpg" % (phase, epoch, i)),
            np.uint8(recon_images[i].cpu().data.view(size).numpy() * 255),
        )


def get_confusion_matrices(
    exp_config: ExpConfig, dataset_types: List = ["test"], normalize=None
):
    confusion_matrices = {}
    for dataset_type in dataset_types:
        confusion_matrices[dataset_type] = get_confusion_matrix(
            exp_config, dataset_type, normalize=normalize
        )
    return confusion_matrices


def get_confusion_matrix(
    exp_config: ExpConfig, dataset_type: str = "test", normalize=None
):
    all_preds, all_labels, _ = get_preds_labels(
        exp_config=exp_config, dataset_type=dataset_type
    )
    return confusion_matrix(all_labels, all_preds, normalize=normalize)


def get_preds_labels(exp_config: ExpConfig, dataset_type: str = "test"):
    device = get_device()
    model = exp_config.model_config.model.to(device).eval()
    dataloader = exp_config.data_loader_dict[dataset_type]
    all_labels = []
    all_preds = []
    all_idc = []

    for sample in tqdm(dataloader, desc="Compute predictions"):
        # inputs = sample[exp_config.data_key].to(device)
        index = sample[exp_config.index_key]
        inputs = sample[exp_config.data_key]
        labels = sample[exp_config.label_key]
        if exp_config.extra_feature_key is not None:
            extra_features = sample[exp_config.extra_feature_key].float().to(device)
        else:
            extra_features = None
        outputs = model(inputs, extra_features)["outputs"]
        _, preds = torch.max(outputs, 1)

        all_labels.extend(list(labels.detach().cpu().numpy()))
        all_preds.extend(list(preds.detach().cpu().numpy()))
        all_idc.extend(list(index))
    return np.array(all_preds), np.array(all_labels), np.array(all_idc)
