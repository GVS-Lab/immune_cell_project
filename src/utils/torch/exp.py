import copy
import logging
import os
import time
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from src.helper.models import ExpConfig, ModelConfig
from src.utils.torch.general import get_device


def train_val_test_loop(
    output_dir: str,
    exp_config: ExpConfig,
    num_epochs: int = 500,
    early_stopping: int = 20,
    device: str = None,
    save_freq: int = -1,
) -> Tuple[dict, dict, dict]:
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get available device, if cuda is available the GPU will be used
    if not device:
        device = get_device()

    # Store start time of the training
    start_time = time.time()

    # Initialize early stopping counter
    es_counter = 0
    if early_stopping < 0:
        early_stopping = num_epochs

    loss_dict = {"train": [], "val": [], "test": None}

    # Reserve space for best classifier weights
    best_model_weights = exp_config.model_config.model.cpu().state_dict()
    best_accuracy = 0
    best_epoch = -1

    logging.debug(
        "Start training of classifier {}".format(str(exp_config.model_config.model))
    )

    # Iterate over the epochs
    for i in range(num_epochs):
        logging.debug("---" * 20)
        logging.debug("---" * 20)
        logging.debug("Started epoch {}/{}".format(i + 1, num_epochs))
        logging.debug("---" * 20)

        # Check if early stopping is triggered
        if es_counter > early_stopping:
            logging.debug(
                "Training was stopped early due to no improvement of the validation"
                " balanced accuracy for {} epochs.".format(early_stopping)
            )
            break
        if i % save_freq == 0:
            checkpoint_dir = "{}/epoch_{}".format(output_dir, i)
            os.makedirs(checkpoint_dir, exist_ok=True)

        # Iterate over training and validation phase
        for phase in ["train", "val"]:
            epoch_statistics = process_single_epoch(
                exp_config=exp_config, phase=phase, device=device, epoch=i,
            )

            logging.debug(
                "{} LOSS STATISTICS FOR EPOCH {}: ".format(phase.upper(), i + 1)
            )

            if "loss" in epoch_statistics:
                logging.debug(
                    "Classification loss for : {:.8f}".format(epoch_statistics["loss"])
                )

            if "clf_accuracy" in epoch_statistics:
                logging.debug(
                    "Classification accuracy for: {:.8f}".format(
                        epoch_statistics["clf_accuracy"]
                    )
                )

            if "clf_balanced_accuracy" in epoch_statistics:
                logging.debug(
                    "Classification balanced accuracy: {:.8f}".format(
                        epoch_statistics["clf_balanced_accuracy"]
                    )
                )

            epoch_loss = epoch_statistics["loss"]
            loss_dict[phase].append(epoch_loss)
            logging.debug("***" * 20)
            logging.debug("Total {} loss: {:.8f}".format(phase, epoch_loss))
            logging.debug("***" * 20)

            if phase == "val":
                # Save classifier states if current parameters give the best validation loss

                if epoch_statistics["clf_balanced_accuracy"] > best_accuracy:
                    best_epoch = i
                    es_counter = 0
                    best_accuracy = epoch_statistics["clf_balanced_accuracy"]

                    best_model_weights = copy.deepcopy(
                        exp_config.model_config.model.cpu().state_dict()
                    )
                    best_model_weights = best_model_weights

                    torch.save(
                        best_model_weights,
                        "{}/best_model_weights.pth".format(output_dir),
                    )
                else:
                    es_counter += 1

            # Save classifier at checkpoints and visualize performance
            if i % save_freq == 0:
                torch.save(
                    exp_config.model_config.model.state_dict(),
                    "{}/classifier.pth".format(output_dir),
                )

    # Training complete
    time_elapsed = time.time() - start_time

    logging.debug("###" * 20)
    logging.debug(
        "Training completed in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, int(time_elapsed % 60)
        )
    )

    logging.debug("Best model found at epoch {}".format(best_epoch + 1))
    logging.debug("***" * 20)

    # Load best classifier
    exp_config.model_config.model.load_state_dict(best_model_weights)

    if "test" in exp_config.data_loader_dict:
        epoch_statistics = process_single_epoch(
            exp_config=exp_config, phase="test", device=device
        )

        logging.debug("TEST LOSS STATISTICS")

        if "loss" in epoch_statistics:
            logging.debug(
                "Test classification loss: {:.8f}".format(epoch_statistics["loss"])
            )

        if "clf_accuracy" in epoch_statistics:
            logging.debug(
                "Test classification accuracy: {:.8f}".format(
                    epoch_statistics["clf_accuracy"]
                )
            )

        if "clf_balanced_accuracy" in epoch_statistics:
            logging.debug(
                "Test classification balanced accuracy: {:.8f}".format(
                    epoch_statistics["clf_balanced_accuracy"]
                )
            )

        loss_dict["test"] = epoch_statistics["loss"]
        best_loss_dict = {
            "train": loss_dict["train"][best_epoch],
            "val": loss_dict["val"][best_epoch],
            "test": loss_dict["test"],
        }
        logging.debug("***" * 20)

        logging.debug("***" * 20)

        # Visualize classifier performance
        test_dir = "{}/test".format(output_dir)
        os.makedirs(test_dir, exist_ok=True)

        torch.save(
            exp_config.model_config.model.state_dict(),
            "{}/classifier.pth".format(test_dir),
        )

    return (
        exp_config.model_config.model,
        loss_dict,
        best_loss_dict,
    )


def process_single_epoch(
    exp_config: ExpConfig,
    phase: str = "train",
    device: str = "cuda:0",
    epoch: int = -1,
) -> dict:
    exp_model_config = exp_config.model_config
    data_loader_dict = exp_config.data_loader_dict
    data_loader = data_loader_dict[phase]
    data_key = exp_config.data_key
    label_key = exp_config.label_key
    extra_feature_key = exp_config.extra_feature_key

    # Initialize epoch statistics    recon_loss = 0
    loss = 0
    n_correct = 0
    n_total = 0
    labels = np.array([])
    preds = np.array([])

    # Iterate over batches
    for index, samples in enumerate(
        tqdm(data_loader, desc="Epoch {} progress for {} phase".format(epoch, phase))
    ):
        # Set model_configs
        exp_model_config.inputs = samples[data_key]
        exp_model_config.labels = samples[label_key]
        if extra_feature_key is not None:
            exp_config.extra_features = samples[extra_feature_key]

        batch_statistics = process_single_batch(
            model_config=exp_model_config, phase=phase, device=device,
        )

        if "n_correct" in batch_statistics:
            n_correct += batch_statistics["n_correct"]

        if "n_total" in batch_statistics:
            n_total += batch_statistics["n_total"]

        if "preds" in batch_statistics:
            preds = np.append(preds, batch_statistics["preds"])

        if "labels" in batch_statistics:
            labels = np.append(labels, batch_statistics["labels"])

        loss += batch_statistics["loss"]

    # Get average over batches for statistics
    loss /= len(data_loader.dataset)
    if n_total != 0:
        accuracy = n_correct / n_total
    else:
        accuracy = -1
    if len(preds) > 0:
        bac = balanced_accuracy_score(labels, preds)
    else:
        bac = -1

    epoch_statistics = {
        "total_loss": loss,
    }

    epoch_statistics["loss"] = loss
    epoch_statistics["clf_accuracy"] = accuracy
    epoch_statistics["clf_balanced_accuracy"] = bac

    return epoch_statistics


# Todo thin that function
def process_single_batch(
    model_config: ModelConfig,
    phase: str = "train",
    device: str = "cuda:0",
    model_base_type: str = None,
) -> dict:
    # Get all parameters of the configuration for domain i
    model = model_config.model
    optimizer = model_config.optimizer
    inputs = model_config.inputs
    labels = model_config.labels
    extra_features = model_config.extra_features
    train = model_config.trainable

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    # Set classifier to train if defined in respective configuration
    model.to(device)

    if phase == "train" and train:
        model.train()
        optimizer.zero_grad()
    else:
        model.eval()

    # Forward pass of the classifier
    inputs = inputs
    labels = torch.LongTensor(labels).to(device)

    if extra_features is not None:
        extra_features = extra_features.float().to(device)

    outputs = model(inputs, extra_features)

    outputs = outputs["outputs"]
    loss = model_config.loss_function(outputs, labels)
    _, preds = torch.max(outputs, 1)
    n_total = preds.size(0)
    n_correct = torch.sum(torch.eq(labels, preds)).item()

    # Backpropagate loss and update parameters if we are in the training phase
    if phase == "train":
        loss.backward()
        if train:
            optimizer.step()
            model.updated = True
        scheduler.step(loss)

    # Get summary statistics
    batch_size = labels.size(0)

    batch_statistics = dict()

    batch_statistics["loss"] = loss.item() * batch_size
    batch_statistics["n_correct"] = n_correct
    batch_statistics["n_total"] = n_total
    batch_statistics["preds"] = preds.detach().cpu().numpy()
    batch_statistics["labels"] = labels.detach().cpu().numpy()

    return batch_statistics
