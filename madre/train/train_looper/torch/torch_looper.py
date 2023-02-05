from typing import Any, Dict, Tuple

import torch
from torch.nn.modules.loss import _Loss
from tqdm import tqdm

from madre.core.model import Model, TorchModel
from madre.data.data_source import TorchDataSource
from madre.train.optimizer import Optimizer
from madre.train.train_looper import TrainLooper


class TorchLooper(TrainLooper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def train_batch(
        self, model: Model, optimizer: Optimizer, criteria: _Loss, batch: Tuple
    ):

        input, labels = batch
        images = input.to(self._experiment.device)
        labels = labels.to(self._experiment.device)

        # Zero gradient before every batch
        optimizer.zero_grad()

        # Inference
        output = model(images)

        # Compute loss
        loss = criteria(output, labels)
        loss.backward()

        # Adjust weights
        optimizer.step()

        return {"loss": loss, "output": output, "labels": labels}

    def train_epoch(
        self,
        model: TorchModel,
        optimizer: Optimizer,
        criteria: _Loss,
        train_data_source: TorchDataSource,
    ):

        model.on_train_epoch_start()

        pbar = tqdm(train_data_source)
        for batch in pbar:

            result = self.train_batch(
                model=model, optimizer=optimizer, criteria=criteria, batch=batch
            )

            self._update_batch_history(result)

    @torch.no_grad()
    def val_batch(self, model: TorchModel, criteria: _Loss, batch: Tuple):

        input, labels = batch
        images = input.to(self._experiment.device)
        labels = labels.to(self._experiment.device)

        # Inference
        output = model(images)

        # Compute loss
        loss = criteria(output, labels)

        result = {"loss": loss, "output": output, "labels": labels}

        return result

    @torch.no_grad()
    def val_epoch(
        self,
        model: TorchModel,
        criteria: _Loss,
        val_data_source: TorchDataSource,
    ):

        model.on_val_epoch_start()

        pbar = tqdm(val_data_source)
        for batch in pbar:

            result = self.val_batch(model=model, criteria=criteria, batch=batch)

            self._update_batch_history(result, split="val")

    def _update_batch_history(self, result: Dict[Any, Any], split: str = "train"):

        if "batches_loss" in self._history.get(split, []):
            self._history[split]["batches_loss"] = {}

        i_batch = len(self._history[split]["batches_loss"])
        if self._history[split]["batches_loss"].get(i_batch) is None:
            self._history[split]["batches_loss"][i_batch] = []

        self._history[split]["batches_loss"][i_batch].append(result["loss"].item())
