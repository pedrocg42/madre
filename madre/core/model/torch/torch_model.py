from core.model import Model
from torch import Tensor, nn


class TorchModel(Model):
    def __init__(self, model: nn.Module):
        self._model = model

    def __call__(self, input: Tensor):
        return self._inference(input=input)

    def _inference(self, input: Tensor):
        return self._model(input)

    def on_train_epoch_start(self):
        self._model.train()

    def on_val_epoch_start(self):
        self._model.eval()
