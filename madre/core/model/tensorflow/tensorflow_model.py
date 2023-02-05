from core.model import Model
from tensorflow.python.keras import Model as TFModel


class TensorflowModel(Model):
    def __init__(self, model: TFModel):
        self._model = model
