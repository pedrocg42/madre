from abc import ABC, abstractmethod

from madre.core.experiment import Experiment


class TrainLooper(ABC):
    def __init__(self, experiment: Experiment, **kwargs):

        self._history = {
            "train": {},
            "val": {},
        }
        self._experiment = experiment

    def __call__(self, *args, **kwargs):
        return self._train_epoch(*args, **kwargs)

    @abstractmethod
    def _train_epoch(self, *args, **kwargs):
        pass
