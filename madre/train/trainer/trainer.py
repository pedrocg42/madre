from abc import ABC, abstractmethod


class Trainer(ABC):
    def __call__(self, **kwargs):
        return self._train(**kwargs)

    @abstractmethod
    def _train(self, *args, **kwargs):
        pass
