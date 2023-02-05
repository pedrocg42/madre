from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def build(self, *args, **kwargs):
        pass
