from abc import ABC, abstractmethod


class DataSource(ABC):
    @abstractmethod
    def build(self, *args, **kwargs):
        pass
