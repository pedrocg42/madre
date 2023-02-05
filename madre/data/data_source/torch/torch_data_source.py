from torch.utils.data import Dataset

from madre.data.data_source import DataSource


class TorchDataSource(DataSource):
    def __init__(self, dataset: Dataset):

        self.__dataset = dataset
