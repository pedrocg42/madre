import yaml


class Experiment:
    def __init__(self, path: str):

        with open(path, "r") as f:
            self.__experiment = yaml.load(f)

        for key in self.__experiment:
            setattr(self, key, self.__experiment[key])
