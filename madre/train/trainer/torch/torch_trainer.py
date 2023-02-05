from core.model import TorchModel
from loguru import logger
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer

from madre.core.experiment import Experiment
from madre.train.train_looper import TorchTrainLooper
from madre.train.trainer import Trainer


class TorchTrainer(Trainer):
    def __init__(
        self,
        train_looper: TorchTrainLooper,
        criteria: _Loss,
        optimizer=Optimizer,
        **experiment,
    ):
        super().__init__(**experiment)

        self._train_looper = train_looper
        self._criteria = criteria
        self._optimizer = optimizer

    def build(self, model: TorchModel, experiment: Experiment):

        logger.info(" > Building model")
        self._model = model(experiment)
        self._model.to(experiment.device)
        logger.info(self._model)
        logger.info(
            f"> > Total parameters: {sum(param.numel() for param in self._model.parameters())}"
        )

        self._optimizer_ = self._optimizer(
            self._model.parameters(), lr=self._learning_rate
        )

    def _train(self, experiment):

        for epoch in range(self._num_epochs):

            logger.info(f"Training epoch {epoch} of {self._num_epochs}")

            self._train_looper.train_epoch(
                model=self._model,
                optimizer=self._optimizer,
                criteria=self._criteria,
                train_data_source=experiment.train_data_source,
            )

            if experiment.val_data_source is not None:
                self._train_looper.val_epoch(
                    model=self._model,
                    optimizer=self._optimizer,
                    criteria=self._criteria,
                    val_data_source=experiment.val_data_source,
                )
