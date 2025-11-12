from . import AbstractLearningRule

import torch

class HebbianLearningRule(AbstractLearningRule):
    def __init__(self, learningRate: float = 1, maxEpochs: int = 1):
        super().__init__(learningRate, maxEpochs)

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        weightMatrix = states @ states.T
        weightMatrix = self._cleanupStep(weightMatrix)
        return weightMatrix
    