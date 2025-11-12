from . import AbstractLearningRule

import torch

class HebbianLearningRule(AbstractLearningRule):
    def __init__(self):
        super().__init__(learningRate=1, maxEpochs=1)

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        weightMatrix = states @ states.T
        weightMatrix = self._cleanupStep(weightMatrix)
        return weightMatrix
    