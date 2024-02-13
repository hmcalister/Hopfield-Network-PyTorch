from . import AbstractLearningRule

import torch

class HebbianLearningRule(AbstractLearningRule):
    def __init__(self):
        super().__init__()

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        weightMatrix = torch.einsum("bi,bj->ij", states.T, states.T)
        weightMatrix = self._cleanupStep(weightMatrix)
        return weightMatrix