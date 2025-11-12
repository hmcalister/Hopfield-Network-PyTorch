from tqdm import tqdm
import torch

from . import AbstractLearningRule, HebbianLearningRule
from .. import HopfieldNetwork


class DeltaLearningRule(AbstractLearningRule):
    def __init__(self, network, learningRate: float=1, maxEpochs: int = 100):
        super().__init__(learningRate, maxEpochs)
        self.network: HopfieldNetwork = network

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        steppedStates = torch.clone(states)
        self.network.stepStates(steppedStates)
        weightMatrixUpdate = self.learningRate * (states-steppedStates)@states.T
        return self._cleanupStep(weightMatrixUpdate)