import torch
from .AbstractLearningRule import AbstractLearningRule
from .. import HopfieldNetwork


class DeltaLearningRule(AbstractLearningRule):
    def __init__(self, network: HopfieldNetwork, learningRate: float=1, maxEpochs: int = 100):
        super().__init__(learningRate, maxEpochs)
        self.network: HopfieldNetwork = network

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        steppedStates = torch.clone(states)
        self.network.stepStates(steppedStates)
        weightMatrixUpdate = self.learningRate * (states-steppedStates)@states.T
        return self._cleanup(weightMatrixUpdate, states)