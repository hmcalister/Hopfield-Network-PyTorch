from tqdm import tqdm
import torch

from . import AbstractLearningRule, HebbianLearningRule
from .. import HopfieldNetwork


class DeltaLearningRule(AbstractLearningRule):
    def __init__(self, network, learningRate: float=1, maxEpochs: int = 100):
        super().__init__()
        self.network: HopfieldNetwork = network
        self.learningRate = learningRate
        self.maxEpochs = maxEpochs

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        W = self.network.weightMatrix
        self.network.weightMatrix = HebbianLearningRule()(states)

        epoch = 0
        epochsProgressbar = tqdm(total=self.maxEpochs, desc="Delta Learning Rule Epochs")
        while not torch.all(self.network.stable(states)) and epoch < self.maxEpochs:
            steppedStates = torch.clone(states)
            self.network.stepStates(steppedStates)

            self.network.weightMatrix += self.learningRate * (states-steppedStates)@states.T
            self.network.weightMatrix = self.network.weightMatrix.fill_diagonal_(0)
            # self.network.weightMatrix = (self.network.weightMatrix + self.network.weightMatrix.T) / 2

            epoch += 1
            epochsProgressbar.update(1)
        epochsProgressbar.close()
        
        weightMatrix = self.network.weightMatrix
        weightMatrix = self._cleanupStep(weightMatrix)
        self.network.weightMatrix = W
        return weightMatrix