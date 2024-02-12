from abc import ABC, abstractmethod
from tqdm import tqdm
import numpy as np
import torch

class AbstractLearningRule(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        """
        Train the network on the given states.

        :param trainingStates torch.Tensor: The states to train on. Note the device of the states must be correctly set before being passed.
        :returns torch.Tensor: The weight matrix update.
        """
        pass
    
    def _cleanupStep(self, matrix: torch.Tensor) -> torch.Tensor:
        matrix = matrix.fill_diagonal_(0)
        matrixNorm = torch.max(matrix.abs())
        # matrixNorm = matrix.abs().sum()
        if matrixNorm > 0:
            matrix /= matrixNorm
        return matrix


class HebbianLearningRule(AbstractLearningRule):
    def __init__(self):
        super().__init__()

    def __call__(self, states: torch.Tensor) -> torch.Tensor:
        weightMatrix = torch.einsum("bi,bj->ij", states.T, states.T)
        weightMatrix = self._cleanupStep(weightMatrix)
        return weightMatrix
    
class DeltaLearningRule(AbstractLearningRule):
    def __init__(self, network, learningRate: float=1, maxEpochs: int = 100):
        super().__init__()
        self.network = network
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

            self.network.weightMatrix += self.learningRate * torch.einsum("bi,bj->ij", (states-steppedStates).T, states.T)
            self.network.weightMatrix = self.network.weightMatrix.fill_diagonal_(0)

            epoch += 1
            epochsProgressbar.update(1)
        epochsProgressbar.close()
        
        weightMatrix = self.network.weightMatrix
        weightMatrix = self._cleanupStep(weightMatrix)
        self.network.weightMatrix = W
        return weightMatrix