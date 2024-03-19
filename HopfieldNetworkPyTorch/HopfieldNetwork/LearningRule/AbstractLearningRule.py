from abc import ABC, abstractmethod
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
        # matrix = (matrix + matrix.T) / 2

        matrixNorm = torch.max(matrix.abs())
        # matrixNorm = matrix.abs().sum()
        if matrixNorm > 0:
            matrix /= matrixNorm
        return matrix
