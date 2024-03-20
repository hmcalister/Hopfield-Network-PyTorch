import numpy as np
from tqdm import tqdm

import torch

from .InteractionFunction import AbstractInteractionFunction

class ModernHopfieldNetwork():
    
    def __init__(self, dimension: int, nMemories: int):
        self.dimension = dimension
        self.memories = torch.rand(size=(self.dimension, nMemories))
        self.neurons = torch.zeros(self.dimension)

    def setMemories(self, memories: torch.Tensor | np.ndarray):
        if memories.shape[0] != self.dimension:
            raise ValueError("memories should have shape (network.dimension, network.nMemories)")

        self.memories = torch.tensor(memories, dtype=torch.float32)
    
    def learnMemories(self, trainingData: torch.Tensor):
        """
        TODO: Implement backprop to learn memories representing training data.
        """

        raise NotImplementedError