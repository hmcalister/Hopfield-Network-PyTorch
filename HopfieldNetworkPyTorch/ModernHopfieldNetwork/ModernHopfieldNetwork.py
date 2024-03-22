import numpy as np
from tqdm import tqdm

import torch

from .InteractionFunction import AbstractInteractionFunction

class ModernHopfieldNetwork():
    
    def __init__(self, dimension: int, nMemories: int, torchDevice: str):
        
         # The dimension of the network
        self.dimension = dimension
        
        # The memories of the network
        self.memories = torch.rand(size=(self.dimension, nMemories))
        
        # The pytorch device to perform calculations on.
        self.device  = torchDevice
        
        # The interaction function of the network
        self.interactionFunction: AbstractInteractionFunction

    def setMemories(self, memories: torch.Tensor | np.ndarray):
        if memories.shape[0] != self.dimension:
            raise ValueError("memories should have shape (network.dimension, network.nMemories)")

        self.memories = torch.tensor(memories, dtype=torch.float32).to(self.device)
    
    def learnMemories(self, trainingData: torch.Tensor):
        """
        TODO: Implement backprop to learn memories representing training data.
        """

        raise NotImplementedError
    
    def energy(self, X: torch.Tensor):
        """
        Calculates and returns the energy of a set of states.
        Energy is calculated as the interaction function applied to the sum of all memories dot the state.
        -F(sum M.T @ x)

        Note X must have shape (self.dimension, n) where n is the number of states 
        """

        return -self.interactionFunction(torch.sum(self.memories.T @ X, axis=0))


    def stepStates(self, X: torch.Tensor):
        """
        Step the given states according to the energy difference rule. 
        Step implies only a single update is made, no matter if the result is stable or not.

        Note X must have shape (self.dimension, n) where n is the number of states to update
        """

        updateOrder = np.arange(0, self.dimension)
        np.random.shuffle(updateOrder)

        for i in updateOrder:
            X_invert = X.clone()
            X_invert[i, :] *= -1
            energyDifference = self.energy(X_invert) - self.energy(X)

            # If the energy is lower (hence, energy difference is negative) then flip
            # Note torch.where selects from the second arg when predicate is true, third when false
            X = torch.where(energyDifference >= 0, X[i, :], X_invert[i, :])
        