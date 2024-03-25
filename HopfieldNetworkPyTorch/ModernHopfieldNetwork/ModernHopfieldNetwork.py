import numpy as np
from tqdm import tqdm

import torch

from .InteractionFunction import AbstractInteractionFunction

class ModernHopfieldNetwork():
    
    def __init__(self, dimension: int, nMemories: int, torchDevice: str, interactionFunction: AbstractInteractionFunction):
        
         # The dimension of the network
        self.dimension = dimension
        
        # The memories of the network
        self.memories = torch.rand(size=(self.dimension, nMemories)).to(torchDevice)
        
        # The interaction function of the network
        self.interactionFunction = interactionFunction

    def setMemories(self, memories: torch.Tensor):
        """
        Set the memories of the network directly. Note the memories must be moved to the preferred device before being passed.

        :param memories: The new memories of the network. Must be of shape (network.dimension, network.nMemories) and be moved to the preferred device.
        """
        if memories.shape != self.memories.shape:
            raise ValueError("memories should have shape (network.dimension, network.nMemories)")

        self.memories = memories
    
    def learnMemories(self, X: torch.Tensor, learningRate: float, ):
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

        return -self.interactionFunction(self.memories.T @ X).sum(axis=0)

    def stepStates(self, X: torch.Tensor):
        """
        Step the given states according to the energy difference rule. 
        Step implies only a single update is made, no matter if the result is stable or not.

        Note X must have shape (self.dimension, n) where n is the number of states to update
        X must already be moved to the correct device. This can be done with X.to(network.device)

        :param X: The tensor of states to step. 
            Tensor must be on the correct device and have shape (dimension, n), for any n
        """

        if X.device != self.memories.device:
            print(f"ERR: State device ({X.device}) does not match network device ({self.memories.device}). No computation performed.")
            return

        updateOrder = np.arange(0, self.dimension)
        np.random.shuffle(updateOrder)

        for i in updateOrder:
            X_inverse = X.clone()
            X_inverse[i, :] *= -1
            energyDifference = self.energy(X_inverse) - self.energy(X)
            # print(i, energyDifference)
            
            # If the energy is lower (energy difference is negative) then flip
            # Note torch.where selects from the second arg when predicate is true, third when false
            X[i, :] = torch.where(energyDifference>=0, X[i, :], X_inverse[i, :])

    def updateStates(self, X: torch.Tensor, maxIterations: int = 100):
        """
        Update the states some number of times.

        :param X: The tensor of states to step. 
            Tensor must be on the correct device and have shape (dimension, n), for any n
        :param maxIterations: The integer number of iterations to update the states for.
        """

        if X.device != self.memories.device:
            print(f"ERR: State device ({X.device}) does not match network device ({self.memories.device}). No computation performed.")
            return

        for _ in tqdm(range(maxIterations)):
            X_prev = X.clone()
            self.stepStates(X)
            if torch.all(X_prev == X):
                break
        