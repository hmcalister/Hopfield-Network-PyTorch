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

        :param memories: The new memories of the network. Must be of shape (network.dimension, nMemories) and be moved to the preferred device.
        """
        if memories.shape != self.memories.shape:
            raise ValueError("memories should have shape (network.dimension, nMemories)")

        memories = memories.to(self.memories.device)
        self.memories = memories
    
    def learnMemories(self, X: torch.Tensor, learningRate: float, ):
        """
        TODO: Implement backprop to learn memories representing training data.

        :param X: States, a tensor of shape (network.dimension, nStates).
        """

        raise NotImplementedError
    
    def energy(self, X: torch.Tensor):
        """
        Calculates and returns the energy of a set of states.
        Energy is calculated as the interaction function applied to the sum of all memories dot the state.
        -F(sum M.T @ x)

        :param X: States, a tensor of shape (network.dimension, nStates).
        """

        return -self.interactionFunction(torch.sum(self.memories.T @ X, axis=0))
    
    def stable(self, X: torch.Tensor):
        """
        Calculate the stability of each state given.

        :param X: States, a tensor of shape (network.dimension, nStates).
        :returns: A (nStates) tensor of booleans with each entry the stability of a state.
        """

    def stepStates(self, X: torch.Tensor):
        """
        Step the given states according to the energy difference rule. 
        Step implies only a single update is made, no matter if the result is stable or not.

        Note X must have shape (self.dimension, nStates) where n is the number of states to update
        X must already be moved to the correct device. This can be done with X.to(network.device)

        :param X: The tensor of states to step. 
            Tensor must be on the correct device and have shape (network.dimension, nStates)
        """

        if X.device != self.memories.device:
            print(f"ERR: State device ({X.device}) does not match network device ({self.memories.device}). No computation performed.")
            return

        updateOrder = np.arange(self.dimension)
        # np.random.shuffle(updateOrder)

        for neuronIndex in updateOrder:
            x0 = X.clone()
            x0[neuronIndex, :] = 0
            e0 = self.memories.T @ x0

            memoryRow = self.memories[neuronIndex, :].unsqueeze(1)
            energyDifference = self.interactionFunction(memoryRow + e0) - self.interactionFunction(-memoryRow + e0)
            energyDifference = energyDifference.sum(axis=0)
            X[neuronIndex, :] = torch.where(energyDifference>=0, 1, -1)

    def relaxStates(self, X: torch.Tensor, maxIterations: int = 100, verbose: bool = False):
        """
        Update the states some number of times.

        :param X: The tensor of states to step. 
            Tensor must be on the correct device and have shape (network.dimension, nStates)
        :param maxIterations: The integer number of iterations to update the states for.
        :param verbose: Flag to show progress bar
        """

        if X.device != self.memories.device:
            print(f"ERR: State device ({X.device}) does not match network device ({self.memories.device}). No computation performed.")
            return
        
        for _ in tqdm(range(maxIterations), desc="Relax States", disable=not verbose):
            X_prev = X.clone()
            self.stepStates(X)
            if torch.all(X_prev == X):
                break
        