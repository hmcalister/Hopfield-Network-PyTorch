import numpy as np
from tqdm import tqdm

import torch

from .InteractionFunction import AbstractInteractionFunction

class ModernHopfieldNetwork():
    
    def __init__(self, dimension: int, nMemories: int, torchDevice: str, interactionFunction: AbstractInteractionFunction):
        
         # The dimension of the network
        self.dimension = dimension
        
        # The memories of the network
        self.memories = torch.rand(size=(self.dimension, nMemories), requires_grad=True, device=torchDevice)
        
        # The interaction function of the network
        self.interactionFunction = interactionFunction

    def setMemories(self, memories: torch.Tensor):
        """
        Set the memories of the network directly. Note the memories must be moved to the preferred device before being passed.

        :param memories: The new memories of the network. Must be of shape (network.dimension, nMemories) and be moved to the preferred device.
        """
        if memories.shape != self.memories.shape:
            raise ValueError("memories should have shape (network.dimension, nMemories)")

        memories = memories.requires_grad_().to(self.memories.device)
        self.memories = memories
    
    def learnMemories(self, X: torch.Tensor,
                        maxEpochs: int = 100,
                        learningRate: float = 0.1,
                        learningRateDecay: float = 1.0,
                        momentum: float = 0.0,
                        beta: float = 0.01,
                        batchSize: int = 128,
                        errorPower: int = 1,
                        eps: float = 0,
                        epsLength: int = 5,
                        verbose: int = 2,
                      ):
        """
        Stabilize a set of states X by gradient descent and back propagation.
        Mostly implements the methods detailed in Krotov and Hopfield 2016 (appendix A).

        :param X: States, a tensor of shape (network.dimension, n)
            These are the states will be stabilized by learning
        :param maxEpochs: The maximum number of epochs to train for
        :param learningRate: The learning rate of the backpropagation
        :param learningRateDecay: The decay of learningRate per epoch
        :param momentum: The value of momentum for the gradient descent
        :param beta: Controls the slope of the tanh activation function
        :param batchSize: The size of batches
        :param errorPower: The power to apply to the error when summing the loss
        :param eps: An epsilon measure for early stopping. If loss decreases by less than epsilon for epsLength epochs, return.
            Set to 0 for no early stopping.
        :param epsLength: The number of epochs that loss must decrease less than eps for early stopping to trigger.
        :param verbose: An integer to indicate verbosity
            - 0: No output by epoch
            - 1: A progress bar over the epochs
            - 2: A newline for each epoch
        :return: A list of the loss history over the epochs
        """

        history = []
        previousGrads = torch.zeros_like(self.memories).to(self.memories.device)
        progressBar = tqdm(range(maxEpochs), desc="Epoch", disable=(verbose!=1))
        for epoch in progressBar:
            shuffledIndices = torch.randperm(X.size(1))
            X = X[:, shuffledIndices]
            batches = torch.chunk(X, 1+X.shape[1] // batchSize, dim=1)
            epochTotalLoss = 0

            for batch in batches:
                batch = batch.detach()
                loss = 0
                for i in np.arange(self.dimension):
                    x0 = batch.clone().detach()
                    x0[i, :] = 0
                    e0 = self.memories.T @ x0

                    memoryRow = self.memories[i, :].unsqueeze(1)
                    energyDifference = self.interactionFunction(memoryRow + e0) - self.interactionFunction(-memoryRow + e0)
                    energyDifference = energyDifference.sum(axis=0)
                    updateValue = torch.tanh(beta * energyDifference)
                    loss += (updateValue - batch[i, :]).pow(2*errorPower).sum()
                loss.backward()
                with torch.no_grad():
                    epochTotalLoss += loss.item() / (np.prod(batch.shape))
                    grads = momentum * previousGrads - self.memories.grad
                    previousGrads = grads
                    gradMaxVal = torch.max(torch.abs(grads))
                    self.memories += learningRate * grads / (gradMaxVal + 0.0001)

                    self.memories.grad.clamp_(-1,1)
                    self.memories.grad = None

            history.append(epochTotalLoss/len(batches))
            if verbose==1:
                progressBar.set_postfix({"Loss": f"{np.round(history[-1], 5):.5f}"})
            if verbose==2:
                print(f"Epoch {epoch:04}: Loss {history[-1]}")
            if eps != 0 and len(history)>epsLength:
                # An array of differences, positive if the loss is decreasing and negative otherwise
                differences =  history[-(epsLength+1):-1] - history[-1]
                # Check if any differences are negative i.e. loss increased?
                if np.all(differences<eps): break
            learningRate *= learningRateDecay

        return history
    
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

        updateOrder = np.arange(self.dimension)
        np.random.shuffle(updateOrder)

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
        
        for _ in tqdm(range(maxIterations), desc="Relax States", disable=not verbose):
            X_prev = X.clone()
            self.stepStates(X)
            if torch.all(X_prev == X):
                break
        