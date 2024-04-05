import numpy as np
from tqdm import tqdm

import torch

from .InteractionFunction import AbstractInteractionFunction

class ModernHopfieldNetwork():
    
    def __init__(self, dimension: int, nMemories: int, torchDevice: str, interactionFunction: AbstractInteractionFunction):
        
         # The dimension of the network
        self.dimension = dimension
        
        # The memories of the network
        self.memories = torch.rand(size=(nMemories, self.dimension), requires_grad=True, device=torchDevice)
        
        # The interaction function of the network
        self.interactionFunction = interactionFunction

    def setMemories(self, memories: torch.Tensor):
        """
        Set the memories of the network directly. Note the memories must be moved to the preferred device before being passed.

        :param memories: The new memories of the network. Must be of shape (nMemories, network.dimension) and be moved to the preferred device.
        """
        if memories.shape != self.memories.shape:
            raise ValueError("memories should have shape (nMemories, network.dimension)")

        memories = memories.requires_grad_().to(self.memories.device)
        self.memories = memories
    
    def learnMemories(self, X: torch.Tensor,
                        maxEpochs: int = 100,
                        initialLearningRate: float = 0.1,
                        learningRateDecay: float = 1.0,
                        momentum: float = 0.0,
                        initialTemperature: float = 100,
                        finalTemperature: float = 100,
                        batchSize: int = 128,
                        errorPower: int = 1,
                        precision: float = 1.0e-30,
                        verbose: int = 2,
                      ):
        """
        Stabilize a set of states X by gradient descent and back propagation.
        Mostly implements the methods detailed in Krotov and Hopfield 2016 (appendix A).

        :param X: States, a tensor of shape (n, network.dimension)
            These are the states will be stabilized by learning, i.e. the Learned States.
        :param maxEpochs: The maximum number of epochs to train for
        :param initialLearningRate: The learning rate of the backpropagation
        :param learningRateDecay: The decay of learningRate per epoch
        :param momentum: The value of momentum for the gradient descent
        :param initialTemperature: The initial temperature of the network.
            Controls the slope of the tanh activation function, beta = 1/(temperature**interactionVertex)
        :param finalTemperature: The final temperature of the network.
        :param batchSize: The size of batches
        :param errorPower: The power to apply to the error when summing the loss
        :param precision: The minimum precision of the weight update, avoids division by zero errors
        :param verbose: An integer to indicate verbosity
            - 0: No output by epoch
            - 1: A progress bar over the epochs
            - 2: A newline for each epoch
        :return: A list of the loss history over the epochs
        """

        # We take the transpose so dimensions line up 
        # now of shape (dimension, nStates)
        X = X.T

        history = []
        memoryGrads = torch.zeros_like(self.memories).to(self.memories.device)
        epochProgressbar = tqdm(range(maxEpochs), desc="Epoch", disable=(verbose!=1))
        interactionVertex = self.interactionFunction.n

        for epoch in range(maxEpochs):
            learningRate = initialLearningRate*learningRateDecay**epoch
            temperature = initialTemperature + (finalTemperature-initialTemperature) * epoch/maxEpochs
            beta = 1/(temperature**interactionVertex)

            shuffledIndices = torch.randperm(X.shape[1])
            X = X[:, shuffledIndices]

            epochTotalLoss = 0
            for batchIndex in range(X.shape[1] // batchSize):
                batch = X[:, batchIndex*batchSize :(batchIndex+1)*batchSize].detach()
                
                tiledBatch = torch.tile(batch, (1,self.dimension))
                tiledBatchClampOn = torch.clone(tiledBatch)
                tiledBatchClampOff = torch.clone(tiledBatch)
                for d in range(self.dimension):
                    tiledBatchClampOn[d,d*batchSize:(d+1)*batchSize] = 1
                    tiledBatchClampOff[d,d*batchSize:(d+1)*batchSize] = -1
                onSimilarity = self.interactionFunction(self.memories @ tiledBatchClampOn)
                offSimilarity = self.interactionFunction(self.memories @ tiledBatchClampOff)
                Y = torch.tanh(beta*torch.sum(onSimilarity-offSimilarity, axis=0)).reshape(batch.shape)
                # loss = Y.sum()
                loss = torch.sum((Y - batch)**(2*errorPower))
                loss /= (batchSize * self.dimension)
                loss.backward()

                with torch.no_grad():
                    epochGrads = self.memories.grad
                    memoryGrads = momentum * memoryGrads + epochGrads
                    maxGradMagnitude = torch.max(torch.abs(memoryGrads), axis=1).values.reshape(self.memories.shape[0], 1)
                    maxGradMagnitude[maxGradMagnitude<precision] = precision
                    maxGradMagnitudeTiled = torch.tile(maxGradMagnitude, (1, self.dimension))
                    self.memories -= learningRate * memoryGrads / maxGradMagnitudeTiled
                    self.memories = self.memories.clamp_(-1,1)
                    self.memories.grad = None

                    # TODO: Loss
                    epochTotalLoss += loss.item() 

            history.append(epochTotalLoss)
            if verbose==1:
                epochProgressbar.set_postfix({"Loss": f"{history[-1]:4e}"})
                epochProgressbar.update()
            if verbose==2:
                print(f"Epoch {epoch:04}: Loss {history[-1]}")

        return history
    
    def energy(self, X: torch.Tensor):
        """
        Calculates and returns the energy of a set of states.
        Energy is calculated as the interaction function applied to the sum of all memories dot the state.
        -F(sum M.T @ x)

        :param X: States, a tensor of shape (nStates, network.dimension).
        """

        return -self.interactionFunction(torch.sum(self.memories @ X.T, axis=0))
    
    def stable(self, X: torch.Tensor):
        """
        Calculate the stability of each state given.

        :param X: States, a tensor of shape (nStates, network.dimension).
        :returns: A (nStates) tensor of booleans with each entry the stability of a state.
        """

    def stepStates(self, X: torch.Tensor):
        """
        Step the given states according to the energy difference rule. 
        Step implies only a single update is made, no matter if the result is stable or not.

        Note X must have shape (nStates, self.dimension) where n is the number of states to update
        X must already be moved to the correct device. This can be done with X.to(network.device)

        :param X: The tensor of states to step. 
            Tensor must be on the correct device and have shape (nStates, network.dimension)
        """

        states = X.T
        tiledStates = torch.tile(states, (1,self.dimension))
        tiledStatesClampOn = torch.clone(tiledStates)
        tiledStatesClampOff = torch.clone(tiledStates)
        for d in range(self.dimension):
            tiledStatesClampOn[d,d*X.shape[0]:(d+1)*X.shape[0]] = 1
            tiledStatesClampOff[d,d*X.shape[0]:(d+1)*X.shape[0]] = -1
        onSimilarity = self.interactionFunction(self.memories @ tiledStatesClampOn)
        offSimilarity = self.interactionFunction(self.memories @ tiledStatesClampOff)
        Y = torch.tanh(torch.sum(onSimilarity-offSimilarity, axis=0))
        X = torch.reshape(Y, states.shape)
        return X.T

    def relaxStates(self, X: torch.Tensor, maxIterations: int = 100, verbose: bool = False):
        """
        Update the states some number of times.

        :param X: The tensor of states to step. 
            Tensor must be on the correct device and have shape (nStates, network.dimension)
        :param maxIterations: The integer number of iterations to update the states for.
        :param verbose: Flag to show progress bar
        """
        
        for _ in tqdm(range(maxIterations), desc="Relax States", disable=not verbose):
            X_prev = X.clone()
            X = self.stepStates(X)
            if torch.all(X_prev == X):
                break
        return X
        