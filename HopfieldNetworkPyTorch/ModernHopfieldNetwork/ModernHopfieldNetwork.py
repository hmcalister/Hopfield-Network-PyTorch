import numpy as np
from tqdm import tqdm
from typing import Callable

import torch

from HopfieldNetworkPyTorch import utils

from .InteractionFunction import AbstractInteractionFunction

class ModernHopfieldNetwork():
    
    def __init__(self, dimension: int, nMemories: int, torchDevice: str, interactionFunction: AbstractInteractionFunction, itemBatchSize: int = None, neuronBatchSize: int = None):
        """
        Create a new modern Hopfield network with the specified dimension and number of memories.
        Note the interaction function must implement InteractionFunction.AbstractInteractionFunction, which exposes a variable n representing the interaction vertex
        If using direct memory storage (that is, not learning the memories), calling network.setMemories after the constructor will allow for memories to be placed into the network.

        :param dimension: The dimension of the network.
        :param nMemories: The number of memories the network will hold. Memories should be in the range [-1,1]
        :param torchDevice:  The pytorch device to store the memories on, e.g. "cpu" or "cuda".
        :param interactionFunction: An implementation of InteractionFunction.AbstractInteractionFunction.
        :param itemBatchSize: Sets the batch size for items, i.e. how many items are processed at once. None (default) indicates no batching, process all items at once.
        :param neuronBatchSize: Sets the batch size for neurons, i.e. how many neurons are processed at once. None (default) indicates no batching, process all neurons at once.
        """
        
        self.dimension = dimension
        self.memories = torch.rand(size=(self.dimension, nMemories), requires_grad=True, device=torchDevice, dtype=torch.float64)
        self.interactionFunction = interactionFunction

        self.itemBatchSize = itemBatchSize
        self.neuronBatchSize = neuronBatchSize

    def setItemBatchSize(self, itemBatchSize: int) :
        self.itemBatchSize = itemBatchSize

    def setNeuronBatchSize(self, neuronBatchSize: int) :
        self.neuronBatchSize = neuronBatchSize

    def setMemories(self, memories: torch.Tensor):
        """
        Set the memories of the network directly. Note the memories must be moved to the preferred device before being passed.

        :param memories: The new memories of the network. Must be of shape (network.dimension, nMemories) and be moved to the preferred device.
        """
        if memories.shape != self.memories.shape:
            raise ValueError("memories should have shape (network.dimension, nMemories)")
        self.memories = memories.requires_grad_().to(self.memories.device)
    
    def learnMemories(self, X: torch.Tensor,
                        maxEpochs: int = 100,
                        initialLearningRate: float = 0.1,
                        learningRateDecay: float = 1.0,
                        momentum: float = 0.0,
                        initialTemperature: float = 100,
                        finalTemperature: float = 100,
                        errorPower: int = 1,
                        precision: float = 1.0e-30,
                        neuronMask: torch.Tensor = None,
                        verbose: int = 2,
                      ):
        """
        Stabilize a set of states X by gradient descent and back propagation.
        Mostly implements the methods detailed in Krotov and Hopfield 2016 (appendix A).

        :param X: States, a tensor of shape (network.dimension, n)
            These are the states will be stabilized by learning, i.e. the Learned States.
        :param maxEpochs: The maximum number of epochs to train for
        :param initialLearningRate: The learning rate of the backpropagation
        :param learningRateDecay: The decay of learningRate per epoch
        :param momentum: The value of momentum for the gradient descent
        :param initialTemperature: The initial temperature of the network.
            Controls the slope of the tanh activation function, beta = 1/(temperature**interactionVertex)
        :param finalTemperature: The final temperature of the network.
        :param errorPower: The power to apply to the error when summing the loss
        :param precision: The minimum precision of the weight update, avoids division by zero errors
        :param neuronMask: A mask of neuron indices to update during learning. 
            If passed, only the specified indices will be updated (have energy difference calculated). Other indices will be clamped.
            If None (default), all indices will be updated.
        :param verbose: An integer to indicate verbosity
            - 0: No output by epoch
            - 1: A progress bar over the epochs
            - 2: A newline for each epoch
        :return: A list of the loss history over the epochs
        """

        # The neurons to train, either masked by the function call or all neurons
        neuronMask = neuronMask if neuronMask is not None else torch.arange(X.shape[0])
        # The size of neuron-wise batches. If not passed, use all neurons in one batch
        neuronBatchSize = self.neuronBatchSize if self.neuronBatchSize is not None else X.shape[0]
        # Calculate the number of neuron batches. Note this will smooth the number of neurons in each batch,
        # so the passed neuronBatchSize is more of an upper limit
        numNeuronBatches = np.ceil(neuronMask.shape[0] / neuronBatchSize).astype(int)
        # Get the neuron batches, sets of indices to train at once
        neuronBatches = torch.chunk(neuronMask, numNeuronBatches)

        # A tensor of all item indices
        itemIndices = torch.arange(X.shape[1])
        # The size of the item-wise batches. If not passed, use all items in one batch
        itemBatchSize = self.itemBatchSize if self.itemBatchSize is not None else X.shape[1]
        # Calculate the number of item batches. Note this will smooth the number of items in each batch,
        # so the passed itemBatchSize is more of an upper limit
        numItemBatches = np.ceil(itemIndices.shape[0] / itemBatchSize).astype(int)

        history = []
        interactionVertex = self.interactionFunction.n
        memoryGrads = torch.zeros_like(self.memories).to(self.memories.device)
        epochProgressbar = tqdm(range(maxEpochs), desc="Epoch", disable=(verbose!=1))
        for epoch in range(maxEpochs):
            epochTotalLoss = 0

            # Shuffle the learned items so we are not learning the exact same batches each epoch
            shuffledIndices = torch.randperm(X.shape[1])
            X = X[:, shuffledIndices]
            
            # Determine the value of beta for this epoch
            learningRate = initialLearningRate*learningRateDecay**epoch
            temperature = initialTemperature + (finalTemperature-initialTemperature) * epoch/maxEpochs
            beta = 1/(temperature**interactionVertex)

            # Determine the batches for this epoch, based on the newly shuffled states and the previously calculated batch numbers
            itemBatches = torch.chunk(X, numItemBatches, dim=1)
            for itemBatchIndex in range(numItemBatches):
                itemBatch = itemBatches[itemBatchIndex].detach()
                currentItemBatchSize = itemBatch.shape[1]

                itemBatchLoss = 0
                for neuronBatchIndex in range(numNeuronBatches):
                    neuronIndices = neuronBatches[neuronBatchIndex].detach()
                    neuronBatchNumIndices = neuronIndices.shape[0]
                
                    tiledBatchClampOn = torch.tile(itemBatch, (1,neuronBatchNumIndices))
                    tiledBatchClampOff = torch.clone(tiledBatchClampOn)
                    for i, d in enumerate(neuronIndices):
                        tiledBatchClampOn[d,i*currentItemBatchSize:(i+1)*currentItemBatchSize] = 1
                        tiledBatchClampOff[d,i*currentItemBatchSize:(i+1)*currentItemBatchSize] = -1
                    onSimilarity = self.interactionFunction(self.memories.T @ tiledBatchClampOn)
                    offSimilarity = self.interactionFunction(self.memories.T @ tiledBatchClampOff)
                    Y = torch.tanh(beta*torch.sum(onSimilarity-offSimilarity, axis=0)).reshape([neuronBatchNumIndices, currentItemBatchSize])
                    
                    neuronBatchLoss = torch.sum((Y - itemBatch[neuronIndices])**(2*errorPower))
                    itemBatchLoss += neuronBatchLoss

                itemBatchLoss.backward()
                with torch.no_grad():
                    epochGrads = self.memories.grad
                    memoryGrads = momentum * memoryGrads + epochGrads
                    maxGradMagnitude = torch.max(torch.abs(memoryGrads), axis=0).values.reshape(1, self.memories.shape[1])
                    maxGradMagnitude[maxGradMagnitude<precision] = precision
                    maxGradMagnitudeTiled = torch.tile(maxGradMagnitude, (self.dimension, 1))
                    self.memories -= learningRate * memoryGrads / maxGradMagnitudeTiled
                    self.memories = self.memories.clamp_(-1,1)
                    self.memories.grad = None
                    epochTotalLoss += itemBatchLoss.item() / (neuronMask.shape[0] * itemIndices.shape[0])

            history.append(epochTotalLoss)
            if verbose==1:
                epochProgressbar.set_postfix({"Loss": f"{history[-1]:4e}"})
                epochProgressbar.update()
            if verbose==2:
                print(f"Epoch {epoch:04}: Loss {history[-1]}")

        return history
    
    @torch.no_grad()
    def energy(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculates and returns the energy of a set of states.
        Energy is calculated as the interaction function applied to the sum of all memories dot the state.
        -F(sum M.T @ x)

        :param X: States, a tensor of shape (network.dimension, nStates).
        :return: A tensor of shape (nStates) measuring the energy of each state.
        """

        return -self.interactionFunction(torch.sum(self.memories.T @ X, axis=0))
    
    # def stable(self, X: torch.Tensor):
    #     """
    #     Calculate the stability of each state given.

    #     :param X: States, a tensor of shape (network.dimension, nStates).
    #     :returns: A (nStates) tensor of booleans with each entry the stability of a state.
    #     """

    @torch.no_grad()
    def stepStates(self, X: torch.Tensor, neuronMask: torch.Tensor = None, activationFunction: Callable = utils.BipolarHeaviside):
        """
        Step the given states according to the energy difference rule. 
        Step implies only a single update is made, no matter if the result is stable or not.

        Note X must have shape (network.dimension, nStates) where n is the number of states to update
        X must already be moved to the correct device. This can be done with X.to(network.device)

        :param X: The tensor of states to step. 
            Tensor must be on the correct device and have shape (network.dimension, nStates)
        :param neuronMask: A mask of neuron indices to update. If passed, only the specified indices are updated. Other indices will be clamped.
            If None (default), all indices will be updated.
        :param activationFunction: The function to apply to the resulting step. For complex activation functions, use
            currying via lambda (e.g. `lambda X: torch.nn.Softmax(dim=0)(X)`)
        """

        neuronMask = neuronMask if neuronMask is not None else torch.arange(X.shape[0])
        neuronBatchSize = self.neuronBatchSize if self.neuronBatchSize is not None else X.shape[0]
        numNeuronBatches = np.ceil(neuronMask.shape[0] / neuronBatchSize).astype(int)
        neuronIndexBatches = torch.chunk(neuronMask, numNeuronBatches)

        itemIndices = torch.arange(X.shape[1])
        itemBatchSize = self.itemBatchSize if self.itemBatchSize is not None else X.shape[1]
        numItemBatches = np.ceil(itemIndices.shape[0] / itemBatchSize).astype(int)
        itemIndexBatches = torch.chunk(itemIndices, numItemBatches)

        for itemBatchIndices in itemIndexBatches:
            itemBatchIndices = itemBatchIndices.detach()
            currentItemBatchSize = itemBatchIndices.shape[0]
            items = X[:, itemBatchIndices]

            for neuronBatchIndices in neuronIndexBatches:
                neuronBatchIndices = neuronBatchIndices.detach()
                neuronBatchNumIndices = neuronBatchIndices.shape[0]

                tiledBatchClampOn = torch.tile(items, (1,neuronBatchNumIndices))
                tiledBatchClampOff = torch.clone(tiledBatchClampOn)
                for i, d in enumerate(neuronBatchIndices):
                    tiledBatchClampOn[d,i*currentItemBatchSize:(i+1)*currentItemBatchSize] = 1
                    tiledBatchClampOff[d,i*currentItemBatchSize:(i+1)*currentItemBatchSize] = -1
                onSimilarity = self.interactionFunction(self.memories.T @ tiledBatchClampOn)
                offSimilarity = self.interactionFunction(self.memories.T @ tiledBatchClampOff)
                
                Y = activationFunction(torch.sum(onSimilarity-offSimilarity, axis=0))
                Y = torch.reshape(Y, [neuronBatchNumIndices, currentItemBatchSize])
                X[neuronBatchIndices[:, None], itemBatchIndices] = Y

    @torch.no_grad()
    def relaxStates(self, X: torch.Tensor, maxIterations: int = 100, neuronMask: torch.Tensor = None, verbose: bool = False):
        """
        Update the states some number of times.

        :param X: The tensor of states to step. 
            Tensor must be on the correct device and have shape (network.dimension, nStates)
        :param maxIterations: The integer number of iterations to update the states for.
        :param neuronMask: A mask of neuron indices to update. If passed, only the specified indices are updated. Other indices will be clamped.
            If None (default), all indices will be updated.
        :param verbose: Flag to show progress bar
        """
        
        for _ in tqdm(range(maxIterations), desc="Relax States", disable=not verbose):
            X_prev = X.clone()
            self.stepStates(X, neuronMask)
            if torch.all(X_prev == X):
                break
        