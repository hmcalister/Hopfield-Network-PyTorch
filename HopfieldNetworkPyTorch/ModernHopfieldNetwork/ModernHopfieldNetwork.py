import numpy as np
from tqdm import tqdm

import torch

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
        self.memories = torch.rand(size=(self.dimension, nMemories), requires_grad=True, device=torchDevice)
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
                        batchSize: int = None,
                        errorPower: int = 1,
                        precision: float = 1.0e-30,
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
        :param batchSize: The size of batches. Defaults to None, which sets the batchSize to the number of states (X.shape[1])
        :param errorPower: The power to apply to the error when summing the loss
        :param precision: The minimum precision of the weight update, avoids division by zero errors
        :param verbose: An integer to indicate verbosity
            - 0: No output by epoch
            - 1: A progress bar over the epochs
            - 2: A newline for each epoch
        :return: A list of the loss history over the epochs
        """

        if batchSize is None:
            batchSize = X.shape[1]

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
            numBatches = np.ceil(X.shape[1] / batchSize).astype(int)
            batches = torch.chunk(X, numBatches, dim=1)
            for batchIndex in range(numBatches):
                batch = batches[batchIndex].detach()
                currentBatchSize = batch.shape[1]
                
                tiledBatchClampOn = torch.tile(batch, (1,self.dimension))
                tiledBatchClampOff = torch.clone(tiledBatchClampOn)
                for d in range(self.dimension):
                    tiledBatchClampOn[d,d*currentBatchSize:(d+1)*currentBatchSize] = 1
                    tiledBatchClampOff[d,d*currentBatchSize:(d+1)*currentBatchSize] = -1
                onSimilarity = self.interactionFunction(self.memories.T @ tiledBatchClampOn)
                offSimilarity = self.interactionFunction(self.memories.T @ tiledBatchClampOff)
                Y = torch.tanh(beta*torch.sum(onSimilarity-offSimilarity, axis=0)).reshape(batch.shape)
                
                loss = torch.sum((Y - batch)**(2*errorPower))
                loss /= (currentBatchSize * self.dimension)
                loss.backward()

                with torch.no_grad():
                    epochGrads = self.memories.grad
                    memoryGrads = momentum * memoryGrads + epochGrads
                    maxGradMagnitude = torch.max(torch.abs(memoryGrads), axis=0).values.reshape(1, self.memories.shape[1])
                    maxGradMagnitude[maxGradMagnitude<precision] = precision
                    maxGradMagnitudeTiled = torch.tile(maxGradMagnitude, (self.dimension, 1))
                    self.memories -= learningRate * memoryGrads / maxGradMagnitudeTiled
                    self.memories = self.memories.clamp_(-1,1)
                    self.memories.grad = None
                    epochTotalLoss += loss.item() 

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
    def stepStates(self, X: torch.Tensor, batchSize: int = None):
        """
        Step the given states according to the energy difference rule. 
        Step implies only a single update is made, no matter if the result is stable or not.

        Note X must have shape (network.dimension, nStates) where n is the number of states to update
        X must already be moved to the correct device. This can be done with X.to(network.device)

        :param X: The tensor of states to step. 
            Tensor must be on the correct device and have shape (network.dimension, nStates)
        :param batchSize: The size of batches. Defaults to None, which sets the batchSize to the number of states (X.shape[1])
        """

        if batchSize is None:
            batchSize = X.shape[1]

        numBatches = np.ceil(X.shape[1] / batchSize).astype(int)
        batches = torch.chunk(X, numBatches, dim=1)
        batchViewStartIndex = 0
        for batchIndex in range(numBatches):
            batch = batches[batchIndex].detach()
            currentBatchSize = batch.shape[1]
            
            # First we make two tensors of shape (dimension, dimension*nStates)
            # 
            # The first index walks over the dimension, while the second holds a flattened copy
            # of each state in X. For each index in newShape[0] there is an entire copy of 
            # X with that particular index set to 1 (clampOn) or -1 (clampOff)
            #
            # So tiledStatesClampOn[0].reshape(X.shape) will return a tensor that looks
            # exactly like X but the entire first dimension is set to 1.
            tiledBatch = torch.tile(batch, (1,self.dimension))
            tiledBatchClampOn = torch.clone(tiledBatch)
            tiledBatchClampOff = torch.clone(tiledBatch)
            for d in range(self.dimension):
                tiledBatchClampOn[d,d*currentBatchSize:(d+1)*currentBatchSize] = 1
                tiledBatchClampOff[d,d*currentBatchSize:(d+1)*currentBatchSize] = -1
            onSimilarity = self.interactionFunction(self.memories.T @ tiledBatchClampOn)
            offSimilarity = self.interactionFunction(self.memories.T @ tiledBatchClampOff)
        
            Y = torch.sign(torch.sum(onSimilarity-offSimilarity, axis=0))
            Y[Y==0] = 1
            Y = torch.reshape(Y, batch.shape)
            X[:, batchViewStartIndex :batchViewStartIndex + currentBatchSize] = Y
            batchViewStartIndex += currentBatchSize

    @torch.no_grad()
    def relaxStates(self, X: torch.Tensor, maxIterations: int = 100, batchSize: int = None, verbose: bool = False):
        """
        Update the states some number of times.

        :param X: The tensor of states to step. 
            Tensor must be on the correct device and have shape (network.dimension, nStates)
        :param maxIterations: The integer number of iterations to update the states for.
        :param batchSize: The size of batches. Defaults to None, which sets the batchSize to the number of states (X.shape[1])
        :param verbose: Flag to show progress bar
        """
        
        for _ in tqdm(range(maxIterations), desc="Relax States", disable=not verbose):
            X_prev = X.clone()
            self.stepStates(X, batchSize=batchSize)
            if torch.all(X_prev == X):
                break
        