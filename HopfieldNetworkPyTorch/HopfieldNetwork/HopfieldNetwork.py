from tqdm import tqdm
import numpy as np
import torch

from .LearningRule import AbstractLearningRule

class HopfieldNetwork():
    def __init__(self, dimension: int, learningRule: AbstractLearningRule, torchDevice: str):
        """
        Create a new Hopfield Network of the specified dimension.

        :param dimension: The dimension of the network.
        :param learningRule: The learning rule. Must implement LearningRule.AbstractLearningRule
        :param torchDevice:  The pytorch device to store the weight matrix on.
        """

        # The dimension of the network
        self.dimension = dimension

        # The learning rule of the network
        self.learningRule: AbstractLearningRule = learningRule

        # Initialize the weight matrix to zeros
        self.weightMatrix = torch.zeros(size=(dimension, dimension)).to(torchDevice)

    def learnMemories(self, X: torch.Tensor):
        """
        Train the network on the given states.
        Note before training the network, the user must manually set the value of network.learningRule to something OTHER than AbstractLearningRule.

        :param X torch.Tensor: The states to train on, a tensor of shape (network.dimension, numStates).
            This tensor is left untouched by the function call, if needed it is cloned. Do not clone it yourself.
        """
    
        self.weightMatrix += self.learningRule(X)

    def energy(self, X: torch.Tensor) -> torch.Tensor:
        """
        Calculate the energy of the given states. 
        Energy is given by $-0.5 W \dot X \otimes X$, i.e. the field of each neuron ($W \dot X$) elementwise multiplied by the state.
        Note the energy is negative if the neuron is stable.

        :param X torch.Tensor: The states to train on, a tensor of shape (network.dimension, numStates).
        :returns torch.Tensor: A (network.dimension, numStates) tensor with each entry the energy of neuron in a state.
        """

        stateFields = torch.matmul(self.weightMatrix, X)
        stateEnergies = -0.5 * torch.mul(X, stateFields)
        return stateEnergies

    def stable(self, X: torch.tensor) -> torch.Tensor:
        """
        Calculate the stability of each state given.
        Stability is determined by checking the energy of each neuron in a state. If any of the neurons in a state are unstable (E>0) that state is unstable.
        
        :param X torch.Tensor: States, a tensor of shape (network.dimension, numStates).
        :returns torch.Tensor: A (numStates) tensor of booleans with each entry the stability of a state.
        """
        
        return torch.all(self.energy(X)<=0, axis=0)

    def stepStates(self, X: torch.Tensor):
        """
        Step the given states by updating all neurons once.
        Note the tensor given is updated in place, so clone the tensor beforehand if required.
        
        :param X torch.Tensor: States, a tensor of shape (network.dimension, numStates).
        """

        updateOrder = np.arange(self.dimension)
        np.random.shuffle(updateOrder)

        for neuronIndex in updateOrder:
            stateFields = torch.matmul(self.weightMatrix[neuronIndex], X)
            X[neuronIndex, stateFields<=0] = -1
            X[neuronIndex, stateFields>0] = 1
        
    def relaxStates(self, X: torch.Tensor, maxIterations: int=100, verbose: bool=False):
        """
        Relax the given states until either all states are stable, or a specified number of epochs (network.maxEpochs) is reached.
        Note the tensor given is updated in place, so clone the tensor beforehand if required.

        :param X torch.Tensor: States, a tensor of shape (network.dimension, numStates).
        :param maxIterations int: The maximum number of iterations to relax states for
        :param verbose bool: Boolean to show progress bar
        """
        
        epoch = 0
        epochsProgressbar = tqdm(total=maxIterations, desc="Relax States", disable=not verbose)
        # loop while not everything is stable and the epoch is below the maximum reached
        while not torch.all(self.stable(X)) and epoch < maxIterations:
            self.stepStates(X)
            epoch+=1
            epochsProgressbar.update(1)
        epochsProgressbar.close()