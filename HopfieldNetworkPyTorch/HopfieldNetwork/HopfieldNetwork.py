from tqdm import tqdm
import numpy as np
import torch

from .LearningRule import AbstractLearningRule

class HopfieldNetwork():
    def __init__(self, dimension: int, torchDevice: str):
        """
        Create a new Hopfield Network of the specified dimension.

        Note before training the network, the user must manually set the value of network.learningRule to something OTHER than AbstractLearningRule

        :param dimension int: The dimension of the network.
        :param torchDevice str:  If CUDA is available, pass it here
        """

        # The dimension of the network
        self.dimension = dimension

        # The learning rule of the network
        self.learningRule: AbstractLearningRule

        # The pytorch device to perform calculations on.
        self.device  = torchDevice

        # Maximum number of epochs when relaxing states
        self.maxEpochs = 100

        self.weightMatrix = torch.zeros(size=(dimension, dimension)).to(self.device)

    def trainNetwork(self, X: torch.Tensor):
        """
        Train the network on the given states.
        Note before training the network, the user must manually set the value of network.learningRule to something OTHER than AbstractLearningRule.

        :param X torch.Tensor: The states to train on, a tensor of shape (network.dimension, numStates).
            This tensor is left untouched by the function call, if needed it is cloned. Do not clone it yourself.
        """
    
        self.weightMatrix += self.learningRule(X)

    def energyFunction(self, X: torch.Tensor) -> torch.Tensor:
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
        
        :param X torch.Tensor: The states to train on, a tensor of shape (network.dimension, numStates).
        :returns torch.Tensor: A (numStates) tensor of booleans with each entry the stability of a state.
        """
        
        return torch.all(self.energyFunction(X)<=0, axis=0)

    def stepStates(self, X: torch.Tensor):
        """
        Step the given states by updating all neurons once.
        Note the tensor given is updated in place, so clone the tensor beforehand if required.
        
        :param X torch.Tensor: The states to train on, a tensor of shape (network.dimension, numStates).
        """

        neuronOrder = np.random.permutation(self.dimension)
        for neuronIndex in neuronOrder:
            stateFields = torch.matmul(self.weightMatrix[neuronIndex], X)
            X[neuronIndex, stateFields<=0] = -1
            X[neuronIndex, stateFields>0] = 1
        
    def relaxStates(self, X: torch.Tensor, verbose: bool=False):
        """
        Relax the given states until either all states are stable, or a specified number of epochs (network.maxEpochs) is reached.
        Note the tensor given is updated in place, so clone the tensor beforehand if required.

        :param X torch.Tensor: The states to train on, a tensor of shape (network.dimension, numStates).
        :param verbose bool: Boolean to show progress bar
        """
        
        epoch = 0
        epochsProgressbar = tqdm(total=self.maxEpochs, desc="Relax States Epochs", disable=not verbose)
        # loop while not everything is stable and the epoch is below the maximum reached
        while not torch.all(self.stable(X)) and epoch < self.maxEpochs:
            self.stepStates(X)
            epoch+=1
            epochsProgressbar.update(1)
        epochsProgressbar.close()