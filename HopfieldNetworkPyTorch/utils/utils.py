import torch

def createBipolarStates(dimension: int, numStates: int) -> torch.Tensor:
    """
    Create a new set of bipolar states (that is, states with values of -1/+1) of a given size.
    States are generated randomly, and are not guaranteed to be orthogonal, unique, or related.

    :param dimension: The dimension of each state.
    :param numStates: The number of states to generate.
    :returns: A tensor of shape (dimension, numStates) representing the bipolar states.
    """

    X = 0.5*torch.ones(size=(dimension, numStates))
    X = 2*torch.bernoulli(X)-1
    return X