from HopfieldNetworkPyTorch.ModernHopfieldNetwork import ModernHopfieldNetwork, InteractionFunction
from HopfieldNetworkPyTorch.utils import createBipolarStates
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def measureSimilarities(learnedStates, states, trialStr):
    similarities = torch.abs(learnedStates.T @ states)
    mostSimilar = torch.max(similarities, axis=0).values / x.shape[0]
    print(trialStr)
    print("\tFinal State Average Similarity to Memory:\t", torch.mean(mostSimilar).item())
    print("\tFinal State Average Standard Deviation of Similarity:\t", torch.std(mostSimilar).item())

# The dimension of the Hopfield Network
dimension = 64

# The number of memory vectors to use, only for Learned Memory Storage
nMemories = 10

# The number of states to learn
nLearnedStates = 20

# The number of probe states
nStates = 100

# Other hyperparameters for learning

interactionVertex = 3
temperature = 50
itemBatchSize = 20
neuronBatchSize = 16

learnedStates = createBipolarStates(dimension, nLearnedStates).to(device)
X = createBipolarStates(dimension, nStates).to(device)
interactionFunc = InteractionFunction.RectifiedPolynomialInteractionFunction(n=interactionVertex)

# -----------------------------------------------------------------------------
# Initial Similarities
# -----------------------------------------------------------------------------
x = X.clone()
measureSimilarities(learnedStates, x, "Initial Similarity")

# -----------------------------------------------------------------------------
# Direct Memory Storage
# -----------------------------------------------------------------------------
network = ModernHopfieldNetwork(dimension, nLearnedStates, device, interactionFunc, itemBatchSize, neuronBatchSize)
network.setMemories(learnedStates)
x = X.clone()
network.relaxStates(x)
measureSimilarities(learnedStates, x, "Direct Memory Storage")

# -----------------------------------------------------------------------------
# Learned Memory Storage
# -----------------------------------------------------------------------------
network = ModernHopfieldNetwork(dimension, nMemories, device, interactionFunc, itemBatchSize, neuronBatchSize)
network.learnMemories(learnedStates,
                        maxEpochs = 500,
                        initialLearningRate = 1e-2,
                        learningRateDecay = 0.999,
                        momentum = 0.6,
                        initialTemperature=temperature,
                        finalTemperature=temperature,
                        errorPower = 2,
                        verbose=1
                    )

x = X.clone()
network.relaxStates(x)
measureSimilarities(learnedStates, x, "Learned Memory Storage")