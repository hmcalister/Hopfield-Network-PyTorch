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

dimension = 100
nMemories = 10
nLearnedStates = 20
nStates = 1000
interactionVertex = 3
temperature = 40
batchSize = 128

learnedStates = createBipolarStates(dimension, nLearnedStates).to(device)
X = createBipolarStates(dimension, nStates).to(device)
interactionFunc = InteractionFunction.RectifiedPolynomialInteractionFunction(n=interactionVertex)

x = X.clone()
measureSimilarities(learnedStates, x, "Initial Similarity")

# Direct Memory Storage
network = ModernHopfieldNetwork(dimension, nLearnedStates, device, interactionFunc)
network.setMemories(learnedStates)
x = X.clone()
network.relaxStates(x, batchSize=batchSize)
measureSimilarities(learnedStates, x, "Direct Memory Storage")


# Learned Memories
network = ModernHopfieldNetwork(dimension, nMemories, device, interactionFunc)
network.learnMemories(learnedStates,
                        maxEpochs = 1000,
                        initialLearningRate = 4e-2,
                        learningRateDecay = 0.998,
                        momentum = 0.6,
                        batchSize = batchSize,
                        initialTemperature=temperature,
                        finalTemperature=temperature,
                        errorPower = 2,
                        verbose=1
                    )

x = X.clone()
network.relaxStates(x, batchSize=batchSize)
measureSimilarities(learnedStates, x, "Learned Memory Storage")