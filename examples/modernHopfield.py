from HopfieldNetworkPyTorch.ModernHopfieldNetwork import ModernHopfieldNetwork, InteractionFunction
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def createBipolarStates(nStates: int, dimension: int):
    X = 2*torch.rand(size=(dimension, nStates))-1
    X = 2*torch.heaviside(X, torch.tensor(0.0))-1
    return X

def measureSimilarities(states, trialStr):
    similarities = torch.abs(learnedStates.T @ states)
    mostSimilar = torch.max(similarities, axis=0).values / dimension
    print(trialStr)
    print("\tFinal State Average Similarity to Memory:\t", torch.mean(mostSimilar).item())
    print("\tFinal State Average Standard Deviation of Similarity:\t", torch.std(mostSimilar).item())
    # print(states)
    # print(torch.mean(x, axis=1))


dimension = 100
nMemories = 10
nLearnedStates = 20
nStates = 1000
interactionVertex = 3

learnedStates = createBipolarStates(nLearnedStates, dimension).to(device)
X = createBipolarStates(nStates, dimension).to(device)
interactionFunc = InteractionFunction.RectifiedPolynomialInteractionFunction(n=interactionVertex)

x = X.clone()
measureSimilarities(x, "Initial Similarity")

# Direct Memory Storage
network = ModernHopfieldNetwork(dimension, nLearnedStates, device, interactionFunc)
network.setMemories(learnedStates)
x = X.clone()
network.relaxStates(x)
measureSimilarities(x, "Direct Memory Storage")


# Learned Memories
network = ModernHopfieldNetwork(dimension, nMemories, device, interactionFunc)
T = 40
network.learnMemories(learnedStates,
                        maxEpochs = 1000,
                        initialLearningRate = 4e-2,
                        learningRateDecay = 0.998,
                        momentum = 0.6,
                        batchSize = nLearnedStates,
                        initialTemperature=T,
                        finalTemperature=T,
                        errorPower = 2,
                        verbose=1
                    )

x = X.clone()
network.relaxStates(x)
measureSimilarities(x, "Learned Memory Storage")