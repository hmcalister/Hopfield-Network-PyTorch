from HopfieldNetworkPyTorch.HopfieldNetwork import HopfieldNetwork, LearningRule
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def createBipolarStates(nStates: int, dimension: int):
    X = 0.5*torch.ones(size=(dimension, nStates))
    X = 2*torch.bernoulli(X)-1
    return X

def measureSimilarities(learnedStates, states, trialStr):
    similarities = torch.abs(learnedStates.T @ states)
    mostSimilar = torch.max(similarities, axis=0).values / x.shape[0]
    print(trialStr)
    print("\tFinal State Average Similarity to Memory:\t", torch.mean(mostSimilar).item())
    print("\tFinal State Average Standard Deviation of Similarity:\t", torch.std(mostSimilar).item())

dimension = 100
nLearnedStates = 10
nStates = 1000
learningRule = LearningRule.HebbianLearningRule()

learnedStates = createBipolarStates(nLearnedStates, dimension).to(device)
X = createBipolarStates(nStates, dimension).to(device)

x = X.clone()
measureSimilarities(learnedStates, x, "Initial Similarity")

network = HopfieldNetwork(dimension, learningRule, device)
network.learnMemories(learnedStates)
x = X.clone()
network.relaxStates(x)
measureSimilarities(learnedStates, x, "Trained Similarity")
