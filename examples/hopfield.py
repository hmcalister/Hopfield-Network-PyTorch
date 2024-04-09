from HopfieldNetworkPyTorch.HopfieldNetwork import HopfieldNetwork, LearningRule
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
nLearnedStates = 10
nStates = 1000
learningRule = LearningRule.HebbianLearningRule()

learnedStates = createBipolarStates(dimension, nLearnedStates).to(device)
X = createBipolarStates(dimension, nStates).to(device)

x = X.clone()
measureSimilarities(learnedStates, x, "Initial Similarity")

network = HopfieldNetwork(dimension, learningRule, device)
network.learnMemories(learnedStates)
x = X.clone()
network.relaxStates(x)
measureSimilarities(learnedStates, x, "Trained Similarity")
