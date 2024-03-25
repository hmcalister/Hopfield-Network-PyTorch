from HopfieldNetworkPyTorch.ModernHopfieldNetwork import ModernHopfieldNetwork, InteractionFunction
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def createBipolarStates(dimension: int, nStates: int):
    X = 2*torch.rand(size=(dimension, nStates))-1
    X = 2*torch.heaviside(X, torch.tensor(0.0))-1
    return X


dimension = 100
nMemories = 12
memories = createBipolarStates(dimension, nMemories).to(device)
interactionFunc = InteractionFunction.PolynomialInteractionFunction(n=1)

n = ModernHopfieldNetwork(dimension, nMemories, device, interactionFunc)
n.setMemories(memories)


nStates = 10
X = createBipolarStates(dimension, nStates).to(device)
n.updateStates(X)
