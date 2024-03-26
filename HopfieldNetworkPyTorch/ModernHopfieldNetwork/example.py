from HopfieldNetworkPyTorch.ModernHopfieldNetwork import ModernHopfieldNetwork, InteractionFunction
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def createBipolarStates(dimension: int, nStates: int):
    X = 2*torch.rand(size=(dimension, nStates))-1
    X = 2*torch.heaviside(X, torch.tensor(0.0))-1
    return X

dimension = 100
nMemories = 5
nStates = 10
memories = createBipolarStates(dimension, nMemories).to(device)
X = createBipolarStates(dimension, nStates).to(device)


interactionFunc = InteractionFunction.PolynomialInteractionFunction(n=2)
network = ModernHopfieldNetwork(dimension, nMemories, device, interactionFunc)
network.setMemories(memories)

x = X.clone()
network.relaxStates(x)
similarities = torch.abs(memories.T @ x)
print(torch.max(similarities, axis=0).values)