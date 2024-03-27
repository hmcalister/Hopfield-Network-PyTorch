from HopfieldNetworkPyTorch.ModernHopfieldNetwork import ModernHopfieldNetwork, InteractionFunction
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def createBipolarStates(dimension: int, nStates: int):
    X = 2*torch.rand(size=(dimension, nStates))-1
    X = 2*torch.heaviside(X, torch.tensor(0.0))-1
    return X

dimension = 100
nMemories = 10
nStates = 100
memories = createBipolarStates(dimension, nMemories).to(device)
X = createBipolarStates(dimension, nStates).to(device)
interactionFunc = InteractionFunction.RectifiedPolynomialInteractionFunction(n=3)

# Direct Memory Storage
network = ModernHopfieldNetwork(dimension, nMemories, device, interactionFunc)
network.setMemories(memories)

x = X.clone()
network.relaxStates(x)
similarities = torch.abs(memories.T @ x)
mostSimilar = torch.max(similarities, axis=0).values
print("Direct Memory Storage")
print("\tFinal State Average Similarity to Memory:\t", torch.mean(mostSimilar).item())
print("\tFinal State Average Standard Deviation of Similarity:\t", torch.std(mostSimilar).item())

# Learned Memories
network = ModernHopfieldNetwork(dimension, 10*nMemories, device, interactionFunc)
network.learnMemories(memories, 
                      maxEpochs=250,
                      learningRate=0.05, 
                      learningRateDecay=0.999, 
                      momentum=0.8, 
                      beta=0.01,
                      errorPower=1,
                      verbose=1
                      )

x = X.clone()
network.relaxStates(x)
similarities = torch.abs(memories.T @ x)
mostSimilar = torch.max(similarities, axis=0).values
print("Learned Memory Storage")
print("\tFinal State Average Similarity to Memory:\t", torch.mean(mostSimilar).item())
print("\tFinal State Average Standard Deviation of Similarity:\t", torch.std(mostSimilar).item())