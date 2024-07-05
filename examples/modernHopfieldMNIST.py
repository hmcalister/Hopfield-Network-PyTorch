from HopfieldNetworkPyTorch.ModernHopfieldNetwork import ModernHopfieldNetwork, InteractionFunction
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import torch
import torchvision
device = "cuda" if torch.cuda.is_available() else "cpu"

imageShape = (28, 28)
def displayStatesAsImage(statesTensor: torch.Tensor, numImages: int, fig_kw: dict = {}) -> tuple[Figure, list[Axes]]:
    """
    Given a set of tensors of shape (imageShape[0]*imageShape[1]*numClasses, N), take only the image neurons of the first numImages items and display them.
    """

    numSubplot = np.ceil(np.sqrt(numImages)).astype(int)
    fig, axes = plt.subplots(numSubplot, numSubplot, **fig_kw)
    for ax in np.ravel(axes):
        ax.axis("off")
    
    for itemIndex, ax in zip(range(numImages), np.ravel(axes)):
        targetMemory: np.ndarray = statesTensor[:imageShape[0]*imageShape[1], itemIndex].to("cpu").detach().numpy()
        targetMemory = targetMemory.reshape(imageShape)
        ax.imshow(targetMemory)
        # ax.imshow(targetMemory, vmin=-1, vmax=1)
    
    return fig, axes

# --------------------------------------------------------------------------------
# Load MNIST dataset
# --------------------------------------------------------------------------------

threshold = 0.8
datasetRoot: str = "datasets/MNIST"

mnistTrain = torchvision.datasets.MNIST(root=datasetRoot, download=True, train=True)
mnistTest = torchvision.datasets.MNIST(root=datasetRoot, download=True, train=False)

train = mnistTrain.data.reshape(-1, 28*28)
train = torch.where(train < threshold, -1, 1)
y_train = torch.nn.functional.one_hot(mnistTrain.targets)
y_train = 2*y_train-1
train = torch.concat((train, y_train), dim=1)
train = train.type(torch.float64).T

test = mnistTest.data.reshape(-1, 28*28)
test = torch.where(test < threshold, -1, 1)
y_test = torch.nn.functional.one_hot(mnistTest.targets)
y_test = 2*y_test-1
test = torch.concat((test, y_test), dim=1)
test = test.type(torch.float64).T

# displayStatesAsImage(train[:, :28*28], 25, fig_kw={"figsize": (12,12)})
# plt.suptitle("Example Images from Training Data")
# plt.show()

# --------------------------------------------------------------------------------
# Create Hopfield network of specific size
# --------------------------------------------------------------------------------

# All pixels plus all class neurons
dimension = 28*28+10
nMemories = 256

interactionVertex = 10
temperature = 1.1
itemBatchSize = 128
neuronBatchSize = None

# Only update the class neurons
neuronMask = torch.arange(dimension-10, dimension)

interactionFunc = InteractionFunction.LeakyRectifiedPolynomialInteractionFunction(n=interactionVertex)
network = ModernHopfieldNetwork(dimension, nMemories, device, interactionFunc, itemBatchSize, neuronBatchSize)

# displayStatesAsImage(network.memories[:, :28*28], 25, fig_kw={"figsize": (12,12)})
# plt.suptitle("Initial Memories")
# plt.show()

X = train[:, :10000].to(device)
network.learnMemories(X,
                        maxEpochs = 100,
                        initialLearningRate = 1e-1,
                        learningRateDecay = 0.999,
                        momentum = 0.6,
                        initialTemperature=temperature,
                        finalTemperature=temperature,
                        errorPower = 1,
                        clampMemories=True,
                        neuronMask=neuronMask,
                        verbose=1,
                    )
del X

displayStatesAsImage(network.memories[:, :28*28], 25, fig_kw={"figsize": (12,12)})
plt.suptitle("Learned Memories")
plt.show()