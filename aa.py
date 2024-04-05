import numpy as np
from hmcalisterHopfieldUtils.prototype import createUniformlyRandomVector, prototypeGenerationWithBernoulliVector
from hmcalisterHopfieldUtils.hopfield import bipolarHeaviside
from tqdm import tqdm

dimension = 13
numMemories = 17
numPrototypes = 10
numLearnedStates = 23
numStates = 11

prototypes = createUniformlyRandomVector(numPrototypes, dimension)
learnedStates = prototypeGenerationWithBernoulliVector(prototypes, 0.2, numLearnedStates)
learnedStates = bipolarHeaviside(np.array(learnedStates)).T
states = createUniformlyRandomVector(numStates, dimension)
states = bipolarHeaviside(np.array(states)).T

interactionVertexPower = 30
errorPower = 2
beta = 1e-3
initialLearningRate = 4.0e-2
learningRateDecay = 0.998
momentum = 0.6
maxEpochs = 1000
batchSize = numLearnedStates

memories=np.random.normal(0.0, 1.0, (numMemories, dimension))
memoryGrads = np.zeros((numMemories, dimension))

auxillary=-np.ones((dimension, dimension*batchSize))
for d in range(dimension):
    auxillary[d, d*batchSize:(d+1)*batchSize]=1

trainingError=[]
for epoch in range(maxEpochs):
    learningRate=initialLearningRate*learningRateDecay**epoch

    shuffleIndices = np.random.permutation(numLearnedStates)
    learnedStates = learnedStates[:, shuffleIndices]
    epochTotalCorrect = 0
    for batchIndex in range(numLearnedStates//batchSize):
        # Shape: (dimension, batchSize)
        batch=learnedStates[:,batchIndex*batchSize:(batchIndex+1)*batchSize]
        # Shape: (1, dimension*batchSize)
        batchFlattened = np.reshape(batch, (1,dimension*batchSize))

        # Shape: (dimension, dimension*batchSize)
        # Now we have dimension copies of each state
        # So tiledBatch[:, n*batchSize] is the same state for any n
        tiledBatch = np.tile(batch, (1, dimension))

        tiledBatchInverted = np.copy(tiledBatch)
        for d in range(dimension):
            tiledBatchInverted[d, d*batchSize:(d+1)*batchSize] *= -1

        # Shape: (numMemories, dimension*batchSize)
        rectifiedMemoriesSimilarity = np.maximum(np.dot(memories, tiledBatch), 0)
        rectifiedMemoriesInvertedSimilarity = np.maximum(np.dot(memories, tiledBatchInverted), 0)
        
        # TODO: Ensure the subtraction is correctly orientated  
        Y = np.tanh(beta*np.sum(
            rectifiedMemoriesSimilarity**interactionVertexPower - rectifiedMemoriesInvertedSimilarity**interactionVertexPower,
            axis=0))

        # TODO: Ensure the subtraction is correctly orientated  
        derivativeOfMemories = np.dot(
            np.tile((batchFlattened-Y)**(2*errorPower-1)*(1-Y**2), (numMemories, 1))*rectifiedMemoriesInvertedSimilarity**(interactionVertexPower-1), tiledBatchInverted.T
        ) - np.dot(
            np.tile((batchFlattened-Y)**(2*errorPower-1)*(1-Y**2), (numMemories, 1))*rectifiedMemoriesSimilarity**(interactionVertexPower-1), tiledBatch.T
        )

        # Memories, Dimension
        memoryGrads = momentum*memoryGrads + derivativeOfMemories
        maxGradMagnitude = np.max(np.absolute(memoryGrads), axis=1).reshape(numMemories, 1)
        maxGradMagnitudeTiled = np.tile(maxGradMagnitude, (1,dimension)) + 0.00001
        memories -= learningRate * memoryGrads / maxGradMagnitudeTiled
        memories = np.clip(memories, -1, 1)

        Yr = Y.reshape((dimension, batchSize))
        Yr = bipolarHeaviside(Yr)
        numCorrect = np.sum(np.all(batch == Yr, axis=0))
        epochTotalCorrect += numCorrect
    err = (1.0 - epochTotalCorrect/numLearnedStates)
    print(f"Epoch {epoch:04}\n\tError: {err:0.4f}\n\tAverage |Grad|: {np.average(np.absolute(memoryGrads))}")

        