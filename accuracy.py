import torch

def EscapeAndOverkillRates(minibatchOutputVariable, minibatchTargetLabelsVariable):
    numberOfLabelsToFind = 0
    numberOfFalsePositives = 0
    numberOfFalseNegatives = 0
    numberOfSamples = minibatchOutputVariable.data.shape[0]
    numberOfAttributes = minibatchOutputVariable.data.shape[1]

    for sampleNdx in range(numberOfSamples):
        for attributeNdx in range(numberOfAttributes):
            predictionIsPositive = minibatchOutputVariable.data[sampleNdx, attributeNdx] > 0.5
            targetIsPositive = minibatchTargetLabelsVariable.data[sampleNdx, attributeNdx] > 0
            if targetIsPositive:
                numberOfLabelsToFind += 1
            if predictionIsPositive and not targetIsPositive:
                numberOfFalsePositives += 1
            if not predictionIsPositive and targetIsPositive:
                numberOfFalseNegatives += 1
    return numberOfFalseNegatives/numberOfLabelsToFind, numberOfFalsePositives/(numberOfSamples * numberOfAttributes - numberOfLabelsToFind)