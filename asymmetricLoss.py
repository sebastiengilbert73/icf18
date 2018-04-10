import torch

class AsymmetricL1Loss(torch.nn.Module):
    def __init__(self, penaltyForFalseNegative):
        super(AsymmetricL1Loss, self).__init__()
        self.penaltyForFalseNegative = penaltyForFalseNegative

    def forward (self, computedOutputVariable, targetOutputVariable):
        numberOfSamples = computedOutputVariable.data.shape[0]
        #numberOfAttributes = computedOutputVariable.data.shape[1]
        differenceVariable = computedOutputVariable - targetOutputVariable
        minusDifferenceVariable = -1.0 * differenceVariable
        lossTensorVariable = self.penaltyForFalseNegative * torch.abs(minusDifferenceVariable) + 1.0 * torch.abs(differenceVariable)

        """sum = 0
        for sampleNdx in range(numberOfSamples):
            for attributeNdx in range(numberOfAttributes):
                computedOutput = computedOutputVariable.data[sampleNdx][attributeNdx]
                targetOutput = targetOutputVariable.data[sampleNdx][attributeNdx]
                difference = computedOutput - targetOutput
                if difference > 0: # computedOutput > targetOutput -> False positive
                    sum += difference
                else: # targetOutput > computedOutput -> false negative
                    sum += self.penaltyForFalseNegative * abs(difference)
        """
        return (1.0 / numberOfSamples) * torch.sum(lossTensorVariable)