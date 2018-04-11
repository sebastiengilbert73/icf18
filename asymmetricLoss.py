import torch

class AsymmetricL1Loss(torch.nn.Module):
    def __init__(self, penaltyForFalseNegativeVector):
        super(AsymmetricL1Loss, self).__init__()
        self.penaltyForFalseNegativeVector = penaltyForFalseNegativeVector

    def forward (self, computedOutputVariable, targetOutputVariable):
        numberOfSamples = computedOutputVariable.data.shape[0]
        differenceVariable = computedOutputVariable - targetOutputVariable
        minusDifferenceVariable = -1.0 * differenceVariable
        print ("self.penaltyForFalseNegativeVector.shape =", self.penaltyForFalseNegativeVector.shape)
        print ("minusDifferenceVariable.data.shape =", minusDifferenceVariable.data.shape)
        print ("differenceVariable.data.shape =", differenceVariable.data.shape)
        lossTensorVariable = torch.autograd.Variable(self.penaltyForFalseNegativeVector.repeat(numberOfSamples)) * torch.abs(minusDifferenceVariable) + \
                                                     1.0 * torch.abs(differenceVariable)

        return (1.0 / numberOfSamples) * torch.sum(lossTensorVariable)