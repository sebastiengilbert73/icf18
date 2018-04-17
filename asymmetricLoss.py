import torch

class AsymmetricL1Loss(torch.nn.Module):
    def __init__(self, penaltyForFalseNegativeVector):
        super(AsymmetricL1Loss, self).__init__()
        self.penaltyForFalseNegativeVector = penaltyForFalseNegativeVector

    def forward (self, computedOutputVariable, targetOutputVariable):
        numberOfSamples = computedOutputVariable.data.shape[0]
        numberOfAttributes = computedOutputVariable.data.shape[1]
        differenceVariable = computedOutputVariable - targetOutputVariable
        """minusDifferenceVariable = -1.0 * differenceVariable
        expandedPenaltyVariable = torch.autograd.Variable(self.penaltyForFalseNegativeVector.repeat(numberOfSamples, 1))
        if minusDifferenceVariable.is_cuda:
            expandedPenaltyVariable = expandedPenaltyVariable.cuda()
        lossTensorVariable = expandedPenaltyVariable * torch.abs(minusDifferenceVariable) + \
                                                     1.0 * torch.abs(differenceVariable)
        print ("numberOfSamples =", numberOfSamples)
        """
        lossTensorVariable = differenceVariable
        for sampleNdx in range(numberOfSamples):
            for attributeNdx in range(numberOfAttributes):
                if differenceVariable.data[sampleNdx, attributeNdx] < 0: # False negative
                    lossTensorVariable.data[sampleNdx, attributeNdx] = -1.0 * \
                        self.penaltyForFalseNegativeVector[attributeNdx] * differenceVariable.data[sampleNdx, attributeNdx]
        return (1.0 / numberOfSamples) * torch.sum(lossTensorVariable)

class AsymmetricL2Loss(torch.nn.Module):
    def __init__(self, penaltyForFalseNegativeVector):
        super(AsymmetricL2Loss, self).__init__()
        self.penaltyForFalseNegativeVector = penaltyForFalseNegativeVector

    def forward(self, computedOutputVariable, targetOutputVariable):
        numberOfSamples = computedOutputVariable.data.shape[0]
        numberOfAttributes = computedOutputVariable.data.shape[1]
        differenceVariable = computedOutputVariable - targetOutputVariable
        """minusDifferenceVariable = -1.0 * differenceVariable
        expandedPenaltyVariable = torch.autograd.Variable(self.penaltyForFalseNegativeVector.repeat(numberOfSamples, 1))
        if minusDifferenceVariable.is_cuda:
            expandedPenaltyVariable = expandedPenaltyVariable.cuda()"""
        lossTensorVariable = differenceVariable * differenceVariable
        for sampleNdx in range(numberOfSamples):
            for attributeNdx in range(numberOfAttributes):
                if differenceVariable.data[sampleNdx, attributeNdx] < 0:
                    lossTensorVariable.data[sampleNdx, attributeNdx] = self.penaltyForFalseNegativeVector[attributeNdx] * \
                                                                            lossTensorVariable.data[sampleNdx, attributeNdx]
        return (1.0 / numberOfSamples) * torch.sum(lossTensorVariable)

    def MultiplyPenaltiesBy(self, factor):
        print ("AsymmetricL2Loss.MultiplyPenaltiesBy(): Multiplying penalties by", factor)
        for attributeNdx in range(self.penaltyForFalseNegativeVector.shape[0]):
            self.penaltyForFalseNegativeVector[attributeNdx] = factor * self.penaltyForFalseNegativeVector[attributeNdx]

    def MovePenaltiesTowardOne(self, alpha):
        for attributeNdx in range(self.penaltyForFalseNegativeVector.shape[0]):
            currentPenalty = self.penaltyForFalseNegativeVector[attributeNdx]
            if currentPenalty > 1: # Ex.: alpha = 0.1; currentPenalty = 3: newPenalty = 1 + 0.9 * 2 = 2.8
                newPenalty = 1 + (1 - alpha) * (currentPenalty - 1)
            elif currentPenalty < 1: # Ex.: alpha = 0.1; currentPenalty = 0.6: newPenalty = 1 - 0.9 * 0.4 = 0.64
                newPenalty = 1 - (1 - alpha) * (1 - currentPenalty)
            self.penaltyForFalseNegativeVector[attributeNdx] = newPenalty