import torch
import torchvision
import argparse
import ast
import os

import loadFromJson

print('trainer.py')

parser = argparse.ArgumentParser()
parser.add_argument('baseDirectory', help='The directory containing the files train.json and validation.json')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--architecture', help='The neural network architecture (Default: resnet18)', default='resnet18')
parser.add_argument('--learningRate', help='The learning rate (Default: 0.001)', type=float, default=0.001)
parser.add_argument('--momentum', help='The learning momentum (Default: 0.9)', type=float, default=0.9)
parser.add_argument('--imageSize', help='The image size (width, height) (Default: (256, 256))', default='(256, 256)')
parser.add_argument('--numberOfEpochs', help='Number of epochs (Default: 200)', type=int, default=200)
parser.add_argument('--minibatchSize', help='Minibatch size (Default: 64)', type=int, default=64)
parser.add_argument('--numberOfImagesForValidation', help='The number of images used for validation (Default: 128)', type=int, default=128)
parser.add_argument('--maximumNumberOfTrainingImages', help='The maximum number of training images (Default: 0, which means no limit)', type=int, default=0)


args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()

imageSize = ast.literal_eval(args.imageSize)

trainImporter = loadFromJson.Importer(os.path.join(args.baseDirectory, 'train.json'), args.maximumNumberOfTrainingImages)
validationImporter = loadFromJson.Importer(os.path.join(args.baseDirectory, 'validation.json'), 0,
                                           numberOfAttributes=trainImporter.NumberOfAttributes())

# Create a neural network, an optimizer and a loss function
if args.architecture == 'resnet18':
    imageSize = (224, 224)
    neuralNet = torchvision.models.resnet18(pretrained=True)
    for param in neuralNet.parameters():
        param.requires_grad = False
    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    neuralNet.fc =  torch.nn.Linear(512, trainImporter.NumberOfAttributes())  # Add a sigmoid final transformation

    # Create an optimizer
    optimizer = torch.optim.SGD(neuralNet.fc.parameters(), lr=args.learningRate, momentum=args.momentum)

    # Create a loss function
    lossFunction = torch.nn.MultiLabelMarginLoss()
else:
    raise NotImplementedError("trainer.py Architecture '{}' is not implemented".format(args.architecture))

if args.cuda:
    neuralNet.cuda() # Move to GPU

sigmoidFcn = torch.nn.Sigmoid()

minibatchIndicesListList = trainImporter.MinibatchIndices(args.minibatchSize)

# Initial validation loss before training
validationIndicesList = validationImporter.MinibatchIndices(args.numberOfImagesForValidation)[0] # Keep the first list
validationImgsTensor, validationLabelsTensor = validationImporter.Minibatch(validationIndicesList, imageSize)
if args.cuda:
    validationImgsTensor = validationImgsTensor.cuda()
    validationLabelsTensor = validationLabelsTensor.cuda()

# Validation loss
validationOutput = sigmoidFcn( neuralNet(torch.autograd.Variable(validationImgsTensor) ))
validationLoss = lossFunction(validationOutput, torch.autograd.Variable(validationLabelsTensor))
print("Epoch 0: Average train loss = ?; validationLoss = {}".format(validationLoss.data[0]))

for epoch in range(1, args.numberOfEpochs + 1):
    averageTrainLoss = 0
    for minibatchListNdx in range(len(minibatchIndicesListList)):
        #print ("minibatchListNdx = ", minibatchListNdx)
        print('.', end="", flush=True) # Print a dot without line return, right now
        minibatchIndicesList = minibatchIndicesListList[minibatchListNdx]
        thisMinibatchSize = len(minibatchIndicesList)
        minibatchImgsTensor, minibatchTargetLabelsTensor = trainImporter.Minibatch(minibatchIndicesList, imageSize)

        # Wrap in Variable
        minibatchImgsTensor = torch.autograd.Variable(minibatchImgsTensor)
        minibatchTargetLabelsTensor = torch.autograd.Variable(minibatchTargetLabelsTensor)

        if args.cuda:
            minibatchImgsTensor = minibatchImgsTensor.cuda()
            minibatchTargetLabelsTensor = minibatchTargetLabelsTensor.cuda()

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        actualOutput = sigmoidFcn( neuralNet(minibatchImgsTensor) ) # Add a sigmoid to the output of the neural network (the last layer is linear)

        # Loss
        loss = lossFunction(actualOutput, minibatchTargetLabelsTensor)
        #print ("loss =", loss)

        # Backward pass
        loss.backward()

        # Parameters update
        optimizer.step()

        averageTrainLoss += loss.data[0]
    averageTrainLoss = averageTrainLoss / len(minibatchIndicesListList)
    #print ("averageTrainLoss =", averageTrainLoss)

    # Validation
    validationIndicesList = validationImporter.MinibatchIndices(args.numberOfImagesForValidation)[0] # Keep the first list
    validationImgsTensor, validationLabelsTensor = validationImporter.Minibatch(validationIndicesList, imageSize)
    if args.cuda:
        validationImgsTensor = validationImgsTensor.cuda()
        validationLabelsTensor = validationLabelsTensor.cuda()
    # Validation loss
    validationOutput = sigmoidFcn( neuralNet(torch.autograd.Variable(validationImgsTensor) ))
    validationLoss = lossFunction(validationOutput, torch.autograd.Variable(validationLabelsTensor))
    #print ("validationLoss =", validationLoss)

    print("\nEpoch {}: Average train loss = {}; validationLoss = {}".format(epoch, averageTrainLoss,
                                                                                         validationLoss.data[0])),
                                                                                         #accuracy))