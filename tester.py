import torch
import torchvision
import argparse
import os
import ast

import loadFromJson

print ('tester.py')

parser = argparse.ArgumentParser()
parser.add_argument('baseDirectory', help='The directory containing the files test.json')
parser.add_argument('neuralNetworkFilename', help='The neural network filename')
parser.add_argument('--architecture', help='The neural network architecture (Default: resnet18)', default='resnet18')
parser.add_argument('--imageSize', help='The image size (width, height) (Default: (256, 256))', default='(256, 256)')
parser.add_argument('--numberOfLabels', help='The number of labels (Default: 228)', type=int, default=228)
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')

args = parser.parse_args()
args.cuda = not args.disable_cuda and torch.cuda.is_available()
imageSize = ast.literal_eval(args.imageSize)

testImporter = loadFromJson.Importer(os.path.join(args.baseDirectory, 'test.json'), 0, numberOfAttributes=args.numberOfLabels,
                                      testDataset=True)

# Create a neural network, an optimizer and a loss function
if args.architecture == 'resnet18':
    imageSize = (224, 224)
    expansion = 1
    neuralNet = torchvision.models.resnet18(pretrained=True)
    # Replace the last fully-connected layer
    neuralNet.fc =  torch.nn.Linear(512 * expansion, testImporter.NumberOfAttributes())
elif args.architecture == 'resnet152':
    imageSize = (224, 224)
    expansion = 4
    neuralNet = torchvision.models.resnet152(pretrained=True)
    # Replace the last fully-connected layer
    neuralNet.fc = torch.nn.Linear(512 * expansion, testImporter.NumberOfAttributes())
elif args.architecture == 'alexnet':
    imageSize = (224, 224)
    neuralNet = torchvision.models.alexnet(pretrained=True)
    # Replace the last fully-connected layer
    neuralNet.classifier = torch.nn.Sequential(
        torch.nn.Dropout(),
        torch.nn.Linear(256 * 6 * 6, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(),
        torch.nn.Linear(4096, 4096),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(4096, testImporter.NumberOfAttributes()),
    )
else:
    raise NotImplementedError("tester.py Architecture '{}' is not implemented".format(args.architecture))

if args.cuda:
    neuralNet.load_state_dict(torch.load(args.neuralNetworkFilename))
    neuralNet.cuda() # Move to GPU
else:
    neuralNet.load_state_dict(torch.load(args.neuralNetworkFilename, map_location=lambda storage, location: storage))

#sigmoidFcn = torch.nn.Sigmoid()
submissionFile = open("submission_" + args.neuralNetworkFilename + '.csv', 'w')
submissionFile.write('image_id,label_id\n')

# Loop through the images
numberOfImages = testImporter.NumberOfSamples()
for imageNdx in range(numberOfImages):
    print ("imageNdx = {}/{}".format( imageNdx + 1, numberOfImages))
    minibatchIndicesList = [ imageNdx ] # Minibatch of a single index
    imageTensor, _ = testImporter.Minibatch(minibatchIndicesList, imageSize)
    if args.cuda:
        imageTensor = imageTensor.cuda()
    imageVariable = torch.autograd.Variable(imageTensor)
    #outputVariable = sigmoidFcn( neuralNet(imageVariable))
    outputVariable = neuralNet(imageVariable)
    #print ("outputVariable =", outputVariable)

    # List the labels: outputVariable.data.shape = torch.Size([1, 228])
    labelsList = []
    for labelNdx in range(testImporter.NumberOfAttributes()):
        if outputVariable.data[0, labelNdx] >= 0.5:
            labelsList.append(labelNdx + 1) # labels are 1-based
    print ("labelsList =", labelsList)
    submissionLine = str(imageNdx + 1) + ','
    for label in labelsList:
        submissionLine += str(label) + ' '
    submissionFile.write(submissionLine + '\n')

submissionFile.close()

