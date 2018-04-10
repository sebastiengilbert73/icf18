# loadFromJson.py

import json
import torch
import torchvision
import PIL
import requests
import io
import numpy

class Importer:
    def __init__(self, filepath, maximumNumberOfTrainingImages, numberOfAttributes=0):
        with open(filepath, 'r') as openFile:
            fileAsString = openFile.read()
        jsonObj = json.loads(fileAsString)
        self.imageIds = []
        self.imageIdToUrl = {}
        self.imageIdToLabels = {}


        # Create a dictionary of image_id to url

        imageIdUrlList = jsonObj["images"]
        for imageIdUrlDic in imageIdUrlList:
            #print ("loadFromJson.Importer.__init__(): imageIdUrlDic =", imageIdUrlDic)
            image_id = int (imageIdUrlDic["imageId"])
            self.imageIds.append(image_id)
            url = imageIdUrlDic["url"]
            self.imageIdToUrl[image_id] = url

        # Create a dictionary of image_id to labels list
        self.labels = set() # Set of existing labels
        if numberOfAttributes > 0:
            for number in range(1, numberOfAttributes + 1):
                self.labels.add(number)

        imageIdLabelsList = jsonObj["annotations"]
        for imageIdLabelIdsDic in imageIdLabelsList:
            image_id = int( imageIdLabelIdsDic["imageId"])
            labelsStrList = imageIdLabelIdsDic["labelId"]
            labelsList = [ int(labelStr) for labelStr in labelsStrList]
            self.imageIdToLabels[image_id] = labelsList
            self.labels.update(labelsList)

        if maximumNumberOfTrainingImages > 0 and maximumNumberOfTrainingImages < len(self.imageIds):
            for imageIdNdx in range(maximumNumberOfTrainingImages, len(self.imageIds)):
                imageIdToRemove = self.imageIds[imageIdNdx]
                del self.imageIdToUrl[imageIdToRemove]
                del self.imageIdToLabels[imageIdToRemove]
            self.imageIds = self.imageIds[0: maximumNumberOfTrainingImages]



    def Minibatch(self, minibatchIndicesList, imageSize): # imageSize = (width, height)
        imagesTensor = torch.FloatTensor(len(minibatchIndicesList), 3, imageSize[1], imageSize[0]).zero_() # N x C x H x W
        targetLabelsTensor = torch.LongTensor(len(minibatchIndicesList), len(self.labels)).zero_()
        preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.Resize((imageSize[1], imageSize[0])), # Resize expects (h, w)
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize( (0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
        ] )

        minibatchIndex0 = 0
        for index in minibatchIndicesList:
            if index < 0 or index >= len(self.imageIdToUrl):
                raise IndexError("loadFromJson.Importer.Minibatch(): Index {} is out of range [0, {}]".format(index, len(self.imageIdToUrl) - 1))
            image_id = index + 1 # image_id is 1-based
            url = self.imageIdToUrl[image_id]
            labels = self.imageIdToLabels[image_id]
            urlResponded = True
            #print ("loadFromJson.Importer.Minibatch(): index {}: url = {} ; labels = {}".format(index, url, labels))
            try:
                response = requests.get(url)
            except:
                urlResponded = False

            if urlResponded:
                pilImg = PIL.Image.open(io.BytesIO(response.content))
                #pilImg.show()
                imgTensor = preprocessing(pilImg)
                imagesTensor[minibatchIndex0] = imgTensor

                labelsTensor = torch.LongTensor(len (self.labels) ).zero_()
                for label in labels:
                    labelNdx = label - 1 # labels are 1-based
                    labelsTensor[labelNdx] = 1
                targetLabelsTensor[minibatchIndex0] = labelsTensor
            minibatchIndex0 += 1

        return imagesTensor, targetLabelsTensor

    def MinibatchIndices(self, minibatchSize):
        numberOfSamples = len(self.imageIdToUrl)
        shuffledList = numpy.arange(numberOfSamples)
        numpy.random.shuffle(shuffledList)
        minibatchesIndicesList = []
        numberOfWholeLists = int(numberOfSamples / minibatchSize)
        for wholeListNdx in range(numberOfWholeLists):
            minibatchIndices = shuffledList[wholeListNdx * minibatchSize: (wholeListNdx + 1) * minibatchSize]
            minibatchesIndicesList.append(minibatchIndices)
        # Add the last incomplete minibatch
        if numberOfWholeLists * minibatchSize < numberOfSamples:
            lastMinibatchIndices = shuffledList[numberOfWholeLists * minibatchSize:]
            minibatchesIndicesList.append(lastMinibatchIndices)
        return minibatchesIndicesList

    def NumberOfSamples(self):
        return len(self.imageIdToUrl)

    def NumberOfAttributes(self):
        return len(self.labels)

    def AttributesFrequencies(self):
        numberOfAttributes = self.NumberOfAttributes()
        attributesOccurrences = [0] * numberOfAttributes
        for imageId in self.imageIdToLabels:
            labels = self.imageIdToLabels[imageId]
            for labelNdx in range(len(labels)):
                label = labels[labelNdx] - 1 # Labels are 1-based
                attributesOccurrences[label] = attributesOccurrences[label] + 1
        numberOfSamples = self.NumberOfSamples()
        return [x/numberOfSamples for x in attributesOccurrences]