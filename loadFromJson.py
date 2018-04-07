# loadFromJson.py

import json
import torch
import torchvision
import PIL
import requests
import io

class Importer:
    def __init__(self, filepath):
        with open(filepath, 'r') as openFile:
            fileAsString = openFile.read()
        self.jsonObj = json.loads(fileAsString)
        self.imageIds = []

        # Create a dictionary of image_id to url
        self.imageIdToUrl = {}
        imageIdUrlList = self.jsonObj["images"]
        for imageIdUrlDic in imageIdUrlList:
            #print ("loadFromJson.Importer.__init__(): imageIdUrlDic =", imageIdUrlDic)
            image_id = int (imageIdUrlDic["imageId"])
            self.imageIds.append(image_id)
            url = imageIdUrlDic["url"]
            self.imageIdToUrl[image_id] = url

        # Create a dictionary of image_id to labels list
        self.imageIdToLabels = {}
        self.labels = set() # Set of existing labels
        imageIdLabelsList = self.jsonObj["annotations"]
        for imageIdLabelIdsDic in imageIdLabelsList:
            image_id = int( imageIdLabelIdsDic["imageId"])
            labelsStrList = imageIdLabelIdsDic["labelId"]
            labelsList = [ int(labelStr) for labelStr in labelsStrList]
            self.imageIdToLabels[image_id] = labelsList
            self.labels.update(labelsList)


    def Minibatch(self, minibatchIndicesList, imageSize): # imageSize = (width, height)
        imagesTensor = torch.FloatTensor(len(minibatchIndicesList), 3, imageSize[1], imageSize[0]).zero_() # N x C x H x W
        targetLabelsTensor = torch.FloatTensor(len(minibatchIndicesList), len(self.labels)).zero_()
        preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.Resize((imageSize[1], imageSize[0])), # Resize expects (h, w)
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize( (0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
        ] )

        for index in minibatchIndicesList:
            if index < 0 or index >= len(self.imageIdToUrl):
                raise IndexError("loadFromJson.Importer.Minibatch(): Index {} is out of range [0, {}]".format(index, len(self.imageIdToUrl) - 1))
            image_id = index + 1 # image_id is 1-based
            url = self.imageIdToUrl[image_id]
            labels = self.imageIdToLabels[image_id]
            #print ("loadFromJson.Importer.Minibatch(): index {}: url = {} ; labels = {}".format(index, url, labels))
            response = requests.get(url)
            #print ("loadFromJson.Importer.Minibatch(): response =", response)
            pilImg = PIL.Image.open(io.BytesIO(response.content))
            #pilImg.show()
            imgTensor = preprocessing(pilImg)
            imagesTensor[index] = imgTensor

            labelsTensor = torch.FloatTensor(len (self.labels) ).zero_()
            for label in labels:
                labelNdx = label - 1 # labels are 1-based
                labelsTensor[labelNdx] = 1.0
            targetLabelsTensor[index] = labelsTensor

        return imagesTensor, targetLabelsTensor

