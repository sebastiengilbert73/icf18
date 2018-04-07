# loadFromJson.py

import json
import torch

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


    def Minibatch(self, minibatchIndicesList, imageSize):
        imagesTensor = torch.FloatTensor(len(minibatchIndicesList), 3, imageSize[1], imageSize[0])
        targetLabelsTensor = torch.FloatTensor(len(minibatchIndicesList), len(self.labels))

        return imagesTensor, targetLabelsTensor