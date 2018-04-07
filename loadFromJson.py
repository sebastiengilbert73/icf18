# loadFromJson.py

import json


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
        imageIdLabelsList = self.jsonObj["annotations"]
        for imageIdLabelIdsDic in imageIdLabelsList:
            image_id = int( imageIdLabelIdsDic["imageId"])
            labelsStrList = imageIdLabelIdsDic["labelId"]
            labelsList = [ int(labelStr) for labelStr in labelsStrList]
            self.imageIdToLabels[image_id] = labelsList
