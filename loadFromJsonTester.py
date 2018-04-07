# loadFromJsonTester.py
import json
import loadFromJson

print ("loadFromJsonTester.py")
importer = loadFromJson.Importer('/home/sebastien/MachineLearning/BenchmarkDatasets/imaterialist-challenge-fashion-2018/train.json')
dataJson = importer.jsonObj

#print ('dataJson =', dataJson)
#print ("dataJson['annotations'] =", dataJson['annotations'])
print("importer.imageIdToUrl[1] =", importer.imageIdToUrl[1])
print ("importer.imageIdToLabels[1] =", importer.imageIdToLabels[1])
print ("importer.labels =", importer.labels)

minibatchIndicesList = [0]
imagesTensor, targetLabelsTensor = importer.Minibatch(minibatchIndicesList, (256, 256))
print (imagesTensor)