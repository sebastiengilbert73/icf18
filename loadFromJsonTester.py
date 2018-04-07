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
print("importer.imageIdToUrl[2] =", importer.imageIdToUrl[2])
print ("importer.imageIdToLabels[2] =", importer.imageIdToLabels[2])
#print ("importer.labels =", importer.labels)

minibatchIndicesListList = importer.MinibatchIndices(4)
print ("minibatchIndicesListList[0] =", minibatchIndicesListList[0])
imagesTensor, targetLabelsTensor = importer.Minibatch(minibatchIndicesListList[0], (256, 256))
#print (imagesTensor)
print (targetLabelsTensor)