# loadFromJsonTester.py
import json
import loadFromJson

print ("loadFromJsonTester.py")
importer = loadFromJson.Importer('/home/sebastien/MachineLearning/BenchmarkDatasets/imaterialist-challenge-fashion-2018/train.json',
                                 maximumNumberOfTrainingImages=0)

attributesFrequencies = importer.AttributesFrequencies()
print ("attributesFrequencies =", attributesFrequencies)
print (len(attributesFrequencies))