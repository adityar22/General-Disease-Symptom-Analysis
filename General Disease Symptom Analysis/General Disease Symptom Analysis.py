###Import Library
from math import log
from collections import defaultdict
import json
import pprint
import csv

###Create Dataset from read CSV File
filename = "Testing.csv"
def createDataSet():
  features = []
  dataset = []
  with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    features = next(csvreader)
    
    for row in csvreader:
      dataset.append(row)
  return dataset, features

###Calculate entropy to find Information Gain
def calcEntropy(dataset):
  counter = defaultdict(int)
  for record in dataset:
    label = record[-1]
    counter[label] += 1
  entropy = 0.0
  for key in counter:
    probability = counter[key]/len(dataset)
    entropy -= probability * log(probability, 2)
  return entropy

###Split Dataset to attribut tabel to calculate entropy
def splitDataset(dataSet, axis, value):
  retDataSet = []
  for featVec in dataSet:
    if featVec[axis] == value:
      reducedFeatVec = featVec[:axis]
      reducedFeatVec.extend(featVec[axis+1:])
      retDataSet.append(reducedFeatVec)
  return retDataSet

###Find best node by find largest information gain
def chooseBestFeatureToSplit(dataset):
  baseEntropy = calcEntropy(dataset)
  bestInfoGain = 0.0; bestFeature = -1

  numFeat = len(dataset[0]) - 1
  for indx in range(numFeat):
    featValues = {record[indx] for record in dataset}
    featEntropy = 0.0
    for value in featValues:
      subDataset = splitDataset(dataset, indx, value)
      probability = len(subDataset)/float(len(dataset))
      featEntropy += probability * calcEntropy(subDataset)
    
    infoGain = baseEntropy - featEntropy
    if infoGain > bestInfoGain:
      bestInfoGain = infoGain
      bestFeature = indx
  return bestFeature

###Create Decision Tree
def createTree(dataset, features):
  labels = [record[-1] for record in dataset]
  
  #print(labels.count(labels[0]))
  #print(len(labels))

  if labels.count(labels[0]) == len(labels):
    return labels[0]
  if len(dataset[0]) == 1:
    mjcount = max(labels, key=labels.count)
    return (mjcount)
  
  bestFeat = chooseBestFeatureToSplit(dataset)
  bestFeatLabel = features[bestFeat]
  featValues = {record[bestFeat] for record in dataset}
  subLabels = features[:]
  del(subLabels[bestFeat])

  myTree = {bestFeatLabel:{}}
  for value in featValues:
    subDataset = splitDataset(dataset, bestFeat, value)
    subTree = createTree(subDataset, subLabels)
    myTree[bestFeatLabel].update({value: subTree})
  return myTree

###Find predict result from sample data
def predict(inputTree, features, testVec):
  def classify(inputTree, testDict):
    (key, subtree), = inputTree.items()
    testValue = testDict.pop(key)
    print(key)
    print(testValue)
    #print(testDict)
    #print(len(subtree[testValue]))
    if len(subtree[testValue]) != 1:
      print(subtree[testValue])
      return subtree[testValue]
    else:
      print(subtree[testValue])
      return classify(subtree[testValue], testDict)
  
  testDict = dict(zip(features, testVec))
  return classify(inputTree, testDict)

###Print Decision Tree
def pprintTree(tree):
  #pprint.pprint (tree)
  tree_str = json.dumps(tree, indent=4)
  tree_str = tree_str.replace("\n    ", "\n")
  tree_str = tree_str.replace('"', "")
  tree_str = tree_str.replace(',', "")
  tree_str = tree_str.replace("{", "")
  tree_str = tree_str.replace("}", "")
  tree_str = tree_str.replace("    ", " | ")
  tree_str = tree_str.replace("  ", " ")    
  print (tree_str)

### Main Function
def main(testData):
  dataset, features = createDataSet()
  tree = createTree(dataset, features)
  pprintTree(tree)
  
  for vec in testData:
    pred = predict(tree, features, vec)
    print("\n From sample data given, its predicted that the disease will be ",pred)

### Input Sample Data
testData = [('1', '0', '1', '0', '1', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '1', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0', '0',)]
main(testData)
