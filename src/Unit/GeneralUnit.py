'''
Created on 2018年3月4日

@author: IL MARE
'''
import shelve
import Util.DataUtil as DataUtil
import lib.DecisionTreeLib as DTLib
import lib.RFLib as RFLib
import lib.LogisticLib as LRLib
import csv
import time
import numpy as np

filePath = r"G:\研究生课件\数据挖掘\实验数据"

def loadDataSet(filename):
    print("Loading data...")
    dataSet, labelSet = DataUtil.loadDataForRMOrDTModel(filename)
    print("Loaded data!")
    print("Undersampling data...")
    dataSet, labelSet = DataUtil.underSampling(dataSet, labelSet, "yes", "no")
    print("Undersampled data!")
    return dataSet, labelSet

def serializeDTModel():
    start = time.clock()
    dataSet, labelSet = loadDataSet("bank-additional")
    tmp_lst = []
    maxRatio = 0
    finalModel = {}
    for i in range(100):
        trainSet, trainLabel, testSet, testLabel = DataUtil.generateTrainSet(dataSet, labelSet)
        model = DTLib.createDecisionTree(trainSet, trainLabel)
        errorRatio = DTLib.testDTModel(testSet, testLabel, model)
        tmp_lst.append(1 - errorRatio)
        if (1 - errorRatio) > maxRatio:
            maxRatio = 1 - errorRatio
            finalModel = model
    db = shelve.open("MiningModel")
    db["DTModel"] = finalModel
    db["DTModelCorrectRatio"] = maxRatio
    db.close()

def serializeRFModel():
    start = time.clock()
    dataSet, labelSet = loadDataSet("bank-additional")
    maxRatio = 0
    finalModel = None
    for i in range(10):
        trainSet, trainLabel, testSet, testLabel = DataUtil.generateTrainSet(dataSet, labelSet)
        forest = RFLib.generateRandomForest(trainSet, trainLabel, 20)
        errorCount = 0
        for data, label in zip(testSet, testLabel):
            predict_label = RFLib.predictByRandomForest(forest, data)
            if predict_label != label:
                errorCount += 1
        RFratio = float(errorCount) / len(testLabel)
        if (1 - RFratio) > maxRatio:
            maxRatio = 1 - RFratio
            finalModel = forest
        print("RF:total error ratio is %.3f, correct ratio is %.3f" % (RFratio, 1 - RFratio))
    db = shelve.open("MiningModel")
    db["RFModel"] = finalModel
    db["RFModelCorrectRatio"] = maxRatio
    db.close()

def loaddata_temp(filename):
    try:
        fp = open("{0}/{1}.csv".format(filePath, filename), "r")
        reader = csv.reader(fp)
        trainSet = []
        trainLabel = []
        for line in reader:
            trainSet.append(line[0: -1])
            trainLabel.append(int(line[-1]))
        return trainSet, trainLabel
    except Exception as e:
        print(e)
    finally:
        fp.close()

if __name__ == "__main__":
    start = time.clock()
    dataSet, labelSet = loaddata_temp("bank-addtional-format-lr")
    trainSet, trainLabel, testSet, testLabel = DataUtil.generateTrainSet(dataSet, labelSet)
    weight, logList = LRLib.stocGradDescent(trainSet, trainLabel)
    errorCount = 0
    for data, label in zip(testSet, testLabel):
        predict_label = LRLib.classifyVector(data, weight)
        if predict_label != label:
            errorCount += 1
    ratio = errorCount / len(testLabel)
    print("the error ratio is %.3f, the correct ratio is %.3f -- %.3fs" % (ratio, 1 - ratio, time.clock() - start))
    db = shelve.open("MiningModel")
    db["LRModel"] = weight
    db["LRModelCorrectRatio"] = 1 - ratio
    db.close()