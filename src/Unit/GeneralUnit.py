'''
Created on 2018年3月4日

@author: IL MARE
'''
import shelve
import Util.DataUtil as DataUtil
import lib.DecisionTreeLib as DTLib
import lib.RFLib as RFLib
import lib.LogisticLib as LRLib
import lib.SVMLib as SVMLib
import sys

def loadDataSet(filename):
    print("Loading data...")
    dataSet, labelSet = DataUtil.loadDataForRMOrDTModel(filename)
    print("Loaded data!")
    print("Undersampling data...")
    dataSet, labelSet = DataUtil.underSampling(dataSet, labelSet, "yes", "no")
    print("Undersampled data!")
    return dataSet, labelSet

def serializeDTModel():
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
    db = shelve.open("{0}/MiningModel".format(sys.path[0]))
    db["DTModel"] = finalModel
    db["DTModelCorrectRatio"] = maxRatio
    db.close()

def serializeRFModel():
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
    db = shelve.open("{0}/MiningModel".format(sys.path[0]))
    db["RFModel"] = finalModel
    db["RFModelCorrectRatio"] = maxRatio
    db.close()

def serializeLRModel():
    dataSet, labelSet = DataUtil.loadTempDataForSVMOrLRModel("bank-addtional-format-lr")
    trainSet, trainLabel, testSet, testLabel = DataUtil.generateTrainSet(dataSet, labelSet)
    weight, logList = LRLib.stocGradDescent(trainSet, trainLabel)
    errorCount = 0
    for data, label in zip(testSet, testLabel):
        predict_label = LRLib.classifyVector(data, weight)
        if predict_label != label:
            errorCount += 1
    ratio = errorCount / len(testLabel)
    print("the error ratio is %.3f, the correct ratio is %.3f" % (ratio, 1 - ratio))
    db = shelve.open("{0}/MiningModel".format(sys.path[0]))
    db["LRModel"] = weight
    db["LRModelCorrectRatio"] = 1 - ratio
    db.close()

def serializeSVMModel():
    dataSet, labelSet = DataUtil.loadTempDataForSVMOrLRModel("bank-addtional-format-svm")
    dataSet, labelSet = DataUtil.underSampling(dataSet, labelSet, 1, -1)
    trainSet, trainLabel, testSet, testLabel = DataUtil.generateTrainSet(dataSet, labelSet)
    kTup = ("lin", 1.3)
    alphas, b = SVMLib.realSMO(trainSet, trainLabel, 0.6, 0.01, kTup, 10)
    errorCount = 0
    sv, svl = SVMLib.getSupportVectorandSupportLabel(trainSet, trainLabel, alphas)
    for data, label in zip(testSet, testLabel):
        predict_label = SVMLib.predictLabel(data, *[sv, svl, alphas, b, kTup])
        if predict_label != label:
            errorCount += 1
    ratio = errorCount / len(testLabel)
    print("the error ratio is %.3f, the correct ratio is %.3f" % (ratio, 1 - ratio))
    db = shelve.open("{0}/MiningModel".format(sys.path[0]))
    db['SVMModel'] = [sv, svl, alphas, b, kTup]
    db['SVMModelCorrectRatio'] = 1 - ratio
    db.close()

if __name__ == "__main__":
    pass