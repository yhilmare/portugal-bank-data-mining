'''
Created on 2018年3月11日

@author: IL MARE
'''
import shelve
import os
import sys
sys.path.append(os.getcwd())
import Util.DataUtil as DataUtil
import lib.DecisionTreeLib as DTLib
import lib.RFLib as RFLib
import lib.LogisticLib as LRLib
import lib.SVMLib as SVMLib


def testDTModel(filename="bank-additional"):
    db = shelve.open("{0}/MiningModel".format(sys.path[0]))
    maxCorrectRatio = db["DTModelCorrectRatio"]
    model = db["DTModel"]
    db.close()
    dataSet, labelSet = DataUtil.loadDataForRMOrDTModel(filename)
    error = 0
    for data, label in zip(dataSet, labelSet):
        predict_label = DTLib.predictByDTModel(data, model)
        if predict_label != label:
            error += 1
    errorRatio = error / len(dataSet)
    print("DT:error ratio:%.3f, correct ratio:%.3f, correct ratio on trainSet:%.3f" % (errorRatio, 1 - errorRatio, maxCorrectRatio))
 
def testRFModel(filename="bank-additional"):
    db = shelve.open("{0}/MiningModel".format(sys.path[0]))
    maxCorrectRatio = db["RFModelCorrectRatio"]
    model = db["RFModel"]
    db.close()
    dataSet, labelSet = DataUtil.loadDataForRMOrDTModel(filename)
    error = 0
    for data, label in zip(dataSet, labelSet):
        predict_label = RFLib.predictByRandomForest(model, data)
        if predict_label != label:
            error += 1
    errorRatio = error / len(dataSet)
    print("RF:error ratio:%.3f, correct ratio:%.3f, correct ratio on trainSet:%.3f" % (errorRatio, 1 - errorRatio, maxCorrectRatio))
 
def testLRModel():
    db = shelve.open("{0}/MiningModel".format(sys.path[0]))
    maxCorrectRatio = db["LRModelCorrectRatio"]
    weight = db["LRModel"]
    db.close()
    dataSet, labelSet = DataUtil.loadTempDataForSVMOrLRModel("bank-addtional-format-lr")
#     dataSet, labelSet = DataUtil.loadDataForSVMOrLRModel("bank-additional", "lr")
    error = 0
    for data, label in zip(dataSet, labelSet):
        predict_label = LRLib.classifyVector(data, weight)
        if predict_label != label:
            error += 1
    errorRatio = error / len(dataSet)
    print("LR:error ratio:%.3f, correct ratio:%.3f, correct ratio on trainSet:%.3f" % (errorRatio, 1 - errorRatio, maxCorrectRatio))
 
def testSVMModel():
    db = shelve.open("{0}/MiningModel".format(sys.path[0]))
    maxCorrectRatio = db["SVMModelCorrectRatio"]
    model = db["SVMModel"]
    db.close()
    dataSet, labelSet = DataUtil.loadTempDataForSVMOrLRModel("bank-addtional-format-svm")
#     dataSet, labelSet = DataUtil.loadDataForSVMOrLRModel("bank-additional", "svm")
    error = 0
    for data, label in zip(dataSet, labelSet):
        predict_label = SVMLib.predictLabel(data, *model)
        if predict_label != label:
            error += 1
    errorRatio = error / len(dataSet)
    print("SVM:error ratio:%.3f, correct ratio:%.3f, correct ratio on trainSet:%.3f" % (errorRatio, 1 - errorRatio, maxCorrectRatio))

if __name__ == "__main__":
    testDTModel()
    testRFModel()
    testLRModel()
    testSVMModel()