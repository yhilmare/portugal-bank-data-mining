'''
Created on 2018年3月11日

@author: IL MARE
'''
import shelve
import Util.DataUtil as DataUtil
import lib.DecisionTreeLib as DTLib
import lib.RFLib as RFLib

def testDTModel(filename="bank-additional"):
    db = shelve.open("MiningModel")
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
    print("DT: error ratio is %.3f, correct ratio is %.3f" % (errorRatio, 1 - errorRatio))

def testRFModel(filename="bank-additional"):
    db = shelve.open("MiningModel")
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
    print("RF: error ratio is %.3f, correct ratio is %.3f" % (errorRatio, 1 - errorRatio))

if __name__ == "__main__":
    testDTModel()
    testRFModel()