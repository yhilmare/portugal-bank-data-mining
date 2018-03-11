'''
Created on 2018年3月11日

@author: IL MARE
'''
import shelve
import Util.DataUtil as DataUtil
import lib.DecisionTreeLib as DTLib
import lib.RFLib as RFLib
import lib.LogisticLib as LRLib
import csv

filePath = r"G:\研究生课件\数据挖掘\实验数据"

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

def testLRModel():
    db = shelve.open("MiningModel")
    maxCorrectRatio = db["LRModelCorrectRatio"]
    weight = db["LRModel"]
    db.close()
    dataSet, labelSet = loaddata_temp("bank-addtional-format-lr")
    error = 0
    for data, label in zip(dataSet, labelSet):
        predict_label = LRLib.classifyVector(data, weight)
        if predict_label != label:
            error += 1
    errorRatio = error / len(dataSet)
    print("LR: error ratio is %.3f, correct ratio is %.3f" % (errorRatio, 1 - errorRatio))

if __name__ == "__main__":
    testDTModel()
    testLRModel()
    testRFModel()