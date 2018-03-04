'''
Created At 2018-2-18
IL MARE
'''
import time
import csv
from lib import SVMLib as SVMLib
from Util import DataUtil as DataUtil

filePath = r"/Users/yh_swjtu/Desktop/数据挖掘/bank-additional"

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
    # dataSet, labelSet = DataUtil.loadDataForSVMOrLRModel("bank-additional", "svm")#正统方法
    dataSet, labelSet = loaddata_temp("bank-addtional-format-svm")
    dataSet, labelSet = DataUtil.underSampling(dataSet, labelSet, 1, -1)
    trainSet, trainLabel, testSet, testLabel = DataUtil.generateTrainSet(dataSet, labelSet)
    kTup = ("lin", 0.1)
    alphas, b = SVMLib.realSMO(trainSet, trainLabel, 0.6, 0.01, kTup, 10)
    errorCount = 0
    for data, label in zip(testSet, testLabel):
        predict_label = SVMLib.predictLabel(trainSet, trainLabel, alphas, b, data, kTup)
        if predict_label != label:
            errorCount += 1
    ratio = errorCount / len(testLabel)
    print("the error ratio is %.3f, the correct ratio is %.3f -- %.3fs" % (ratio, 1 - ratio, time.clock() - start))