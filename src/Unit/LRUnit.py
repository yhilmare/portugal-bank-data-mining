'''
Created on 2018年3月4日

@author: IL MARE
'''
import Util.DataUtil as DataUtil
from lib import LogisticLib as LRLib
import time
import csv

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
    # dataSet, labelSet = DataUtil.loadDataForSVMOrLRModel("bank-additional")#正统方法
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
    LRLib.plotWeightFig(logList, [i for i in range(0, 6)])