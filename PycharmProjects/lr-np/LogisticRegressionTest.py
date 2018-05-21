# encoding:utf-8

import csv
import random
import numpy
import LogisticRegression as LR

# 读入数据
data = numpy.genfromtxt('input.csv', delimiter=',')
# response is in the first column
Y = data[:, 0] # 类标签数据
X = data[:, 1:] # 实例特征数据


# 随机打散数据，X和Y同步打散
m = len(Y)
index = range(0, m)
random.shuffle(index)
X = X[index, :]
Y = Y[index]

# 使用10折交叉验证
nfold = 10
foldSize = int(m / nfold)

# 看每次的错误情况，然后求平均
trainErr = [0.0] * nfold
testErr = [0.0] * nfold
allIndex = range(0, m)
for i in range(0, nfold):
    # 每次训练和测试
    testIndex = range((foldSize * i), foldSize * (i + 1))
    trainIndex = list(set(allIndex) - set(testIndex))

    trainX = X[trainIndex, :] #训练数据
    trainY = Y[trainIndex]
    testX = X[testIndex, :] #测试数据
    testY = Y[testIndex]

    # 超参数设置
    alpha = 0.05
    lam = 0.1
    model = LR.LogisticRegression(trainX, trainY, alpha, lam)
    model.run(400, printIter=False) #训练模型

    # 在训练集中测试模型效果
    trainPred = model.predict(trainX)
    trainErr[i] = float(sum(trainPred != trainY)) / len(trainY)

    # 在测试集中测试模型效果
    testPred = model.predict(testX)
    testErr[i] = float(sum(testPred != testY)) / len(testY)

    print "train Err=", trainErr[i], "test Err=", testErr[i]
    print " "

# 总体结果
print "summary:"
print "average train err =", numpy.mean(trainErr) * 100, "%"
print "average test err =", numpy.mean(testErr) * 100, "%"
