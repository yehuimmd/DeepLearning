# encoding:utf-8
import random
import numpy


class LogisticRegression(object):
    # 给定训练数据（x, y), 学习率（alpha）和正则化权重（lam）
    def __init__(self, X, Y, alpha=0.0005, lam=0.1, printIter=True):
        x = numpy.array(X)
        m, n = x.shape # 获得样本个数和每个样本特征个数

        # 对数据进预处理， 使得数据满足标准正态分布
        self.xMean = numpy.mean(x, axis=0)
        self.xStd = numpy.std(x, axis=0)
        x = (x - self.xMean) / self.xStd

        # 为了便于处理偏置项b，做的特殊处理，即输入数据补常数1
        const = numpy.array([1] * m).reshape(m, 1)
        self.X = numpy.append(const, x, axis=1)

        self.Y = numpy.array(Y)
        self.alpha = alpha
        self.lam = lam
        self.theta = numpy.array([0.0] * (n + 1))

        self.printIter = printIter
        print "lambda=", self.lam

    # sigmoid函数定义
    def _sigmoid(self, x):
        z = 1.0 / (1.0 + numpy.exp((-1) * x))
        return z

    # 定义损失函数
    def _costFunc(self):
        m, n = self.X.shape
        h_theta = self._sigmoid(numpy.dot(self.X, self.theta)) # 模型预测结果

        # 每个样本的模型损失函数
        cost1 = (-1) * self.Y * numpy.log(h_theta)
        cost2 = (1.0 - self.Y) * numpy.log(1.0 - h_theta)

        # 加上参数的正则项
        cost = (
            sum(cost1 - cost2) + 0.5 * self.lam * sum(self.theta[1:] ** 2)) / m
        return cost

    # 通过求导，更新参数
    def _gradientDescend(self, iters):
        """
        X: 输入数据特征
        Y: 输出目标
        theta: 模型参数
        alpha: 学习率
        lam: 正则化权重
       """

        m, n = self.X.shape
        for i in xrange(0, iters):
            theta_temp = self.theta # 参数原始值

            h_theta = self._sigmoid(numpy.dot(self.X, self.theta)) # 模型预测结果
            diff = h_theta - self.Y
            self.theta[0] = theta_temp[0] - self.alpha * \
                (1.0 / m) * sum(diff * self.X[:, 0]) # 偏置项求导及其更新

            for j in xrange(1, n):
                val = theta_temp[
                    j] - self.alpha * (1.0 / m) * (sum(diff * self.X[:, j]) + self.lam * m * theta_temp[j])
                # 参数求导及其更新
                self.theta[j] = val
            cost = self._costFunc() # 计算当前参数值的情况下，损失函数是多少，损失函数逐步下降才说明程序可能正确

            if self.printIter:
                print "Iteration", i, "\tcost=", cost

    def run(self, iters, printIter=True):
        self.printIter = printIter
        self._gradientDescend(iters)

    # 给定输入数据，根据模型（模型定义及其参数值），计算预测结果
    def predict(self, X):
        m, n = X.shape
        x = numpy.array(X)
        # 数据预处理
        x = (x - self.xMean) / self.xStd
        # 补常数1
        const = numpy.array([1] * m).reshape(m, 1)
        X = numpy.append(const, x, axis=1)
        # 计算预测得分
        pred = self._sigmoid(numpy.dot(X, self.theta))
        numpy.putmask(pred, pred >= 0.5, 1.0) # 得分大于等于0.5则是正例
        numpy.putmask(pred, pred < 0.5, 0.0)

        return pred
