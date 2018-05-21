#encoding:utf-8

import tensorflow as tf
import numpy as np
import input_data


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w):
    return tf.matmul(X, w) #定义计算过程


mnist = input_data.read_data_sets("/home/dl/codes/TF101/MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

X = tf.placeholder("float", [None, 784]) #每个图像为784的向量表示
Y = tf.placeholder("float", [None, 10]) #one-hot表示的类别

w = init_weights([784, 10]) # 初始权重

py_x = model(X, w)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y)) # 计算预测与真实的交叉熵
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

predict_op = tf.argmax(py_x, 1) #预测


with tf.Session() as sess:
    tf.global_variables_initializer().run()

    for i in range(100):
        for start, end in zip(range(0, len(trX), 128), range(128, len(trX)+1, 128)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        predict_results = sess.run(predict_op, feed_dict={X: teX})
        accuracy = np.mean(np.argmax(teY, axis=1) == predict_results)
        print "第", i+1, "轮学习, 准确率为: ", accuracy