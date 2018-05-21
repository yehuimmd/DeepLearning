# encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import random, time
import os, cPickle
from data_utils import *
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_path = 'review_polarity/total.txt'
dump_path = 'review_polarity/data.pkl'
if not os.path.exists(dump_path):
    dataset = DataLoader(data_path)
    cPickle.dump(dataset, open(dump_path, "wb"))
else:
    dataset = cPickle.load(open(dump_path, "rb"))

instances = dataset.dataset
random.shuffle(instances)

m, n = dataset.vocab_size, len(instances)
train_set, test_test = instances[:-1*int(0.3*n)], instances[-1 * int(0.3*n):]
train_X, train_Y = np.asarray([x[0] for x in train_set]), np.asarray([x[1] for x in train_set])
test_X, test_Y = np.asarray([x[0] for x in test_test]), np.asarray([x[1] for x in test_test])

with tf.Graph().as_default():
    x_data = tf.placeholder(tf.float32, [None, m])
    y_data = tf.placeholder(tf.float32, [None, 2])

    W = tf.Variable(tf.truncated_normal([m, 2], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[2]), name="b")

    y_pred = tf.nn.softmax(tf.matmul(x_data, W) + b)

    cost = tf.reduce_mean(-tf.reduce_sum(y_data * tf.log(y_pred),
                                         reduction_indices=1))
    loss = cost + 0.025 * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b))

    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        for epoch in xrange(1, 100):
            epoch_start = time.time()
            loss_sum = 0.0
            for start, end in zip(range(0, len(train_X), 128), range(128, len(train_X) + 1, 128)):
                _, curr_loss = session.run([optimizer, loss],
                                           feed_dict={x_data: train_X[start:end],
                                                      y_data: train_Y[start:end]})
                loss_sum += curr_loss
            epoch_finish = time.time()
            print "第 %d 轮 loss %.5f time %.2f" % (epoch, loss_sum, epoch_finish - epoch_start)

            # 模型测试
            correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_data, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print "第", epoch, "轮学习, 准确率为: ", accuracy.eval({x_data: test_X, y_data: test_Y})



