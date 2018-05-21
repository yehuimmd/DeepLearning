# encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import tensorflow as tf

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# 数据准备
import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

# 超参数设置
learning_rate = 0.001
training_iters = 200000
batch_size = 64
display_step = 20

# 定义网络参数
n_input = 784 # 输入的维度
n_classes = 10 # 标签的维度

# 输入数据：占位符
input_x = tf.placeholder(tf.float32, [None, n_input])
input_y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32)

# 卷积操作
def conv2d(name, l_input, w, b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(l_input, w, strides=[1, 1, 1, 1], padding='VALID'), b), name=name)

# 最大下采样操作
def max_pool(name, l_input, k):
    return tf.nn.max_pool(l_input, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID', name=name)

# 向量转为矩阵
x = tf.reshape(input_x, shape=[-1, 28, 28, 1])

l2_loss = 0
# 第一层
conv1_w = tf.Variable(tf.random_normal([5, 5, 1, 6]))
conv1_b = tf.Variable(tf.random_normal([6]))
l2_loss += tf.nn.l2_loss(conv1_w)
l2_loss += tf.nn.l2_loss(conv1_b)
conv1 = conv2d('conv1', x, conv1_w, conv1_b) # [batch, 24, 24, 6]
pool1 = max_pool('pool1', conv1, k=2) # [batch, 12, 12, 6]
norm1 = tf.nn.dropout(pool1, keep_prob)

# 第二层
# conv2_w = tf.Variable(tf.random_normal([5, 5, 6, 16]))
# conv2_b = tf.Variable(tf.random_normal([16]))
# l2_loss += tf.nn.l2_loss(conv2_w)
# l2_loss += tf.nn.l2_loss(conv2_b)
# conv2 = conv2d('conv2', norm1, conv2_w, conv2_b) # [batch, 8, 8, 16]
# pool2 = max_pool('pool2', conv2, k=2)# [batch, 4, 4, 16]
# norm2 = tf.nn.dropout(pool2, keep_prob)
# print "norm2 shape", norm2.get_shape()

# 全连接层，先把特征图转为向量
dense1 = tf.reshape(norm1, [-1, 12*12*6])
# dense1 = tf.reshape(norm2, [-1, 4*4*16])

mlp1_w = tf.Variable(tf.random_normal([12*12*6, 120]))
# mlp1_w = tf.Variable(tf.random_normal([4*4*16, 120]))
mlp1_b = tf.Variable(tf.random_normal([120]))
l2_loss += tf.nn.l2_loss(mlp1_w)
l2_loss += tf.nn.l2_loss(mlp1_b)
dense1 = tf.nn.relu(tf.matmul(dense1, mlp1_w) + mlp1_b, name='fc1')

out_w = tf.Variable(tf.random_normal([120, n_classes]))
out_b = tf.Variable(tf.random_normal([n_classes]))
l2_loss += tf.nn.l2_loss(out_w)
l2_loss += tf.nn.l2_loss(out_b)
pred_out = tf.matmul(dense1, out_w) + out_b

# 定义损失函数和学习步骤
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_out, labels=input_y))
cost += 0.1 * l2_loss
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# 测试网络
correct_pred = tf.equal(tf.argmax(pred_out, 1), tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# 模型执行过程-训练过程
with tf.Session() as sess:
    # 初始化所有的共享变量
    init = tf.initialize_all_variables()
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 获取批数据
        sess.run(optimizer, feed_dict={input_x: batch_xs, input_y: batch_ys, keep_prob: 0.5})
        if step % display_step == 0:
            # 计算精度
            acc, loss = sess.run([accuracy, cost], feed_dict={input_x: batch_xs, input_y: batch_ys, keep_prob: 1.})
            print "Iter " + str(step*batch_size) + \
                  ", Minibatch Loss= " + "{:.6f}".format(loss) + \
                  ", Training Accuracy= " + "{:.5f}".format(acc)
        step += 1
    print "Optimization Finished!"
    # 计算测试精度
    print "Testing Accuracy:", sess.run(accuracy, feed_dict={input_x: mnist.test.images[:256],
                                                             input_y: mnist.test.labels[:256],
                                                             keep_prob: 1.})
