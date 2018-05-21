# encoding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import math
import os
import random
import numpy as np
import tensorflow as tf

import input_data



data_dir = "/home/hesz/dataset/fb15k/data/"
data_sets = input_data.read_data_sets(data_dir)

batch_size = 1200
embed_size = 50
margin = 1.0
epoch = 500
alpha = 0.0001

##构建模型
graph = tf.Graph()

with graph.as_default():
    pos_hs = tf.placeholder(tf.int32, shape=[None])
    pos_rs = tf.placeholder(tf.int32, shape=[None])
    pos_ts = tf.placeholder(tf.int32, shape=[None])
    neg_hs = tf.placeholder(tf.int32, shape=[None])
    neg_rs = tf.placeholder(tf.int32, shape=[None])
    neg_ts = tf.placeholder(tf.int32, shape=[None])

#ent_embeddings = tf.Variable(tf.random_uniform([ent_num, embed_size], -1.0, 1.0))
ent_embeddings = tf.Variable(tf.truncated_normal([data_sets.ent_num, embed_size], stddev=1.0 / math.sqrt(embed_size)))
rel_embeddings = tf.Variable(tf.truncated_normal([data_sets.rel_num, embed_size], stddev=1.0 / math.sqrt(embed_size)))
ent_embeddings = tf.nn.l2_normalize(ent_embeddings, 1)
rel_embeddings = tf.nn.l2_normalize(rel_embeddings, 1)

phs = tf.nn.embedding_lookup(ent_embeddings, pos_hs)
prs = tf.nn.embedding_lookup(rel_embeddings, pos_rs)
pts = tf.nn.embedding_lookup(ent_embeddings, pos_ts)
nhs = tf.nn.embedding_lookup(ent_embeddings, neg_hs)
nrs = tf.nn.embedding_lookup(rel_embeddings, neg_rs)
nts = tf.nn.embedding_lookup(ent_embeddings, neg_ts)

pos_loss = tf.reduce_sum(tf.abs(phs + prs - pts), 1)
neg_loss = tf.reduce_sum(tf.abs(nhs + nrs - nts), 1)
base_loss = tf.reduce_sum(tf.nn.relu(pos_loss + tf.constant(margin) - neg_loss))

norm_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(phs, 2), 1))
norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(pts, 2), 1))
norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(nhs, 2), 1))
norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(nts, 2), 1))

norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(prs, 2), 1))
norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(nrs, 2), 1))

# norm_loss = tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(phs, 2), 1) - 1.0))
# norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(pts, 2), 1) - 1.0))
# norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(nhs, 2), 1) - 1.0))
# norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(nts, 2), 1) - 1.0))
#
# norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(prs, 2), 1) - 1.0))
# norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(nrs, 2), 1) - 1.0))

loss_fun = base_loss + alpha * norm_loss
optimizer = tf.train.AdagradOptimizer(0.25).minimize(loss_fun)  ##训练

session = tf.Session()
init = tf.initialize_all_variables()
session.run(init)

average_loss = 0

num_steps = epoch * (data_sets.tri_num // batch_size)
for step in xrange(num_steps):
    batch_pos, batch_neg = data_sets.generate_batch(batch_size)
    feed_dict = {pos_hs: [x[0] for x in batch_pos],
                 pos_rs: [x[1] for x in batch_pos],
                 pos_ts: [x[2] for x in batch_pos],
                 neg_hs: [x[0] for x in batch_neg],
                 neg_rs: [x[1] for x in batch_neg],
                 neg_ts: [x[2] for x in batch_neg]}
    (_, loss_val) = session.run([optimizer, loss_fun], feed_dict=feed_dict)
    average_loss += loss_val

    if step % 100 == 99:
        print "Average loss at step ", step, ": ", average_loss / 100
        average_loss = 0
        feed_dict = {pos_hs: [x[0] for x in data_sets.valid_pos],
                     pos_rs: [x[1] for x in data_sets.valid_pos],
                     pos_ts: [x[2] for x in data_sets.valid_pos],
                     neg_hs: [x[0] for x in data_sets.valid_neg],
                     neg_rs: [x[1] for x in data_sets.valid_neg],
                     neg_ts: [x[2] for x in data_sets.valid_neg]}
        pos_scores = session.run(pos_loss, feed_dict=feed_dict)
        neg_scores = session.run(neg_loss, feed_dict=feed_dict)
        new_hints = map(lambda x: 1 if x[0] < x[1] else 0, zip(pos_scores, neg_scores))
        print "valid accuracy %f" % (sum(new_hints) * 1.0 / len(new_hints))  ##评价

print "最终准确率"
feed_dict = {pos_hs: [x[0] for x in data_sets.test_pos],
             pos_rs: [x[1] for x in data_sets.test_pos],
             pos_ts: [x[2] for x in data_sets.test_pos],
             neg_hs: [x[0] for x in data_sets.test_neg],
             neg_rs: [x[1] for x in data_sets.test_neg],
             neg_ts: [x[2] for x in data_sets.test_neg]}
pos_scores = session.run(pos_loss, feed_dict=feed_dict)
neg_scores = session.run(neg_loss, feed_dict=feed_dict)
new_hints = map(lambda x: 1 if x[0] < x[1] else 0, zip(pos_scores, neg_scores))
print "test accuracy %f" % (sum(new_hints) * 1.0 / len(new_hints))

