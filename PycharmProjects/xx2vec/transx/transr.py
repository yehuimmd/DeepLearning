# encoding:utf-8

import math
import numpy as np
import tensorflow as tf

input = "/home/hesz/dataset/fb15k/data/"
output = "/home/hesz/dataset/fb15k/params3/"
batch_size = 1200
emb_size = 50
margin = 1.0
epoch = 500
alpha = 0.0625
print "input: ", input
print "output: ", output

import input_data

dataset = input_data.read_data_sets(input)

ent_num = dataset.ent_num
rel_num = dataset.rel_num


def init_embedding(_emb_num, _emb_size):
    embs = tf.Variable(tf.truncated_normal([_emb_num, _emb_size],
                                           stddev=1.0 / math.sqrt(_emb_size)))
    return tf.nn.l2_normalize(embs, 1)

pos_hs = tf.placeholder(tf.int32, shape=[None])
pos_rs = tf.placeholder(tf.int32, shape=[None])
pos_ts = tf.placeholder(tf.int32, shape=[None])
neg_hs = tf.placeholder(tf.int32, shape=[None])
neg_rs = tf.placeholder(tf.int32, shape=[None])
neg_ts = tf.placeholder(tf.int32, shape=[None])

# 如何使用读入进来的向量呢
ent_embeddings = tf.Variable(np.load(output + 'ent_vecs.npy'))
rel_embeddings = tf.Variable(np.load(output + 'rel_vecs.npy'))
# rel_matrixs = tf.Variable(np.load(data_dir + 'rel_mats.npy'))
# ent_embeddings = init_embedding(ent_num, ent_emb_size)
# rel_embeddings = init_embedding(rel_num, ent_emb_size)
rel_matrixs = tf.Variable([np.eye(emb_size).tolist()] * rel_num)  # 关系矩阵最好初始化为单位矩阵

phs = tf.nn.embedding_lookup(ent_embeddings, pos_hs)
prs = tf.nn.embedding_lookup(rel_embeddings, pos_rs)
pts = tf.nn.embedding_lookup(ent_embeddings, pos_ts)
nhs = tf.nn.embedding_lookup(ent_embeddings, neg_hs)
nrs = tf.nn.embedding_lookup(rel_embeddings, neg_rs)
nts = tf.nn.embedding_lookup(ent_embeddings, neg_ts)

pmat = tf.nn.embedding_lookup(rel_matrixs, pos_rs)
nmat = tf.nn.embedding_lookup(rel_matrixs, neg_rs)

phs2 = tf.reshape(phs, [-1, 1, emb_size])
pts2 = tf.reshape(pts, [-1, 1, emb_size])
nhs2 = tf.reshape(nhs, [-1, 1, emb_size])
nts2 = tf.reshape(nts, [-1, 1, emb_size])

phs_new = tf.batch_matmul(phs2, pmat)
pts_new = tf.batch_matmul(pts2, pmat)
nhs_new = tf.batch_matmul(nhs2, nmat)
nts_new = tf.batch_matmul(nts2, nmat)

phs_new = tf.reshape(phs_new, [-1, emb_size])
pts_new = tf.reshape(pts_new, [-1, emb_size])
nhs_new = tf.reshape(nhs_new, [-1, emb_size])
nts_new = tf.reshape(nts_new, [-1, emb_size])

pos_loss = tf.reduce_sum(tf.abs(phs_new + prs - pts_new), 1)
neg_loss = tf.reduce_sum(tf.abs(nhs_new + nrs - nts_new), 1)

base_loss = tf.reduce_sum(tf.nn.relu(pos_loss + margin - neg_loss))

norm_loss = tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(phs_new, 2), 1) - 1.0))
norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(pts_new, 2), 1) - 1.0))
norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(nhs_new, 2), 1) - 1.0))
norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(nts_new, 2), 1) - 1.0))
norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(phs, 2), 1) - 1))
norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(pts, 2), 1) - 1))
norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(nhs, 2), 1) - 1))
norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(nts, 2), 1) - 1))
norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(prs, 2), 1) - 1))
norm_loss += tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.pow(nrs, 2), 1) - 1))

# norm_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(phs_new, 2), 1))
# norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(pts_new, 2), 1))
# norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(nhs_new, 2), 1))
# norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(nts_new, 2), 1))
# norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(phs, 2), 1))
# norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(pts, 2), 1))
# norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(nhs, 2), 1))
# norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(nts, 2), 1))
# norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(prs, 2), 1))
# norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(nrs, 2), 1))


loss_fun = base_loss + tf.constant(alpha) * norm_loss
# optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss_fun)  ##训练
optimizer = tf.train.AdagradOptimizer(0.25).minimize(loss_fun)

# ent_embeddings = tf.nn.l2_normalize(ent_embeddings, 1)
# rel_embeddings = tf.nn.l2_normalize(rel_embeddings, 1)

session = tf.Session()
init = tf.initialize_all_variables()
session.run(init)

batch_num = dataset.tri_num // batch_size
average_loss = 0
for step in xrange(epoch):
    for tmp_ind in xrange(batch_num):
        batch_pos, batch_neg = dataset.generate_batch(batch_size)
        feed_dict = {
            pos_hs: [x[0] for x in batch_pos],
            pos_rs: [x[1] for x in batch_pos],
            pos_ts: [x[2] for x in batch_pos],
            neg_hs: [x[0] for x in batch_neg],
            neg_rs: [x[1] for x in batch_neg],
            neg_ts: [x[2] for x in batch_neg]}
        (_, loss_val) = session.run([optimizer, loss_fun], feed_dict=feed_dict)
        average_loss += loss_val

    print "Average loss at step ", step, ": ", average_loss / batch_num
    average_loss = 0
    feed_dict = {
        pos_hs: [x[0] for x in dataset.valid_pos],
        pos_rs: [x[1] for x in dataset.valid_pos],
        pos_ts: [x[2] for x in dataset.valid_pos],
        neg_hs: [x[0] for x in dataset.valid_neg],
        neg_rs: [x[1] for x in dataset.valid_neg],
        neg_ts: [x[2] for x in dataset.valid_neg]}
    pos_scores = session.run(pos_loss, feed_dict=feed_dict)
    neg_scores = session.run(neg_loss, feed_dict=feed_dict)
    new_hints = map(lambda x: 1 if x[0] < x[1] else 0, zip(pos_scores, neg_scores))
    print "valid accuracy %f" % (sum(new_hints) * 1.0 / len(new_hints))



np.save(output + 'ent_vecs', session.run(ent_embeddings))
np.save(output + 'rel_vecs', session.run(rel_embeddings))
np.save(output + 'rel_mats', session.run(rel_matrixs))
