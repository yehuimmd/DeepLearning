#encoding:utf-8

import math
import numpy as np
import tensorflow as tf

batch_size = 120
vocabulary_size = 50000
embedding_size = 100
num_sampled = 20


graph = tf.Graph()
with graph.as_default():
  #输入变量的定义
  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

  #模型参数的定义，即需要学习的参数
  embeddings = tf.Variable(tf.random_uniform(
          [vocabulary_size, embedding_size], -1.0, 1.0))
  nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size,
          embedding_size],stddev=1.0/math.sqrt(embedding_size)))
  nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  #计算过程的定义
  embed = tf.nn.embedding_lookup(embeddings, train_inputs)
  loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases,
         embed, train_labels, num_sampled, vocabulary_size))
  #优化过程的定义
  optimizer=tf.train.GradientDescentOptimizer(1.0).minimize(loss)

num_steps = 4001
with tf.Session(graph=graph) as session:
  #初始化变量
  tf.initialize_all_variables().run()
  for step in xrange(num_steps):
    #获得训练数据
    batch_inputs, batch_labels = generate_batch(batch_size)
    feed_dict={train_inputs:batch_inputs, train_labels:batch_labels}
    #执行学习过程
    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
  #最终学习结果
  final_embeddings = embeddings.eval()


input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
targets = tf.placeholder(tf.int64, [batch_size, 1])

with tf.variable_scope('nnlm' + 'embedding'):
    embeddings = tf.Variable(tf.random_uniform([vocab_size, word_dim], -1.0, 1.0))
with tf.variable_scope('nnlm' + 'weight'):
    weight_h = tf.Variable(tf.truncated_normal([seq_length * word_dim + 1, hidden_num],
                    stddev=1.0 / math.sqrt(hidden_num)))
    softmax_w = tf.Variable(tf.truncated_normal([seq_length * word_dim, vocab_size],
                    stddev=1.0 / math.sqrt(seq_length * word_dim)))
    softmax_u = tf.Variable(tf.truncated_normal([hidden_num + 1, vocab_size],
                    stddev=1.0 / math.sqrt(hidden_num)))


inputs_emb = tf.nn.embedding_lookup(embeddings, input_data)
inputs_emb = tf.reshape(inputs_emb, [batch_size, seq_length * word_dim])


inputs_emb_add = tf.concat(1, [inputs_emb, tf.ones([batch_size, 1])])
inputs = tf.tanh(tf.matmul(inputs_emb_add, weight_h))


inputs_add = tf.concat(1, [inputs, tf.ones([batch_size, 1])])
outputs = tf.matmul(inputs_add, softmax_u) + tf.matmul(inputs_emb, softmax_w)
outputs = tf.nn.softmax(outputs)


one_hot_targets = tf.one_hot(tf.squeeze(targets),
                             vocab_size, 1.0, 0.0)
cost = -tf.reduce_mean(
        tf.reduce_sum(tf.log(outputs) * one_hot_targets, 1))



input_data = tf.placeholder(tf.int32, [batch_size, seq_length])
targets = tf.placeholder(tf.int32, [batch_size, seq_length])

with tf.variable_scope('rnnlm'):
    softmax_w = tf.get_variable("softmax_w", [hidden_num, vocab_size])
    softmax_b = tf.get_variable("softmax_b", [vocab_size])
    embedding = tf.get_variable("embedding", [vocab_size, hidden_num])

cell = rnn_cell.GRUCell(hidden_num)
def loop(prev, _):
    prev = tf.matmul(prev, softmax_w) + softmax_b
    prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
    return tf.nn.embedding_lookup(embedding, prev_symbol)

inputs = tf.split(1, seq_length,
         tf.nn.embedding_lookup(embedding, input_data))
inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
initial_state = cell.zero_state(batch_size, tf.float32)
outputs, last_state = seq2seq.rnn_decoder(inputs, initial_state, cell)
output = tf.reshape(tf.concat(1, outputs), [-1, hidden_num])

logits = tf.matmul(output, softmax_w) + softmax_b
probs = tf.nn.softmax(logits)
loss = seq2seq.sequence_loss_by_example([logits],
        [tf.reshape(targets, [-1])],
        [tf.ones([batch_size * seq_length])],
        vocab_size)
self.cost = tf.reduce_sum(loss) / batch_size / seq_length


def fit(self, vocab_size=None, min_occurrences=1):
    word_counts = Counter()
    cooccurrence_counts = defaultdict(float)
    for region in self.tokenized_regions():
        word_counts.update(region)
        for left_context, word, right_context in self.region_context_windows(region):
            for i, context_word in enumerate(left_context[::-1]):
                cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
            for i, context_word in enumerate(right_context):
                cooccurrence_counts[(word, context_word)] += 1 / (i + 1)
    self._words = [word for word, count in word_counts.most_common(vocab_size)
                   if count >= min_occurrences]
    self._word_index = {word: i for i, word in enumerate(self._words)}
    word_set = set(self._words)
    self._cooccurrence_matrix = {
        (self._word_index[words[0]], self._word_index[words[1]]): count
        for words, count in cooccurrence_counts.items()
        if words[0] in word_set and words[1] in word_set
    }


graph = tf.Graph()
with graph.as_default():

  train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
  train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])

  embeddings = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
        stddev=1.0 / math.sqrt(embedding_size)))
  nce_weights = tf.Variable(
        tf.truncated_normal([vocabulary_size, embedding_size],
        stddev=1.0 / math.sqrt(embedding_size)))
  nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

  embed = tf.nn.embedding_lookup(embeddings, train_inputs)
  loss = tf.reduce_mean(
      tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                     num_sampled, vocabulary_size))

  optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)


with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  for epoch in xrange(epoch_size):
    batch_inputs, batch_labels = generate_batch(
        batch_size, num_skips, skip_window)
    feed_dict = {train_inputs : batch_inputs,
                 train_labels : batch_labels}

    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    print "loss: ", loss_val

  final_embeddings = embeddings.eval()


graph = tf.Graph()
with graph.as_default():
  focal_input = tf.placeholder(tf.int32, shape=[batch_size])
  context_input = tf.placeholder(tf.int32, shape=[batch_size])
  cooccu_count = tf.placeholder(tf.float32, shape=[batch_size])

  focal_embeddings = tf.Variable(tf.random_uniform(
          [vocab_size, embedding_size], 1.0, -1.0))
  context_embeddings = tf.Variable(tf.random_uniform(
          [vocab_size, embedding_size], 1.0, -1.0))
  focal_biases = tf.Variable(
          tf.random_uniform([vocab_size], 1.0, -1.0))
  context_biases = tf.Variable(
          tf.random_uniform([vocab_size], 1.0, -1.0))


  focal_embedding = tf.nn.embedding_lookup([focal_embeddings], focal_input)
  context_embedding = tf.nn.embedding_lookup([context_embeddings], context_input)
  focal_bias = tf.nn.embedding_lookup([focal_biases], focal_input)
  context_bias = tf.nn.embedding_lookup([context_biases], context_input)
  weighting_factor = tf.minimum(1.0,tf.pow(cooccu_count/count_max, scaling_factor))
  embedding_product = tf.reduce_sum(tf.mul(focal_embedding, context_embedding), 1)
  log_cooccurrences = tf.log(tf.to_float(cooccu_count))
  distance_expr = tf.square(tf.add_n([
                embedding_product, focal_bias, context_bias,
                tf.neg(log_cooccurrences)]))
  loss = tf.reduce_sum(weighting_factor * distance_expr)

  optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)

  combined_embeddings = focal_embeddings + context_embeddings

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  for epoch in xrange(epoch_size):
    i_s, j_s, counts = prepare_batches()
    feed_dict = {focal_input: i_s,
                 context_input: j_s,
                 cooccu_count: counts}

    _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
    print "loss: ", loss_val

  final_embeddings = combined_embeddings.eval()


from sklearn.manifold import TSNE


tsne = TSNE(perplexity=30, n_components=2,
            init='pca', n_iter=5000)
low_dim_embs = tsne.fit_transform(
        final_embeddings[:500,:])
import matplotlib.pyplot as plt

labels = [reverse_dictionary[i] for i in xrange(500)]
plt.figure(figsize=(18, 18))  #in inches
for i, label in enumerate(labels):
    x, y = low_dim_embs[i,:]
    plt.scatte(x, y)
    plt.annotate(label,
                 xy=(x, y),
                 xytext=(5, 2),
                 textcoords='offset points',
                 ha='right',
                 va='bottom')
plt.savefig(filename)


graph = tf.Graph()
with graph.as_default():
    pos_hs = tf.placeholder(tf.int32, shape=[None])
    pos_rs = tf.placeholder(tf.int32, shape=[None])
    pos_ts = tf.placeholder(tf.int32, shape=[None])
    neg_hs = tf.placeholder(tf.int32, shape=[None])
    neg_rs = tf.placeholder(tf.int32, shape=[None])
    neg_ts = tf.placeholder(tf.int32, shape=[None])


    ent_embeddings = tf.Variable(tf.truncated_normal(
      [ent_num, dim_num], stddev=1.0/math.sqrt(dim_num)))
    rel_embeddings = tf.Variable(tf.truncated_normal(
      [rel_num, dim_num], stddev=1.0/math.sqrt(dim_num)))

    phs = tf.nn.embedding_lookup(ent_embeddings, pos_hs)
    prs = tf.nn.embedding_lookup(rel_embeddings, pos_rs)
    pts = tf.nn.embedding_lookup(ent_embeddings, pos_ts)
    nhs = tf.nn.embedding_lookup(ent_embeddings, neg_hs)
    nrs = tf.nn.embedding_lookup(rel_embeddings, neg_rs)
    nts = tf.nn.embedding_lookup(ent_embeddings, neg_ts)

    pos_loss = tf.reduce_sum(tf.abs(phs + prs - pts), 1)
    neg_loss = tf.reduce_sum(tf.abs(nhs + nrs - nts), 1)
    base_loss = tf.reduce_sum(
            tf.nn.relu(pos_loss + margin - neg_loss))

    norm_loss = tf.reduce_sum(tf.reduce_sum(tf.pow(phs, 2), 1))
    norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(pts, 2), 1))
    norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(nhs, 2), 1))
    norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(nts, 2), 1))
    norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(prs, 2), 1))
    norm_loss += tf.reduce_sum(tf.reduce_sum(tf.pow(nrs, 2), 1))

    loss_fun = base_loss + alpha * norm_loss
    optimizer = tf.train.AdagradOptimizer(0.25).minimize(loss_fun)

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  for epoch in xrange(epoch_size):
    for batch in xrange(batch_num):
      batch_pos, batch_neg = data_sets.generate_batch(batch_size)
      feed_dict = {pos_hs: [x[0] for x in batch_pos],
                  pos_rs: [x[1] for x in batch_pos],
                  pos_ts: [x[2] for x in batch_pos],
                  neg_hs: [x[0] for x in batch_neg],
                  neg_rs: [x[1] for x in batch_neg],
                  neg_ts: [x[2] for x in batch_neg]}
      (_, loss_val) = session.run([optimizer, loss_fun], feed_dict=feed_dict)
      print 'loss: ', loss_val
  final_embeddings = {'ent':ent_embeddings.eval(),
                      'rel':rel_embeddings.eval()}




nemb = tf.nn.l2_normalize(emb, 1)
# Each row of a_emb, b_emb, c_emb is a word's embedding vector.
# They all have the shape [N, emb_dim]
a_emb = tf.gather(nemb, analogy_a)  # a's embs
b_emb = tf.gather(nemb, analogy_b)  # b's embs
c_emb = tf.gather(nemb, analogy_c)  # c's embs
# We expect that d's embedding vectors on the unit hyper-sphere is
# near: c_emb + (b_emb - a_emb), which has the shape [N, emb_dim].
target = c_emb + (b_emb - a_emb)
# Compute cosine distance between each pair of target and vocab.
# dist has shape [N, vocab_size].
dist = tf.matmul(target, nemb, transpose_b=True)
# For each question (row in dist), find the top 4 words.
_, pred_idx = tf.nn.top_k(dist, 4)

#a:北京,b:中国,c:巴黎,d:法国
#d与c+(b-a)是不是最接近
def eval_graph():
    analogy_a = tf.placeholder(dtype=tf.int32)
    analogy_b = tf.placeholder(dtype=tf.int32)
    analogy_c = tf.placeholder(dtype=tf.int32)
    nemb = tf.nn.l2_normalize(emb, 1)
    a_emb = tf.gather(nemb, analogy_a)
    b_emb = tf.gather(nemb, analogy_b)
    c_emb = tf.gather(nemb, analogy_c)
    target = c_emb + (b_emb - a_emb)
    dist = tf.matmul(target, nemb, transpose_b=True)
    _, idxs_graph = tf.nn.top_k(dist, 4)[1]
def eval():
    correct = 0
    total = len(questions)
    idxs = idxs_graph.eval()
    for q_id in xrange(total):
        for j in xrange(4):
          if idxs[q_id, j] == questions[q_id, 3]:
            correct += 1
            break
          elif idxs[q_id, j] in questions[q_id, :3]:
            continue
          else:
            break
    print("%d/%d, 准确率=%.2f%%" % (correct, total,
           correct * 100.0 / total))

norm += tf.reduce_sum(tf.reduce_sum(tf.square(prs), 1))
norm = tf.reduce_sum(tf.nn.relu(tf.reduce_sum(tf.square(phs), 1) - 1.0))

tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, num_sampled, vocabulary_size)

relu(logits) - logits * targets + log(1 + exp(-abs(logits)))

all_ids = array_ops.concat(0, [labels_flat, sampled])
all_w = embedding_ops.embedding_lookup(weights, all_ids)
all_b = embedding_ops.embedding_lookup(biases, all_ids)

row_wise_dots = math_ops.mul(expand_dims(inputs, 1), reshape(true_w, new_true_w_shape))
true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true]) + true_b

sampled_w = array_ops.slice(all_w, array_ops.pack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])
sampled_logits = math_ops.matmul(inputs, sampled_w, transpose_b=True) + sampled_b

out_logits = array_ops.concat(1, [true_logits, sampled_logits])

out_labels = array_ops.concat(
        1, [array_ops.ones_like(true_logits) / num_true,
        array_ops.zeros_like(sampled_logits)])