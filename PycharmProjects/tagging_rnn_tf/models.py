# encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import math, time
import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn, legacy_seq2seq
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode

class BiLSTM_CRF(object):
    def __init__(self, num_chars, num_tags, num_steps=200,
                 embedding_matrix=None, emb_dim=100,
                 hidden_dim=200, num_layers=1,
                 is_training=True, is_crf=True):
        self.learning_rate = 0.002
        self.dropout_rate = 0.5

        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.num_steps = num_steps
        self.num_chars = num_chars
        self.num_tags = num_tags

        self.is_crf = is_crf
        self.global_step = tf.Variable(0, trainable=False)

        # placeholder of x, y and weights
        self.inputs = tf.placeholder(tf.int32, [None, self.num_steps])
        self.targets = tf.placeholder(tf.int32, [None, self.num_steps])
        self.input_lens = tf.placeholder(tf.int32, [None])
        self.targets_weights = tf.placeholder(tf.float32, [None, self.num_steps])

        # char embedding
        if embedding_matrix is not None:
            self.embedding = tf.Variable(embedding_matrix, name="emb", dtype=tf.float32)
        else:
            self.embedding = tf.get_variable("emb", [self.num_chars, self.emb_dim])
        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)  # [batch, step, dim]
        self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])  # [step, batch, dim]

        emb_inputs = tf.unstack(self.inputs_emb) # step个，每个为[batch, dim]
        for layer_i in xrange(self.num_layers):
            with tf.variable_scope('encoder_layer_%d' % layer_i):
                cell_fw = rnn.LSTMCell(
                    self.hidden_dim,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=123),
                    state_is_tuple=False)
                cell_bw = rnn.LSTMCell(
                    self.hidden_dim,
                    initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=113),
                    state_is_tuple=False)
                if is_training:
                    cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob=(1 - self.dropout_rate))
                    cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=(1 - self.dropout_rate))
                (emb_inputs, fw_state, bw_state) = rnn.static_bidirectional_rnn(
                    cell_fw, cell_bw, emb_inputs, dtype=tf.float32,
                    sequence_length=self.input_lens)

        self.outputs = emb_inputs #step个, [batch, tag_num]

        with tf.variable_scope('output_projection'):
            out_w = tf.get_variable(
                'out_w', [self.hidden_dim * 2, self.num_tags], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-4))
            out_w_t = tf.transpose(out_w)
            out_v = tf.get_variable(
                'out_v', [self.num_tags], dtype=tf.float32,
                initializer=tf.truncated_normal_initializer(stddev=1e-4))

        self.logits = [tf.matmul(x, out_w) + out_v for x in self.outputs]
        self.scores = tf.transpose(tf.stack(self.logits), [1, 0, 2])

        if not self.is_crf:
            self.loss = legacy_seq2seq.sequence_loss(
                self.logits, tf.unstack(tf.transpose(self.targets)),
                tf.unstack(tf.transpose(self.targets_weights)))
        else:
            print "use crf train"

            crf_logits = tf.stack(self.logits)  # [step, batch, tag]
            crf_logits = tf.transpose(crf_logits, [1, 0, 2]) # [batch, step, tag]
            crf_logits = tf.reshape(crf_logits, [-1, self.num_steps, self.num_tags]) # [batch, step, tag]

            self.trans = tf.get_variable(
                "transitions",
                shape=[self.num_tags, self.num_tags],
                initializer=tf.truncated_normal_initializer(stddev=1e-4))

            log_likelihood, self.trans = crf_log_likelihood(
                inputs=crf_logits,
                tag_indices=self.targets,
                sequence_lengths=self.input_lens,
                transition_params=self.trans)
            self.loss = tf.reduce_mean(-log_likelihood)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.device("/cpu:0"):
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=2)

    def initilize(self, model_dir, session=None):
        ckpt = tf.train.latest_checkpoint(model_dir)
        if ckpt:
            self.saver.restore(session, ckpt)
            print "Reading model parameters from %s" % ckpt
        else:
            print("Creating model with fresh parameters.")
            session.run(tf.global_variables_initializer())

    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size, num_steps, num_tags] float32, logits
        :param lengths: [batch_size] int32, real length of each sequence
        :param matrix: transaction matric for inference
        :return:
        """
        # inference final labels usa viterbi Algorithm
        paths = []
        for score, length in zip(logits, lengths):
            if self.is_crf:
                path, _ = viterbi_decode(score, matrix)
            else:
                path = tf.argmax(score, axis=-1).eval()
            paths.append(path)
        return paths

    def step(self, chars, lens, tags, weis, is_train, session=None):
        feed = dict()
        feed[self.inputs] = chars
        feed[self.input_lens] = lens
        feed[self.targets] = tags
        feed[self.targets_weights] = weis

        if is_train:
            loss, _ = session.run([self.loss, self.optimizer], feed)
            return None, loss
        else:
            trans = self.trans.eval()
            loss, scores = session.run([self.loss, self.scores], feed)
            batch_tags = self.decode(scores, lens, trans)
            return batch_tags, loss

    def train(self, dataloader, batch_size, epoch_size, step_per_checkpoint, model_dir):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as session:
            self.initilize(model_dir, session)
            print "epoch size is : ", epoch_size
            print "batch size is : ", batch_size
            for epoch in xrange(1, epoch_size + 1):
                epoch_start = time.time()
                loss_sum = 0
                run_id, loss_run = 0, 0
                start = time.time()
                for chars, lens, tags, weis in dataloader.iter_batchs(batch_size):
                    _, step_loss = self.step(chars, lens, tags, weis, True, session)
                    loss_sum += step_loss
                    loss_run += step_loss
                    run_id += 1
                    if run_id % 100 == 0:
                        print "run %d loss %.5f time %.2f" % (run_id, loss_run, time.time() - start)
                        start = time.time()
                        loss_run = 0

                if epoch % step_per_checkpoint == 0:
                    self.saver.save(session, os.path.join(model_dir, 'checkpoint'), global_step=self.global_step)
                epoch_finish = time.time()
                print "epoch %d loss %.5f time %.2f" % (epoch, loss_sum, epoch_finish - epoch_start)

                # 测试
                for chars_t, lens_t, tags_t, weis_t in dataloader.iter_batchs(3, data=dataloader.test_data):
                    predicts, step_loss = self.step(chars_t, lens_t, tags_t, weis_t, False, session)
                    print "real: "
                    print tags_t
                    print "predict: "
                    print predicts
                    break