#encoding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

import numpy as np
import math


class NNLM():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int64, [args.batch_size, 1])
        self.test_words = tf.placeholder(tf.int64, [args.seq_length])

        with tf.variable_scope('nnlm' + 'embedding'):
            embeddings = tf.Variable(tf.random_uniform([args.vocab_size, args.word_dim], -1.0, 1.0))
        with tf.variable_scope('nnlm' + 'weight'):
            weight_h = tf.Variable(tf.truncated_normal([args.seq_length * args.word_dim + 1, args.hidden_num],
                            stddev=1.0 / math.sqrt(args.hidden_num)))
            softmax_w = tf.Variable(tf.truncated_normal([args.seq_length * args.word_dim, args.vocab_size],
                            stddev=1.0 / math.sqrt(args.seq_length * args.word_dim)))
            softmax_u = tf.Variable(tf.truncated_normal([args.hidden_num + 1, args.vocab_size],
                            stddev=1.0 / math.sqrt(args.hidden_num)))

        def infer_output(input_data):
            inputs_emb = tf.nn.embedding_lookup(embeddings, input_data) #[batch_size, seq_len, word_dim]
            #print inputs_emb.get_shape()
            inputs_emb = tf.reshape(inputs_emb, [-1, args.seq_length * args.word_dim])
            #print inputs_emb.get_shape()
            inputs_emb_add = tf.concat(1, [inputs_emb, tf.ones(tf.pack([tf.shape(input_data)[0], 1]))])
            #print inputs_emb_add.get_shape()

            inputs = tf.tanh(tf.matmul(inputs_emb_add, weight_h)) #[batch_size, hidden_num]
            #print inputs.get_shape()
            inputs_add = tf.concat(1, [inputs, tf.ones(tf.pack([tf.shape(input_data)[0], 1]))])
            #print inputs_add.get_shape()
            outputs = tf.matmul(inputs_add, softmax_u) + tf.matmul(inputs_emb, softmax_w) #[batch_size, vocab_size]
            outputs = tf.clip_by_value(outputs, 0.0, 10.0)
            outputs = tf.nn.softmax(outputs)
            #print outputs.get_shape()
            #print self.targets.get_shape()
            return outputs

        outputs = infer_output(self.input_data)
        one_hot_targets = tf.one_hot(tf.squeeze(self.targets), args.vocab_size, 1.0, 0.0)
        #print one_hot_targets.get_shape()

        self.test_outputs = infer_output(tf.expand_dims(self.test_words, 0))
        self.test_outputs = tf.arg_max(self.test_outputs, 1)
        #print self.test_outputs.get_shape()

        self.loss = loss = -tf.reduce_mean(tf.reduce_sum(tf.log(outputs) * one_hot_targets, 1))
        self.optimizer = tf.train.AdagradOptimizer(0.1).minimize(loss)


class RNNLM():
    def __init__(self, args, infer=False):
        self.args = args
        if infer:
            args.batch_size = 1
            args.seq_length = 1

        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        cell = cell_fn(args.hidden_num)

        self.cell = cell# = rnn_cell.MultiRNNCell([cell] * args.num_layers)

        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.initial_state = cell.zero_state(args.batch_size, tf.float32)

        with tf.variable_scope('rnnlm'):
            softmax_w = tf.get_variable("softmax_w", [args.hidden_num, args.vocab_size])
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [args.vocab_size, args.hidden_num])
                inputs = tf.split(1, args.seq_length, tf.nn.embedding_lookup(embedding, self.input_data))
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        def loop(prev, _):
            prev = tf.matmul(prev, softmax_w) + softmax_b
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        outputs, last_state = seq2seq.rnn_decoder(inputs, self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        output = tf.reshape(tf.concat(1, outputs), [-1, args.hidden_num])
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        self.probs = tf.nn.softmax(self.logits)
        loss = seq2seq.sequence_loss_by_example([self.logits],
                [tf.reshape(self.targets, [-1])],
                [tf.ones([args.batch_size * args.seq_length])],
                args.vocab_size)
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        self.final_state = last_state
        self.lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def sample(self, sess, words, vocab, num=200, prime='<START>', sampling_type=1):
        state = self.cell.zero_state(1, tf.float32).eval()
        prime = vocab.keys()[2]
        print prime
        for word in [prime]:
            print (word)
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed = {self.input_data: x, self.initial_state:state}
            [state] = sess.run([self.final_state], feed)

        def weighted_pick(weights):
            t = np.cumsum(weights)
            s = np.sum(weights)
            return(int(np.searchsorted(t, np.random.rand(1)*s)))

        ret = prime
        word = prime
        for n in range(num):
            x = np.zeros((1, 1))
            x[0, 0] = vocab[word]
            feed = {self.input_data: x, self.initial_state:state}
            [probs, state] = sess.run([self.probs, self.final_state], feed)
            p = probs[0]

            if sampling_type == 0:
                sample = np.argmax(p)
            elif sampling_type == 2:
                if word == '\n':
                    sample = weighted_pick(p)
                else:
                    sample = np.argmax(p)
            else: # sampling_type == 1 default:
                sample = weighted_pick(p)

            pred = words[sample]
            ret += ' ' + pred
            word = pred
        return ret
