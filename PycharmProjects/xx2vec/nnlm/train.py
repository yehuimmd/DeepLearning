#encoding:utf-8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from input_data import *

import tensorflow as tf
import argparse
import time
import os
from six.moves import cPickle
from model import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/xinhua',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='../data/',
                       help='directory to store checkpointed models')
    parser.add_argument('--batch_size', type=int, default=1200,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=5,
                       help='RNN sequence length')
    parser.add_argument('--hidden_num', type=int, default=256,
                       help='number of hidden layers')
    parser.add_argument('--word_dim', type=int, default=256,
                       help='number of word embedding')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--model', type=str, default='lstm',
                       help='rnn, gru, or lstm')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')

    args = parser.parse_args()

    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    #data_loader = TextLoader2(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    train_model = NNLM(args)
    #train_model_ = RNNLM(args)

    #test_words = ['必须','从', '根本', '上', '改变']
    #test_words = ['<START>','<START>','<START>', '党中央','国务院']
    test_words = ['<START>','<START>','<START>', '<START>','党中央']
    test_words_ids = [data_loader.vocab.get(w, 1) for w in test_words]

    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        for e in range(args.num_epochs):
            data_loader.reset_batch_pointer()
            for b in range(data_loader.num_batches):
                start = time.time()
                x, y = data_loader.next_batch()
                feed = {train_model.input_data: x, train_model.targets: y}
                train_loss,  _ = sess.run([train_model.loss, train_model.optimizer], feed)
                end = time.time()
                print("{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}" .format(
                        b, data_loader.num_batches,
                        e, train_loss, end - start))

                if b % 10 == 0:
                    feed = {train_model.test_words : test_words_ids}
                    test_outputs, _ = sess.run([train_model.test_outputs], feed)
                    for word in test_words:
                        print word
                    print data_loader.words[test_outputs[0]]

if __name__ == '__main__':
    main()
