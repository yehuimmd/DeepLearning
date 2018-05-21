# encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import os
import cPickle
import random, re
import nltk
import codecs
import math
import numpy as np
from collections import Counter


def tokenizer(sent):
    words = [w.strip() for w in nltk.word_tokenize(sent)]
    words = [w.lower() for w in words if len(w) > 0]
    return words

class DataLoader(object):
    def __init__(self, data_path):
        print "deal with : ", data_path
        self.data_path = data_path
        self.create_vocbulary()
        self.load_data()

    def load_data(self):
        dataset = list()
        with codecs.open(self.data_path, 'rb', 'utf-8') as f:
            for line in f:
                label, text = line.split('\t', 1)
                x_data, y_data = list(), list()
                if label == '1':
                    y_data = [1.0, 0.0]
                else:
                    y_data = [0.0, 1.0]
                word_counts = Counter()
                word_counts.update(tokenizer(text))
                for word in self.vocab_list:
                    value = 1.0 if word in word_counts else 0.0
                    x_data.append(value)
                dataset.append((x_data, y_data))
        self.dataset = dataset
        print "instance number : ", len(self.dataset)

    def create_vocbulary(self, max_vocab_size=30*1000):
        #先获取词表
        word_counts = Counter()
        with codecs.open(self.data_path, 'rb', 'utf-8') as f:
            for line in f:
                label, text = line.split('\t', 1)
                words = tokenizer(text)
                word_counts.update(words)
        word_frqs = word_counts.most_common(max_vocab_size)
        self.vocab_list = [x[0] for x in word_frqs]
        self.vocab_size = len(self.vocab_list)
        self.vocab = dict([(x, y) for (y, x) in enumerate(self.vocab_list)])
        print "vocab size: ", self.vocab_size

if __name__ == '__main__':
    data_path = 'review_polarity/total.txt'
    dataloader = DataLoader(data_path)
    vector, label = dataloader.dataset[5]
    print vector
    print label
    print len(vector)





