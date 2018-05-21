# encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import os
import cPickle
import random
import codecs
import numpy as np
from collections import Counter

from utils import *

PAD, PAD_ID = '_PAD', 0


def tagging_sent(sent):
    words = convert_to_words(sent)
    return word_tagging_BIEO(words)


class DataLoader(object):
    def __init__(self, data_path):
        print "deal with : ", data_path
        self.data_path = data_path
        self.max_sent_len = 80
        self.create_vocbulary()
        self.load_data()

    def create_vocbulary(self, min_frq=0, max_vocab_size=40*1000):
        #先获取词表
        char_counts, tag_counts = Counter(), Counter()
        with codecs.open(os.path.join(self.data_path, 'train_data'), 'rb', 'utf-8') as f:
            for line in f:
                chars, tags = tagging_sent(line.strip())
                char_counts.update(chars)
                tag_counts.update(tags)
        print "char num: ", len(char_counts)
        print "tag num: ", len(tag_counts)
        self.char_vocab_list = [PAD] + list(char_counts.keys())
        self.char_vocab_size = len(self.char_vocab_list)
        self.char_vocab = dict([(x, y) for (y, x) in enumerate(self.char_vocab_list)])
        self.tag_vocab_list = list(tag_counts.keys())
        self.tag_vocab_size = len(self.tag_vocab_list)
        self.tag_vocab = dict([(x, y) for (y, x) in enumerate(self.tag_vocab_list)])

    def load_data(self):
        def filter_data(data):
            new_data = list()
            for char_ids, tag_ids in data:
                if len(char_ids) != len(tag_ids):
                    continue
                if len(char_ids) > self.max_sent_len:
                    continue
                new_data.append((char_ids, tag_ids))
            return new_data
        total_data = list()
        for name in ['train', 'valid', 'test']:
            data_insts = list()
            db_file = os.path.join(self.data_path, '%s_data' % name)
            print "load data from ", db_file
            with codecs.open(db_file, 'rb', 'utf-8') as f:
                for line in f:
                    chars, tags = tagging_sent(line.strip())
                    char_ids = [self.char_vocab.get(w) for w in chars]
                    tag_ids = [self.tag_vocab.get(w) for w in tags]
                    if None in char_ids or None in tag_ids:
                        continue
                    data_insts.append((char_ids, tag_ids))
            total_data.append(data_insts)
            print "origial entire line : ", len(data_insts)
        self.train_data = filter_data(total_data[0])
        self.valid_data = filter_data(total_data[1])
        self.test_data = filter_data(total_data[2])
        print "filter train line : ", len(self.train_data)
        print "filter valid line : ", len(self.valid_data)
        print "filter test line : ", len(self.test_data)

    def iter_batchs(self, batch_size=120, shuffle=True, data=None):
        if data is None:
            data = self.train_data
        if shuffle:
            random.shuffle(data)
        batch_num = len(data) // batch_size
        for kk in xrange(batch_num + 1):
            begin, end = batch_size * kk, batch_size * (kk + 1)
            if begin >= end:
                continue
            if end > len(data):
                end = len(data)
            batch_data = data[begin:end]
            inst_size = len(batch_data)
            word_chars = np.zeros((inst_size, self.max_sent_len), dtype=int)
            word_lengths = np.zeros(inst_size)
            char_tags = np.zeros((inst_size, self.max_sent_len), dtype=int)
            char_weights = np.ones((inst_size, self.max_sent_len), dtype=float)

            for ind in xrange(0, inst_size):
                char_ids, tag_ids = batch_data[ind]
                word_chars[ind] = np.array(char_ids + [PAD_ID] * (self.max_sent_len - len(char_ids)))
                word_lengths[ind] = len(char_ids)
                char_tags[ind] = np.array(tag_ids + [PAD_ID] * (self.max_sent_len - len(tag_ids)))
                char_weights[ind, len(char_ids):] = 0.0
            # yield np.transpose(word_chars), word_lengths, \
            #       np.transpose(char_tags), np.transpose(char_weights)
            yield word_chars, word_lengths, \
                  char_tags, char_weights

if __name__ == '__main__':
    data_path = 'data'
    dataloader = DataLoader(data_path)

    for chars, lens, tags, weis in dataloader.iter_batchs(5):
        print chars
        print lens
        print tags
        print weis



        break

    pass


