# encoding:utf-8

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import argparse
import os, cPickle
import data_utils
import models
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--data-path', default="data")
arg('--model-path', default="run")
arg('--embedding-size', type=int, default=100)
arg('--hidden-size', type=int, default=400)
arg('--num-layers', type=int, default=2)
#arg('--dropout-keep-prob', type=float, default=0.5)
arg('--optimizer', default='adam', help='Optimizer: adam, adadelta')
arg('--learning-rate', type=float, default=0.01)
arg('--batch-size', type=int, default=128)
arg('--epoch-size', type=int, default=10)
arg('--checkpoint-step', type=int, default=2, help='do validation and save after each this many of steps.')
args = parser.parse_args()


def main():
    if not os.path.exists(args.model_path):
        os.mkdir(args.model_path)
    print "save mode to : ", args.model_path

    dump_path = os.path.join(args.data_path, "data.pkl")
    if not os.path.exists(dump_path):
        dataloader = data_utils.DataLoader(args.data_path)
        cPickle.dump(dataloader, open(dump_path, "wb"))
    else:
        dataloader = cPickle.load(open(dump_path, "rb"))

    print "train instance: ", len(dataloader.train_data)
    print "char size: ", dataloader.char_vocab_size
    print "tag size: ", dataloader.tag_vocab_size

    model = models.BiLSTM_CRF(dataloader.char_vocab_size, dataloader.tag_vocab_size,
                              dataloader.max_sent_len, None, args.embedding_size,
                              args.hidden_size, args.num_layers,
                              True, True)

    print "begining to train model"
    model.train(dataloader, args.batch_size, args.epoch_size,
                args.checkpoint_step, args.model_path)


if __name__ == '__main__':
    main()
    pass