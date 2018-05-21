# encoding:utf-8

import numpy as np

#越小越好
def scoring(h, r, t, w1, w2, l2 = False):
    if l2:
        return -np.sum(np.power(np.tanh(h * w1) + np.tanh(r) - np.tanh(t * w2), 2))
    else:
        return -np.sum(np.abs(np.tanh(h * w1) + np.tanh(r) - np.tanh(t * w2)))

def scoring_batch(h, r, t, w1, w2, l2 = False):
    if l2:
        return -np.sum(np.power(np.tanh(h * w1) + np.tanh(r) - np.tanh(t * w2), 2), axis=1)
    else:
        return -np.sum(np.abs(np.tanh(h * w1) + np.tanh(r) - np.tanh(t * w2)), axis=1)

def fact_class(_pos_s, _neg_s, _mar = None):
    if _mar is None:
        min_v = min([min(_pos_s), min(_neg_s)])
        max_v = max([max(_pos_s), max(_neg_s)])
        step = (max_v - min_v) / 100.0
        best_score, best_mar = 0.0, 0.0
        for i in range(101):
            temp_mar = min_v + i * step
            temp_score = fact_class(_pos_s, _neg_s, temp_mar)
            if temp_score > best_score:
                best_score, best_mar = temp_score, temp_mar
        return best_mar, best_score
    else:
        l1 = len(filter(lambda x : x >= _mar, _pos_s))
        l2 = len(filter(lambda x : x < _mar, _neg_s))
        return 1.0 * (l1 + l2) / (len(_pos_s) + len(_neg_s))

def classification(_pos_scores, _neg_scores, _margins = None):
    if _margins is None:
        rel_margins = dict()
        for rel in set([x[1] for x in _pos_scores + _neg_scores]):
            rel_margins[rel], _ = fact_class([x[0] for x in _pos_scores if x[1] == rel],
                                          [x[0] for x in _neg_scores if x[1] == rel])
        rel_margins["all"], _ = fact_class([x[0] for x in _pos_scores],
                                        [x[0] for x in _neg_scores])
        return rel_margins, classification(_pos_scores, _neg_scores, rel_margins)
    else:
        all_margin = _margins["all"]
        acc_count = 0
        acc_count += len(filter(lambda x : x[0] >= _margins.get(x[1], all_margin), _pos_scores))
        acc_count += len(filter(lambda x : x[0] < _margins.get(x[1], all_margin), _neg_scores))
        total = len(_pos_scores) + len(_neg_scores)
        return acc_count * 1.0 / total

class Eval:
    def __init__(self, props, dataset = None, embeddings = None):
        self.init_props(props, dataset)
        self.loadresource(embeddings)

    def init_props(self, props, dataset = None):
        self.input_dir = props['input']
        self.output_dir = props['output']
        self.embed_size = props['size']
        self.nltype = props['type']
        self.l2 = props['l2']
        self.neg_type = props['neg_type']

        # print "input: ", self.input_dir
        # print "output: ", self.output_dir
        # print "size: ", self.embed_size
        # print "nltype: ", self.nltype
        # print "neg type: ", self.neg_type
        # print "is l2?: ", self.l2
        if dataset is None:
            import input_data
            print "read data from ", self.input_dir, ' with ', self.neg_type
            self.dataset = input_data.read_data_sets(self.input_dir, self.neg_type)
        else:
            self.dataset = dataset

    def loadresource(self, embeddings = None):
        self.ent_num = self.dataset.ent_num
        self.rel_num = self.dataset.rel_num
        if self.nltype == 3:
            self.type_num = self.dataset.type_num
        if embeddings is None:
            print "load embedding from ", self.output_dir
            self.embeddings = dict()
            self.embeddings['ent'] = np.load(self.output_dir + "ent_vecs.npy")
            self.embeddings['rel'] = np.load(self.output_dir + "rel_vecs.npy")
            if self.nltype == 2:#每个关系前后两个不同非线性操作
                self.embeddings['head'] = np.load(self.output_dir + "head.npy")
                self.embeddings['tail'] = np.load(self.output_dir + "tail.npy")
            elif self.nltype == 3:#每个类型一个非线性操作
                self.embeddings['type'] = np.load(self.output_dir + "type.npy")
            else:#每个关系一个非线性操作，前后一样
                self.embeddings['rwei'] = np.load(self.output_dir + "rwei.npy")
        else:
            self.embeddings = embeddings

    def tripleclassify(self):
        print "事实分类"
        if self.nltype == 2:
            valid_pos_scores = [(scoring(self.embeddings['ent'][h], self.embeddings['rel'][r], self.embeddings['ent'][t],
                                         self.embeddings['head'][r],  self.embeddings['tail'][r], self.l2), r)
                                for (h, r, t) in self.dataset.valid_pos]
            valid_neg_scores = [(scoring(self.embeddings['ent'][h], self.embeddings['rel'][r], self.embeddings['ent'][t],
                                         self.embeddings['head'][r],  self.embeddings['tail'][r], self.l2), r)
                                for (h, r, t) in self.dataset.valid_neg]
        elif self.nltype == 3:
            valid_pos_scores = [(scoring(self.embeddings['ent'][h], self.embeddings['rel'][r], self.embeddings['ent'][t],
                                         self.embeddings['type'][t1],  self.embeddings['type'][t2], self.l2), r)
                                for (h, r, t, t1, t2) in self.dataset.valid_pos_quin]
            valid_neg_scores = [(scoring(self.embeddings['ent'][h], self.embeddings['rel'][r], self.embeddings['ent'][t],
                                         self.embeddings['type'][t1],  self.embeddings['type'][t2], self.l2), r)
                                for (h, r, t, t1, t2) in self.dataset.valid_neg_quin]
        else:
            valid_pos_scores = [(scoring(self.embeddings['ent'][h], self.embeddings['rel'][r], self.embeddings['ent'][t],
                                         self.embeddings['rwei'][r],  self.embeddings['rwei'][r], self.l2), r)
                                for (h, r, t) in self.dataset.valid_pos]
            valid_neg_scores = [(scoring(self.embeddings['ent'][h], self.embeddings['rel'][r],self.embeddings['ent'][t],
                                         self.embeddings['rwei'][r], self.embeddings['rwei'][r], self.l2), r)
                                for (h, r, t) in self.dataset.valid_neg]

        rel_margins, valid_score = classification(valid_pos_scores, valid_neg_scores)
        print "valid accuracy %f" % valid_score

        if self.nltype == 2:
            test_pos_scores = [(scoring(self.embeddings['ent'][h], self.embeddings['rel'][r], self.embeddings['ent'][t],
                                         self.embeddings['head'][r],  self.embeddings['tail'][r], self.l2), r)
                                for (h, r, t) in self.dataset.test_pos]
            test_neg_scores = [(scoring(self.embeddings['ent'][h], self.embeddings['rel'][r], self.embeddings['ent'][t],
                                         self.embeddings['head'][r],  self.embeddings['tail'][r], self.l2), r)
                                for (h, r, t) in self.dataset.test_neg]
        elif self.nltype == 3:
            test_pos_scores = [(scoring(self.embeddings['ent'][h], self.embeddings['rel'][r], self.embeddings['ent'][t],
                                         self.embeddings['type'][t1],  self.embeddings['type'][t2], self.l2), r)
                                for (h, r, t, t1, t2) in self.dataset.test_pos_quin]
            test_neg_scores = [(scoring(self.embeddings['ent'][h], self.embeddings['rel'][r], self.embeddings['ent'][t],
                                         self.embeddings['type'][t1],  self.embeddings['type'][t2], self.l2), r)
                                for (h, r, t, t1, t2) in self.dataset.test_neg_quin]
        else:
            test_pos_scores = [(scoring(self.embeddings['ent'][h], self.embeddings['rel'][r], self.embeddings['ent'][t],
                                         self.embeddings['rwei'][r],  self.embeddings['rwei'][r], self.l2), r)
                                for (h, r, t) in self.dataset.test_pos]
            test_neg_scores = [(scoring(self.embeddings['ent'][h], self.embeddings['rel'][r], self.embeddings['ent'][t],
                                         self.embeddings['rwei'][r],  self.embeddings['rwei'][r], self.l2), r)
                                for (h, r, t) in self.dataset.test_neg]
        test_score = classification(test_pos_scores, test_neg_scores, rel_margins)
        print "test accuracy %f" % test_score
        return test_score

    def linkpredict(self):
        print "链接预测"
        raw_h_location, raw_t_location = list(), list()
        filter_h_location, filter_t_location = list(), list()
        num = 0

        if self.nltype == 3:
            for (h, r, t, t1, t2) in self.dataset.test_pos_quin:
                num += 1
                if num % 100 == 99:
                    print num
                h_scores = scoring_batch(self.embeddings['ent'],  self.embeddings['rel'][r], self.embeddings['ent'][t],
                                         self.embeddings['type'][t1],  self.embeddings['type'][t2], self.l2)
                h_locations = np.argsort(np.argsort(h_scores))
                loc = self.ent_num - h_locations[h]
                raw_h_location.append(loc)
                for j in self.dataset.r_t_hs_all[r][t]:
                    if h_locations[j] >= h_locations[h]:
                        loc -= 1
                filter_h_location.append(loc + 1)

                t_scores = scoring_batch(self.embeddings['ent'][h],  self.embeddings['rel'][r], self.embeddings['ent'],
                                         self.embeddings['type'][t1],  self.embeddings['type'][t2], self.l2)
                t_locations = np.argsort(np.argsort(t_scores))
                loc = self.ent_num - t_locations[t]
                raw_t_location.append(loc)
                for j in self.dataset.r_h_ts_all[r][h]:
                    if t_locations[j] >= t_locations[t]:
                        loc -= 1
                filter_t_location.append(loc + 1)

        else:
            for (h, r, t) in self.dataset.test_pos:
                num += 1
                if num % 100 == 99:
                    print num
                if self.nltype == 2:
                    h_scores = scoring_batch(self.embeddings['ent'],  self.embeddings['rel'][r], self.embeddings['ent'][t],
                                             self.embeddings['head'][r],  self.embeddings['tail'][r], self.l2)
                else:
                    h_scores = scoring_batch(self.embeddings['ent'],  self.embeddings['rel'][r], self.embeddings['ent'][t],
                                             self.embeddings['rwei'][r],  self.embeddings['rwei'][r], self.l2)
                h_locations = np.argsort(np.argsort(h_scores))
                loc = self.ent_num - h_locations[h]
                raw_h_location.append(loc)
                for j in self.dataset.r_t_hs_all[r][t]:
                    if h_locations[j] >= h_locations[h]:
                        loc -= 1
                filter_h_location.append(loc + 1)

                if self.nltype == 2:
                    t_scores = scoring_batch(self.embeddings['ent'][h],  self.embeddings['rel'][r], self.embeddings['ent'],
                                             self.embeddings['head'][r],  self.embeddings['tail'][r], self.l2)
                else:
                    t_scores = scoring_batch(self.embeddings['ent'][h],  self.embeddings['rel'][r], self.embeddings['ent'],
                                             self.embeddings['rwei'][r],  self.embeddings['rwei'][r], self.l2)
                t_locations = np.argsort(np.argsort(t_scores))
                loc = self.ent_num - t_locations[t]
                raw_t_location.append(loc)
                for j in self.dataset.r_h_ts_all[r][h]:
                    if t_locations[j] >= t_locations[t]:
                        loc -= 1
                filter_t_location.append(loc + 1)

        raw_h_mean = np.mean(raw_h_location)
        print ' h raw mean rank:%4.4f' % (raw_h_mean)
        count = len(filter(lambda e: e <= 10, raw_h_location))
        raw_h_hit10 = 1. * count / len(raw_h_location)
        print ' h raw Hits 10:', raw_h_hit10
        raw_t_mean = np.mean(raw_t_location)
        print ' t raw mean rank:%4.4f' % (raw_t_mean)
        count = len(filter(lambda e: e <= 10, raw_t_location))
        raw_t_hit10 = 1. * count / len(raw_t_location)
        print ' t raw Hits 10:', raw_t_hit10
        print 'mean rank: %d , Hits10:%4.4f' % (int(raw_h_mean + raw_t_mean) / 2, \
                                                (raw_h_hit10 + raw_t_hit10) / 2)

        print ''
        filter_h_mean = np.mean(filter_h_location)
        print ' h filter mean rank:', filter_h_mean
        count = len(filter(lambda e: e <= 10, filter_h_location))
        filter_h_hit10 = 1. * count / len(filter_h_location)
        print ' h filter Hits 10:%4.4f' % (filter_h_hit10)
        filter_t_mean = np.mean(filter_t_location)
        print ' t filter mean rank:', filter_t_mean
        count = len(filter(lambda e: e <= 10, filter_t_location))
        filter_t_hit10 = 1. * count / len(filter_t_location)
        print ' t filter Hits 10:', filter_t_hit10
        print 'mean rank: %d , Hits10:%4.4f' % (int(filter_h_mean + filter_t_mean) / 2, \
                                                (filter_h_hit10 + filter_t_hit10) / 2)
        print ''

        raw_h_location = np.asarray(raw_h_location)
        raw_t_location = np.asarray(raw_t_location)
        filter_h_location = np.asarray(filter_h_location)
        filter_t_location = np.asarray(filter_t_location)

        for name in ['1_1', '1_m', 'm_1', 'm_n']:
            tri_indexs = list()
            for xi, tri in enumerate(self.dataset.test_pos):
                if self.dataset.rel2reltype[tri[1]] == name:
                    tri_indexs.append(xi)
            if len(tri_indexs) == 0:
                continue
            sub_raw_h_location = raw_h_location[tri_indexs]
            raw_h_mean = np.mean(sub_raw_h_location)
            count = len(filter(lambda e: e <= 10, sub_raw_h_location))
            sub_raw_h_hit10 = 1. * count / len(sub_raw_h_location)
            print name + ': h raw mean rank:%4.4f' % raw_h_mean
            print name + ': h raw Hits 10:', sub_raw_h_hit10

            sub_raw_t_location = raw_t_location[tri_indexs]
            raw_t_mean = np.mean(sub_raw_t_location)
            count = len(filter(lambda e: e <= 10, sub_raw_t_location))
            sub_raw_t_hit10 = 1. * count / len(sub_raw_t_location)
            print name + ': t raw mean rank:%4.4f' % raw_t_mean
            print name + ': t raw Hits 10:', sub_raw_t_hit10

            sub_filter_h_location = filter_h_location[tri_indexs]
            filter_h_mean = np.mean(sub_filter_h_location)
            count = len(filter(lambda e: e <= 10, sub_filter_h_location))
            sub_filter_h_hit10 = 1. * count / len(sub_filter_h_location)
            print name + ': h raw mean rank:%4.4f' % filter_h_mean
            print name + ': h filter Hits 10:%4.4f' % sub_filter_h_hit10

            sub_filter_t_location = filter_t_location[tri_indexs]
            filter_t_mean = np.mean(sub_filter_t_location)
            count = len(filter(lambda e: e <= 10, sub_filter_t_location))
            sub_filter_t_hit10 = 1. * count / len(sub_filter_t_location)
            print name + ': t raw mean rank:%4.4f' % filter_t_mean
            print name + ': t filter Hits 10:%4.4f' % (sub_filter_t_hit10)

import tensorflow as tf
if __name__ == "__main__":
    flags = tf.app.flags
    flags.DEFINE_string("output", "/home/hesz/dataset/fb15k/params1/", "")
    flags.DEFINE_string("input", "/home/hesz/dataset/fb15k/data/", "")
    flags.DEFINE_integer("size", 50, "")
    flags.DEFINE_boolean("l2", True, "")
    flags.DEFINE_string("neg", "bern", "")
    flags.DEFINE_integer("type", 1, "the type of non-linear mapping")
    opts = flags.FLAGS

    props = dict()
    props['input'] = opts.input
    props['output'] = opts.output
    props['size'] = opts.size
    props['l2'] = opts.l2
    props['neg_type'] = opts.neg
    props['type'] = opts.type

    print "input: ", props['input']
    print "output: ", props['output']
    print "size: ", props['size']
    print "nltype: ", props['type']
    print "neg type: ", props['neg_type']
    print "is l2?: ", props['l2']
    print "norm entity: ", props['ent_norm']

    # props['input'] = '/home/hesz/dataset/fb15k/data/'
    # props['output'] = '/home/hesz/dataset/fb15k/params2/'
    # props['size'] = 50
    # props['l2'] = True
    # props['neg_type'] = 'unif'
    # props['type'] = 1
    eval = Eval(props)
    eval.tripleclassify()
    eval.linkpredict()