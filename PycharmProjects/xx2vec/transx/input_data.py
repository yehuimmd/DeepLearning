# encoding:utf-8
import random
import os

class DataSets:
    def __init__(self, data_dir, neg_type = 'bern'):
        self.data_dir = data_dir
        self.neg_type = neg_type
        print 'neg type is ', self.neg_type
        self.prepare()
        self.load_data()

    def prepare(self):
        self.ent2id, self.id2ent = dict(), dict()
        self.rel2id, self.id2rel = dict(), dict()
        for ent in map(str.strip, open(self.data_dir + "entities", "r").readlines()):
            self.ent2id[ent] = len(self.ent2id)
            self.id2ent[len(self.id2ent)] = ent
        for rel in map(str.strip, open(self.data_dir + "relations", "r").readlines()):
            self.rel2id[rel] = len(self.rel2id)
            self.id2ent[len(self.id2rel)] = rel
        self.ent_num, self.rel_num = len(self.ent2id), len(self.rel2id)
        print "entity num : " + str(self.ent_num)
        print "relation num : " + str(self.rel_num)

        self.rel2tphhpt = dict() #value的第一个值为平均每个h有多少个不同的t,第二个值为平均每个t有多少个不同的h
        for line in map(str.strip, open(self.data_dir + "rel2tphhpt.txt", "r").readlines()):
            rel, tph, hpt = line.split('\t')[0:3]
            self.rel2tphhpt[int(rel)] = (float(tph), float(hpt))

        self.rel2reltype = dict()
        for line in map(str.strip, open(self.data_dir + "relation2reltype.txt", "r").readlines()):
            rel, reltype = line.split('\t')[0:2]
            self.rel2reltype[int(rel)] = reltype


    def read_triple(self, name):
        triples = map(str.strip, open(name, "r").readlines())
        triples_tris = [map(str.strip, tri) for tri in map(str.split, triples)]
        return [(self.ent2id[x[0]], self.rel2id[x[1]], self.ent2id[x[2]]) for x in triples_tris]

    #把v加入d这个dict中
    def add_dict_kv(self, dic, k, v):
        vs = dic.get(k, set())
        vs.add(v)
        dic[k] = vs
    def add_dict_kkv(self, dic, k1, k2, v):
        k2vs = dic.get(k1, dict())
        vs = k2vs.get(k2, set())
        vs.add(v)
        k2vs[k2] = vs
        dic[k1] = k2vs

    def load_data(self):
        self.train_tri = self.read_triple(self.data_dir + "train-entity-facts")
        self.train_tri_set = set(self.train_tri)
        self.tri_num = len(self.train_tri)
        print "triple num : " + str(self.tri_num)
        self.valid_pos = self.read_triple(self.data_dir + "valid-entity-facts")
        self.test_pos = self.read_triple(self.data_dir + "test-entity-facts")

        if self.neg_type == 'bern':
            self.valid_neg = self.read_triple(self.data_dir + "valid-entity-facts_neg_bern")
            self.test_neg = self.read_triple(self.data_dir + "test-entity-facts_neg_bern")
        else:
            self.valid_neg = self.read_triple(self.data_dir + "valid-entity-facts_neg_unif")
            self.test_neg = self.read_triple(self.data_dir + "test-entity-facts_neg_unif")

        self.heads = set([x[0] for x in self.train_tri + self.valid_pos + self.test_pos])
        self.tails = set([x[2] for x in self.train_tri + self.valid_pos + self.test_pos])
        self.r_heads_train, self.r_tails_train = dict(), dict()
        self.r_heads_all, self.r_tails_all = dict(), dict()
        self.r_h_ts_train, self.r_t_hs_train = dict(), dict()
        self.r_h_ts_all, self.r_t_hs_all = dict(), dict()

        for (h, r, t) in self.train_tri:
            self.add_dict_kv(self.r_heads_train, r, h)
            self.add_dict_kv(self.r_tails_train, r, t)
            self.add_dict_kv(self.r_heads_all, r, h)
            self.add_dict_kv(self.r_tails_all, r, t)
            self.add_dict_kkv(self.r_h_ts_train, r, h, t)
            self.add_dict_kkv(self.r_t_hs_train, r, t, h)
            self.add_dict_kkv(self.r_h_ts_all, r, h, t)
            self.add_dict_kkv(self.r_t_hs_all, r, t, h)
        for (h, r, t) in self.valid_pos + self.test_pos:
            self.add_dict_kv(self.r_heads_all, r, h)
            self.add_dict_kv(self.r_tails_all, r, t)
            self.add_dict_kkv(self.r_h_ts_all, r, h, t)
            self.add_dict_kkv(self.r_t_hs_all, r, t, h)

    def exist(self, h, r, t):
        return (h, r, t) in self.train_tri_set

    #neg_scope表示是否从限定的里面抽取
    def generate_batch(self, batch_size, neg_scope = False):
        batch_pos = random.sample(self.train_tri, batch_size)
        batch_neg = list()
        for (h, r, t) in batch_pos:
            h2, r2, t2 = h, r, t
            scope, num = neg_scope, 0
            head_prop = 500
            if self.neg_type == 'bern':
                tph, hpt = self.rel2tphhpt[r]
                head_prop = int(tph * 1000.0 / (tph + hpt))

            while True:
                if random.randint(0, 999) < head_prop:
                    if scope:
                        h2 = random.sample(self.r_heads_train[r], 1)[0]
                    else:
                        h2 = random.randint(0, self.ent_num - 1)
                else:
                    if scope:
                        t2 = random.sample(self.r_tails_train[r], 1)[0]
                    else:
                        t2 = random.randint(0, self.ent_num - 1)
                if not self.exist(h2, r2, t2):
                    break
                else:
                    num += 1
                    if num > 10:
                        scope = False
            batch_neg.append((h2, r2, t2))
        return batch_pos, batch_neg

    def generate_batch_joint(self, batch_size, neg_scope = False):
        batch_pos = list()
        batch_neg = list()
        for (h, r, t) in random.sample(self.train_tri, batch_size):
            h2, r2, t2 = h, r, t
            scope, num = neg_scope, 0
            head_prop = 500
            is_head = True
            if self.neg_type == 'bern':
                tph, hpt = self.rel2tphhpt[r]
                head_prop = int(tph * 1000.0 / (tph + hpt))
            while True:
                if random.randint(0, 999) < head_prop:
                    is_head = True
                    if scope:
                        h2 = random.sample(self.r_heads_train[r], 1)[0]
                    else:
                        h2 = random.randint(0, self.ent_num - 1)
                else:
                    is_head = False
                    if scope:
                        t2 = random.sample(self.r_tails_train[r], 1)[0]
                    else:
                        t2 = random.randint(0, self.ent_num - 1)
                if not self.exist(h2, r2, t2):
                    break
                else:
                    num += 1
                    if num > 10:
                        scope = False
            if is_head:
                for tmp_h in self.r_t_hs_train[r][t]:
                    batch_pos.append((tmp_h, r, t))
                    batch_neg.append((h2, r2, t2))
            else:
                for tmp_t in self.r_h_ts_train[r][h]:
                    batch_pos.append((h, r, tmp_t))
                    batch_neg.append((h2, r2, t2))

        return batch_pos, batch_neg


def read_data_sets(data_dir, neg_type = 'bern'):
    return DataSets(data_dir, neg_type)