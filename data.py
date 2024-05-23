import random
import torch
import pickle as pkl
import six
import ijson
import tensorflow as tf
import numpy as np

import json

PAD, UNK = '<PAD>', '<UNK>'
CLS = '<CLS>'
STR, END = '<STR>', '<END>'
SEL, rCLS, TL = '<SEL>', '<rCLS>', '<TL>'

def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if six.PY3:
        if isinstance(text, str):
            return text
        elif isinstance(text, bytes):
            return text.decode("utf-8", "ignore")
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    elif six.PY2:
        if isinstance(text, str):
            return text.decode("utf-8", "ignore")
        elif isinstance(text, unicode):
            return text
        else:
            raise ValueError("Unsupported string type: %s" % (type(text)))
    else:
        raise ValueError("Not running on Python2 or Python 3?")


def ListsToTensor(xs, vocab=None, local_vocabs=None, unk_rate=0.):
    pad = vocab.padding_idx if vocab else 0

    def toIdx(w, i):
        if vocab is None:
            return w
        if isinstance(w, list):
            return [toIdx(_, i) for _ in w]
        if random.random() < unk_rate:
            return vocab.unk_idx
        if local_vocabs is not None:
            local_vocab = local_vocabs[i]
            if (local_vocab is not None) and (w in local_vocab):
                return local_vocab[w]
        return vocab.token2idx(w)

    max_len = max(len(x) for x in xs)
    ys = []
    for i, x in enumerate(xs): # change x value to lower(x)
        y = toIdx(x, i) + [pad] * (max_len - len(x))
        ys.append(y)
    data = torch.LongTensor(ys).t_().contiguous()
    return data


def ListsofStringToTensor(xs, vocab, max_string_len=30):  # 다 토큰들의 아이디로 들어간다.
    max_len = max(len(x) for x in xs)
    ys = []
    for x in xs: # change x value to lower(x)
        y = x + [PAD] * (max_len - len(x))
        zs = []
        for z in y:
            z = list(z[:max_string_len])
            zs.append(vocab.token2idx([STR] + z + [END]) + [vocab.padding_idx] * (max_string_len - len(z)))
        ys.append(zs)

    data = torch.LongTensor(ys).transpose(0, 1).contiguous()
    return data


def ArraysToTensor(xs):
    x = np.array([list(x.shape) for x in xs])
    shape = [len(xs)] + list(x.max(axis=0))
    data = np.zeros(shape, dtype=np.int)
    for i, x in enumerate(xs):
        slicing_shape = list(x.shape)
        slices = tuple([slice(i, i + 1)] + [slice(0, x) for x in slicing_shape])
        data[slices] = x
        tensor = torch.from_numpy(data).long()
    return tensor


class InputExample(object):
  """A single multiple choice question."""

  def __init__(
      self,
      example_id,
      question,
      answers,
      label,
      attribute,
  ):
    """Construct an instance."""


    self.example_id = example_id
    self.question = question
    self.answers = answers
    self.label = label
    self.attribute = attribute


class Vocab(object):
    def __init__(self, filename, min_occur_cnt, specials=None):

        idx2token = [PAD, UNK] + (specials if specials is not None else [])
        self._priority = dict()
        num_tot_tokens = 0
        num_vocab_tokens = 0
        for line in open(filename).readlines():
            try:
                token, cnt = line.strip().split('\t')
                cnt = int(cnt)
                num_tot_tokens += cnt
            except:
                print(line)

            if cnt >= min_occur_cnt:
                idx2token.append(token)
                num_vocab_tokens += cnt
            self._priority[token] = int(cnt)
        self.coverage = num_vocab_tokens / num_tot_tokens
        self._token2idx = dict(zip(idx2token, range(len(idx2token))))
        self._idx2token = idx2token
        self._padding_idx = self._token2idx[PAD]
        self._unk_idx = self._token2idx[UNK]

    def priority(self, x):
        return self._priority.get(x, 0)

    @property
    def size(self):
        return len(self._idx2token)

    @property
    def unk_idx(self):
        return self._unk_idx

    @property
    def padding_idx(self):
        return self._padding_idx

    def idx2token(self, x):
        if isinstance(x, list):
            return [self.idx2token(i) for i in x]
        return self._idx2token[x]

    def token2idx(self, x):
        if isinstance(x, list):
            return [self.token2idx(i) for i in x]

        return self._token2idx.get(x, self.unk_idx)

def relation_encoder(vocabs, data, train_flag=True):
    if train_flag:
        all_relations = dict()
        cls_idx = vocabs['relation'].token2idx(CLS)
        rcls_idx = vocabs['relation'].token2idx(rCLS)
        self_idx = vocabs['relation'].token2idx(SEL)

        all_relations[tuple([cls_idx])] = 0
        all_relations[tuple([rcls_idx])] = 1
        all_relations[tuple([self_idx])] = 2

        _relation_type = []

        for bidx, x in enumerate(data):
            n = len(x.attribute['entity'])
            brs = [[2] + [0] * (n)]
            for i in range(n):
                rs = [1]
                for j in range(n):
                    all_path = x.attribute['relation'][str(i)][(str(j))]
                    path = random.choice(all_path)['edge']
                    if len(path) == 0:  # self loop
                        path = [SEL]
                    if len(path) > 8:  # too long distance
                        path = [TL]

                    path = tuple(vocabs['relation'].token2idx(path))
                    rtype = all_relations.get(path, len(all_relations))
                    if rtype == len(all_relations):
                        all_relations[path] = len(all_relations)
                    rs.append(rtype)

                rs = np.array(rs, dtype=np.int)
                brs.append(rs)

            brs = np.stack(brs)
            _relation_type.append(brs)
        _relation_type = ArraysToTensor(_relation_type).transpose_(0, 2)  # 얘가 나중에 batch로 반환되는 relation애들이다.

        B = len(all_relations)

        _relation_bank = dict()
        _relation_length = dict()

        for k, v in all_relations.items():
            # relation
            _relation_bank[v] = np.array(k, dtype=np.int)
            _relation_length[v] = len(k)

        _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]
        _relation_length = [_relation_length[i] for i in range(len(all_relations))]
        _relation_bank = ArraysToTensor(_relation_bank).t_()
        _relation_length = torch.LongTensor(_relation_length)


    else:
        all_relations = dict()
        cls_idx = vocabs['relation'].token2idx(CLS)
        rcls_idx = vocabs['relation'].token2idx(rCLS)
        self_idx = vocabs['relation'].token2idx(SEL)
        pad_idx = vocabs['relation'].token2idx(PAD)

        all_relations[tuple([pad_idx])] = 0
        all_relations[tuple([cls_idx])] = 1
        all_relations[tuple([rcls_idx])] = 2
        all_relations[tuple([self_idx])] = 3

        _relation_type = []
        record = []
        bsz, num_entities, num_paths = 0, 0, 0

        for bidx, x in enumerate(data):
            n = len(x.attribute['entity'])
            num_entities = max(n + 1, num_entities)
            brs = [[[3]] + [[1]] * (n)]
            for i in range(n):
                rs = [[2]]
                for j in range(n):
                    all_r = []
                    all_path = x.attribute['relation'][str(i)][str(j)]
                    path0 = all_path[0]['edge']
                    if len(path0) == 0 or len(path0) > 8:
                        all_path = all_path[:1]
                    for path in all_path:
                        path = path['edge']
                        if len(path) == 0:  # self loop
                            path = [SEL]
                        if len(path) > 8:  # too long distance
                            path = [TL]
                        path = tuple(vocabs['relation'].token2idx(path))
                        rtype = all_relations.get(path, len(all_relations))
                        if rtype == len(all_relations):
                            all_relations[path] = len(all_relations)
                        all_r.append(rtype)

                    record.append(len(all_r))
                    num_paths = max(len(all_r), num_paths)
                    rs.append(all_r)
                brs.append(rs)
            _relation_type.append(brs)
        bsz = len(_relation_type)
        _relation_matrix = np.zeros((bsz, num_entities, num_entities, num_paths))

        for b, x in enumerate(_relation_type):
            for i, y in enumerate(x):
                for j, z in enumerate(y):
                    for k, r in enumerate(z):
                        _relation_matrix[b, i, j, k] = r

        _relation_type = torch.from_numpy(_relation_matrix).transpose_(0, 2).long()

        B = len(all_relations)
        _relation_bank = dict()
        _relation_length = dict()

        for k, v in all_relations.items():
            _relation_bank[v] = np.array(k, dtype=np.int)
            _relation_length[v] = len(k)
            # print(k, v)

        _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]
        _relation_length = [_relation_length[i] for i in range(len(all_relations))]
        _relation_bank = ArraysToTensor(_relation_bank).t_()
        _relation_length = torch.LongTensor(_relation_length)

    return _relation_type ,_relation_bank, _relation_length


def wiki_relation_encoder(args, vocabs, data, train_flag=True):

    data = [x.attribute for x in data]

    if train_flag:
        _wiki_relation_bank_total = []
        _wiki_relation_length_total = []
        _wiki_relation_type_total = []

        for bidx, x, in enumerate(data):
            all_relations = dict()
            cls_idx = vocabs['relation'].token2idx(CLS)
            rcls_idx = vocabs['relation'].token2idx(rCLS)
            self_idx = vocabs['relation'].token2idx(SEL)

            all_relations[tuple([cls_idx])] = 0
            all_relations[tuple([rcls_idx])] = 1
            all_relations[tuple([self_idx])] = 2
            _relation_type = []

            for oidx in range(args.n_answers):
                n = len(x['wiki_entity'][oidx])
                brs = [[2] + [0] * (n)]
                for i in range(n):
                    rs = [1]
                    for j in range(n):
                        all_path = x['wiki_relation'][oidx][str(i)][(str(j))]
                        path = random.choice(all_path)['edge']
                        if len(path) == 0:  # self loop
                            path = [SEL]
                        if len(path) > 8:  # too long distance
                            path = [TL]
                        # change path to lower(path)
                        path = tuple(vocabs['relation'].token2idx(path))
                        rtype = all_relations.get(path, len(all_relations))
                        if rtype == len(all_relations):
                            all_relations[path] = len(all_relations)
                        rs.append(rtype)

                    rs = np.array(rs, dtype=np.int)
                    brs.append(rs)

                brs = np.stack(brs)
                _relation_type.append(brs)
            _relation_type = ArraysToTensor(_relation_type).transpose_(0, 2)

            B = len(all_relations)
            _relation_bank = dict()
            _relation_length = dict()

            for k, v in all_relations.items():

                # relation
                _relation_bank[v] = np.array(k, dtype=np.int)
                _relation_length[v] = len(k)

            _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]
            _relation_length = [_relation_length[i] for i in range(len(all_relations))]
            _relation_bank = ArraysToTensor(_relation_bank).t_()
            _relation_length = torch.LongTensor(_relation_length)

            _wiki_relation_bank_total.append(_relation_bank)
            _wiki_relation_length_total.append(_relation_length)
            _wiki_relation_type_total.append(_relation_type)


    else:
        _wiki_relation_bank_total = []
        _wiki_relation_length_total = []
        _wiki_relation_type_total = []

        for bidx, x in enumerate(data):
            all_relations = dict()
            all_nodes = dict()
            cls_idx = vocabs['relation'].token2idx(CLS)
            rcls_idx = vocabs['relation'].token2idx(rCLS)
            self_idx = vocabs['relation'].token2idx(SEL)
            pad_idx = vocabs['relation'].token2idx(PAD)

            all_relations[tuple([pad_idx])] = 0
            all_relations[tuple([cls_idx])] = 1
            all_relations[tuple([rcls_idx])] = 2
            all_relations[tuple([self_idx])] = 3

            all_nodes[tuple([pad_idx])] = 0
            all_nodes[tuple([cls_idx])] = 1
            all_nodes[tuple([rcls_idx])] = 2
            all_nodes[tuple([self_idx])] = 3

            _relation_type = []
            record = []
            bsz, num_entities, num_paths = 0, 0, 0

            for oidx in range(args.n_answers):
                n = len(x['wiki_entity'][oidx])
                num_entities = max(n + 1, num_entities)
                brs = [[[3]] + [[1]] * (n)]
                for i in range(n):
                    rs = [[2]]
                    for j in range(n):
                        all_r = []
                        all_path = x['wiki_relation'][oidx][str(i)][str(j)]

                        path0 = all_path[0]['edge']

                        if len(path0) == 0 or len(path0) > 8:
                            all_path = all_path[:1]

                        for path in all_path:
                            node = path['node']
                            path = path['edge']

                            if len(path) == 0:  # self loop
                                path = [SEL]
                            if len(path) > 8:  # too long distance
                                path = [TL]
                            # change path to lower(path)
                            path = tuple(vocabs['relation'].token2idx(path))
                            rtype = all_relations.get(path, len(all_relations))

                            if rtype == len(all_relations):
                                all_relations[path] = len(all_relations)
                                all_nodes[path] = node
                            all_r.append(rtype)

                        record.append(len(all_r))
                        num_paths = max(len(all_r), num_paths)
                        rs.append(all_r)
                    brs.append(rs)
                _relation_type.append(brs)
            bsz = len(_relation_type)
            _relation_matrix = np.zeros((bsz, num_entities, num_entities, num_paths))

            for b, x in enumerate(_relation_type):
                for i, y in enumerate(x):
                    for j, z in enumerate(y):
                        for k, r in enumerate(z):
                            _relation_matrix[b, i, j, k] = r

            _relation_type = torch.from_numpy(_relation_matrix).transpose_(0, 2).long()

            B = len(all_relations)
            _relation_bank = dict()
            _relation_length = dict()

            for k, v in all_relations.items():
                _relation_bank[v] = np.array(k, dtype=np.int)
                _relation_length[v] = len(k)

            _relation_bank = [_relation_bank[i] for i in range(len(all_relations))]

            _relation_length = [_relation_length[i] for i in range(len(all_relations))]
            _relation_bank = ArraysToTensor(_relation_bank).t_()
            _relation_length = torch.LongTensor(_relation_length)

            _wiki_relation_bank_total.append(_relation_bank)
            _wiki_relation_length_total.append(_relation_length)
            _wiki_relation_type_total.append(_relation_type)
    return _wiki_relation_type_total,_wiki_relation_bank_total, _wiki_relation_length_total

def batchify(args, data, data_flag, vocabs, unk_rate=0., train=True):

    ####################### Question ############################
    _conc = ListsToTensor([[CLS] + x.attribute['entity'] for x in data], vocabs['entity'],
                          unk_rate=unk_rate)

    _conc_char = ListsofStringToTensor([[CLS] + x.attribute['entity'] for x in data], vocabs['entity_char'])


    _depth = ListsToTensor([[0] + x.attribute['depth'] for x in data])

    answer_tempelate = {'0': 0, '1': 1}
    #{'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}

    _options = [x.attribute['option'] for x in data]
    _answer_key = [answer_tempelate[str(x.label)] for x in data]

    _id = [x.example_id for x in data]
    _token_data = [x.question.split(' ') for x in data]
    if args.omcs:
        _evidence = [x.attribute['omcs'] for x in data]
    else:
        _evidence = ''


    local_token2idx = [x.attribute['token2idx'] for x in data]
    local_idx2token = [x.attribute['idx2token'] for x in data]

    augmented_token = [[STR] + x.attribute['token'] + [END] for x in data]

    _token_in = ListsToTensor(augmented_token, vocabs['token'], unk_rate=unk_rate)[:-1]
    _token_char_in = ListsofStringToTensor(augmented_token, vocabs['token_char'])[:-1]

    _token_out = ListsToTensor(augmented_token, vocabs['token'], local_token2idx)[1:]
    _cp_seq = ListsToTensor([x.attribute['cp_seq'] for x in data], vocabs['token'], local_token2idx)

    abstract = [x.attribute['abstract'] for x in data]

    _relation_type, _relation_bank, _relation_length = relation_encoder(vocabs, data, train_flag=train)

    ####################### Options ############################
    _wiki_concs = []
    _wiki_conc_chars = []
    _wiki_depths = []
    _wiki_relation_types = []
    _wiki_relation_banks = []
    _wiki_relation_lengths = []


    for b in data:

        _wiki_conc = ListsToTensor([[CLS] + x for x in b.attribute['wiki_entity']], vocabs['entity'],
                                 unk_rate=unk_rate)
        _wiki_conc_char = ListsofStringToTensor([[CLS] + x for x in b.attribute['wiki_entity']], vocabs['entity_char'])
        _wiki_depth = ListsToTensor([[0] + x for x in b.attribute['wiki_depth']])

        _wiki_concs.append(_wiki_conc)
        _wiki_conc_chars.append(_wiki_conc_char)
        _wiki_depths.append(_wiki_depth)

    _wiki_relation_type, _wiki_relation_bank, _wiki_relation_length = wiki_relation_encoder(args, vocabs, data, train_flag=train)

    ret = {
        'entity': _conc,
        'entity_char': _conc_char,
        'entity_depth': _depth,
        'relation': _relation_type,
        'relation_bank': _relation_bank,
        'relation_length': _relation_length,
        'wiki_entity': _wiki_concs,
        'wiki_entity_char':_wiki_conc_chars,
        'wiki_entity_depth': _wiki_depths,
        'wiki_relation': _wiki_relation_type,
        'wiki_relation_bank':_wiki_relation_bank,
        'wiki_relation_length':_wiki_relation_length,
        'local_idx2token': local_idx2token,
        'local_token2idx': local_token2idx,
        'token_in': _token_in,
        'token_char_in': _token_char_in,
        'token_out': _token_out,
        'cp_seq': _cp_seq,
        'abstract': abstract,
        'token_data': _token_data,
        'answers': _options,
        'answer_key': _answer_key,
        'id': _id,
        'evidence': _evidence,
        'data_flag':data_flag
    }

    return ret


def read_jsonl(input_file):
    with tf.gfile.Open(input_file, 'r') as f:

        return [json.loads(ln) for ln in f]


def read_large_file(objects, block_size=50):
    block = []
    for i, line in enumerate(objects):
        block.append(line)
        if len(block) == block_size:
            yield block
            block = []

    # don't forget to yield the last block
    if block:
        yield block

class DataLoader(object):
    def __init__(self, args, vocabs, lex_map, batch_size, flag, sampler):

        # if for_train == True:
        #     self.examples = read_jsonl(args.train_data_jsonl)
        # elif for_train == 'Eval':
        #     self.examples = read_jsonl(args.test_data_jsonl)
        #     for_train = False
        # else:
        #     self.examples = read_jsonl(args.dev_data_jsonl)

        self.train_examples = read_jsonl(args.train_data_jsonl)
        self.dev_examples = read_jsonl(args.dev_data_jsonl)
        self.test_examples = read_jsonl(args.test_data_jsonl)

        self.lex_map = lex_map
        self.vocabs = vocabs
        self.batch_size = batch_size
        self.flag = flag
        self.unk_rate = 0.
        self.record_flag = False
        self.args = args
        self.sampler = sampler
        self.filename = [self.args.train_data, self.args.dev_data, self.args.test_data]
        self.examples =  self.train_examples + self.dev_examples + self.test_examples
        # self.filename = [self.args.test_data]
        # self.examples =  self.test_examples


    def set_unk_rate(self, x):
        self.unk_rate = x

    def record(self):
        self.record_flag = True

    def __len__(self):
        return len(self.examples)

    def __iter__(self):
        global_idx = 0
        for curr_file in self.filename:
            with open(curr_file, 'rb') as input_file:
                objects = ijson.items(input_file, 'item')

                for block in read_large_file(objects, 16):
                    fin_examples = []
                    
                    for i, d in enumerate(block):
                        cp_seq, token2idx, idx2token = self.lex_map.get(d['entity'], self.vocabs['entity'])
                        d['cp_seq'] = cp_seq
                        d['token2idx'] = token2idx
                        d['idx2token'] = idx2token

                        tid = d['id']
                        text = d['sentences']
                        targets = d['option']
                        target = d['answer_key']
                        if self.args.omcs:
                            ex = [ex for ex in self.examples if ex['id'] == qid][0]
                            omcss = [
                                choice['omcs'][0] for choice in ex['text']['choices']]

                            d['omcs'] = omcss

                        line_ex = InputExample(tid, text, targets, target, d)
                        fin_examples.append(line_ex)

                    idx = list(range(len(fin_examples)))

                    batches = []
                    num_tokens, data = 0, []
                    for i in idx:
                        data_flag = 'dev'
                        data.append(fin_examples[i])
                        #print(len(data),i, self.args.batch_size, self.batch_size)
                        if i % self.args.batch_size == (self.args.batch_size - 1):
                            if (global_idx in self.sampler):
                                data_flag= 'train'
                            batches.append([data, data_flag])
                            #print("batch length", len(batches[0][0]), data_flag)
                            num_tokens, data = 0, []
                            global_idx += 1

                    for batch in batches:
                        if len(batch[0]) != self.args.batch_size:
                            self.args.batch_size = len(batch[0])

                        if not self.record_flag:
                            if batch[1] == 'train':
                                yield batchify(self.args, batch[0], batch[1], self.vocabs, self.unk_rate, self.flag)
                            else:
                                yield batchify(self.args, batch[0], batch[1], self.vocabs, self.unk_rate, False)

                        else:
                            if batch[1] == 'train':
                                yield batchify(self.args, batch[0], batch[1], self.vocabs, self.unk_rate, self.flag), batch
                            else:
                                yield batchify(self.args, batch[0], batch[1], self.vocabs, self.unk_rate, False), batch
