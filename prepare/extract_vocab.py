#!/usr/bin/env python
#coding: utf-8
from smatch import AMR
from AMRGraph import AMRGraph
from collections import Counter
import json
from AMRGraph import  _is_abs_form
from multiprocessing import Pool
from tqdm.auto import tqdm, trange
import os

class LexicalMap(object):

    def __init__(self):
        pass

    def get(self, entity, vocab=None):
        # entity
        cp_seq = []
        for ent in entity:
            cp_seq.append(ent)

        if vocab is None:
            return cp_seq

        new_tokens = set(cp for cp in cp_seq if vocab.token2idx(cp) == vocab.unk_idx)
        token2idx, idx2token = dict(), dict()
        nxt = vocab.size
        for x in new_tokens:
            token2idx[x] = nxt
            idx2token[nxt] = x
            nxt +=1

        return cp_seq, token2idx, idx2token

class AMRIO:
    def __init__(self):
        pass

    @staticmethod
    def read(file_path):
        with open(file_path, encoding='utf-8') as f:
            for line in f:
                line = line.rstrip()

                if line.startswith('# ::id '):
                    amr_id = line[len('# ::id '):]
                elif line.startswith('# ::tokens '):
                    tokens = json.loads(line[len('# ::tokens '):])
                    tokens = [ to if _is_abs_form(to) else to.lower() for to in tokens]
                elif line.startswith('# ::lemmas '):
                    lemmas = json.loads(line[len('# ::lemmas '):])
                    lemmas = [le if _is_abs_form(le) else le.lower() for le in lemmas]
                elif line.startswith('# ::pos_tags '):
                    pos_tags = json.loads(line[len('# ::pos_tags '):])
                elif line.startswith('# ::ner_tags '):
                    ner_tags = json.loads(line[len('# ::ner_tags '):])
                elif line.startswith('# ::abstract_map '):
                    abstract_map = json.loads(line[len('# ::abstract_map '):])
                elif line.startswith('# ::option '):
                    option = line[len('# ::option '):]
                    option = ast.literal_eval(option)
                elif line.startswith('# ::target '):
                    target = line[len('# ::target '):]
                # elif line.startswith('# ::snt '):
                    # sentence = line[len('# ::snt '):]
                # elif line.startswith('# ::answer '):
                #     answer = line[len('# ::answer '):]
                # elif line.startswith('# ::save-date '):
                elif line.startswith('# ::snt '):
                    sentence = line[len('# ::snt '):]
                    graph_line = AMR.get_amr_line(f)
                    print(amr_id)
                    raw_amr = AMR.parse_AMR_line(graph_line)
                    if raw_amr != None:
                        myamr = AMRGraph(raw_amr)
                    else:
                        continue

                    yield tokens, lemmas, abstract_map, myamr, target, option, amr_id, sentence, raw_amr

def read_file(filename):
    # read preprocessed amr file
    token, lemma, abstract, amrs, targets, options, amr_ids, sentences, raw_amrs = [], [], [], [], [], [], [], [], []


    for _tok, _lem, _abstract, _myamr, _target, _option, _amr_id, _sentence, _raw_amr in AMRIO.read(filename):
        token.append(_tok)
        lemma.append(_lem)
        abstract.append(_abstract)
        amrs.append(_myamr)
        targets.append(_target)
        options.append(_option)
        amr_ids.append(_amr_id)
        sentences.append(_sentence)
        raw_amrs.append(_raw_amr)

    print ('read from %s, %d amrs'%(filename, len(token)))
    return amrs, token, lemma, abstract, targets, options, amr_ids, sentences, raw_amrs

def make_vocab(batch_seq, char_level=False):
    cnt = Counter()
    for seq in batch_seq:
        cnt.update(seq)
    if not char_level:
        return cnt
    char_cnt = Counter()
    #print(cnt)
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt
"""
def make_vocab_list(cnt, char_cnt, batch_seq, char_level=False):
    for seq in batch_seq:
        cnt.update(seq)
    if not char_level:
        return cnt
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt
"""
def write_vocab(vocab, path):
    with open(path, 'w') as fo:
        for x, y in vocab.most_common():
        #for x, y in vocab:
            fo.write('%s\t%d\n'%(x,y))



import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--amr_files', type=str, nargs='+')
    parser.add_argument('--nprocessors', type=int, default=4)
    parser.add_argument('--extend', type=bool, default=False)
    parser.add_argument('--option_amr', type=bool, default=False)
    parser.add_argument('--entity_seed', type=str)
    parser.add_argument('--answer_len', type=int, default=5)
    parser.add_argument('--omcs', type=bool, default=False)
    return parser.parse_args()

import ast

if __name__ == "__main__":
    # collect entities and relations

    args = parse_config()

    amrs, token, lemma, abstract, targets, options, amr_ids, sentences, raw_amrs = read_file(args.train_data)
    lexical_map = LexicalMap()


    def work(data):
        amr, lem, tok = data
        entity, depth, relation, ok, ARGs = amr.collect_entities_and_relations()
        assert ok, "not connected"
        lexical_entities = set(lexical_map.get(entity))

        return entity, depth, relation, ARGs

    pool = Pool(args.nprocessors)

    if args.extend == True:
         res = pool.map(work, zip(amrs, lemma, token), len(amrs) // args.nprocessors)
    else:
         res = pool.map(work, zip(amrs, lemma, token), len(amrs) // args.nprocessors)

    tot_pairs = 0
    multi_path_pairs = 0
    tot_paths = 0
    extreme_long_paths = 0
    avg_path_length = 0.
    ent, rel = [], []
    token_list = token

    for entity, depth, relation, ARGs in res:
        ent.append(entity)

        for x in relation:
             for y in relation[x]:
                tot_pairs += 1
                if len(relation[x][y]) > 1:
                    multi_path_pairs += 1
                for path in relation[x][y]:
                    tot_paths += 1
                    path_len = path['length']
                    rel.append(path['edge'])
                    if path_len > 8:
                        extreme_long_paths += 1
                    avg_path_length += path_len

    print("amr reading complete")
    print("tagme vocab started")
    tagme_wiki_vocab_folder = './wikinet/vocab/politifact_wiki_vocab/'
    files = [f for f in os.listdir(tagme_wiki_vocab_folder) if os.path.isfile(os.path.join(tagme_wiki_vocab_folder, f))]
    for file in files:
        file_path = os.path.join(tagme_wiki_vocab_folder, file)
        with open(file_path, 'r') as f:
            for line in tqdm(f.readlines()):
                line = line.strip().split('\t')
                try:
                    for i in line[1:]:
                        en = ast.literal_eval(i)
                        token.append(en[0])
                        token_list.append([en[0]])
                        token.append(en[2])
                        token_list.append([en[2]])
                        rel.append([en[1]])
                   # print(line)
                except IndexError:
                    print('pass')
    print("wikinet started")
    #make relation dictionary
    with open('./wikinet/wikinet.txt', 'r') as f:
        for line in tqdm(f.readlines()):
            line = line.strip().split('\t')
            try:
                for i in line[1:]:
                    en = ast.literal_eval(i)
                    token.append(en[0])
                    token.append(en[2])
                    rel.append([en[1]])
               # print(line)
            except IndexError:
                print('pass')

            
    # make vocabularies
    # print(token)
    # print(rel)
    print("writing1")
    token_vocab, token_char_vocab = make_vocab(token, char_level=True)
    #print(token_vocab)
    #token_vocab, token_char_vocab = make_vocab_list(token_vocab, token_char_vocab, token_list, char_level=True)
    print("writing2")
    lemma_vocab, lemma_char_vocab = make_vocab(lemma, char_level=True)
    print("writing3")
    ent_vocab, ent_char_vocab = make_vocab(ent, char_level=True)

    num_token = sum(len(x) for x in token)
    rel_vocab = make_vocab(rel)
    #print(rel_vocab)

    print('make vocabularies')
    write_vocab(token_vocab, 'token_vocab')
    write_vocab(token_char_vocab, 'token_char_vocab')
    # write_vocab(lemma_vocab, 'lem_vocab')
    # write_vocab(lemma_char_vocab, 'lem_char_vocab')
    write_vocab(ent_vocab, 'entity_vocab')
    write_vocab(ent_char_vocab, 'entity_char_vocab')
    write_vocab(rel_vocab, 'relation_vocab')
