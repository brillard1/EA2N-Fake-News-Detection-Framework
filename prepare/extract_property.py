#!/usr/bin/env python
#coding: utf-8
import itertools
import json
from prepare.smatch import AMR
from prepare.AMRGraph import AMRGraph
from prepare.AMRGraph import  _is_abs_form
from prepare.Wiki_AMR_Graph import WikiAMRGraph
from collections import Counter
from multiprocessing import Pool
from prepare.pathFinder import WikiTag
from tqdm.auto import tqdm

class LexicalMap(object):

    def __init__(self):
        pass

    def get(self, entity, vocab=None):
        cp_seq = []
        for conc in entity:
            cp_seq.append(conc)

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
                elif line.startswith('# ::snt '):
                    sentence = line[len('# ::snt '):]
                    graph_line = AMR.get_amr_line(f)
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
    for x, y in cnt.most_common():
        for ch in list(x):
            char_cnt[ch] += y
    return cnt, char_cnt


def write_vocab(vocab, path):
    with open(path, 'w') as fo:
        for x, y in vocab.most_common():
            fo.write('%s\t%d\n'%(x,y))



import argparse
def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--amr_files', type=str, nargs='+')
    parser.add_argument('--nprocessors', type=int, default=4)
    parser.add_argument('--entity_seed', type=str)
    parser.add_argument('--tagme_tokenID', type=int)

    return parser.parse_args()

import ast
if __name__ == "__main__":
    # collect entities and relations
    args = parse_config()
    cn = dict()
    lexical_map = LexicalMap()

    wikiNetP = WikiTag([], 1, 0, 0)
    entity_dict, id_dict, relation_dict = wikiNetP.read_entities_relations()
    wiki_entity = wikiNetP.read_entities()

    def work(data):
        amr, lem, tok = data
        entity, depth, relation, ok, ARGs = amr.collect_entities_and_relations()
        assert ok, "not connected"

        return entity, depth, relation, ARGs

    pool = Pool(args.nprocessors)

    for file in args.amr_files:
        cnt = 0
        with open(
                 file[0:file.index('.txt')] + '_wiki_extended_final.json',
                 'w', encoding='utf-8', ) as json_result:
            json_result.write('[')
            my_data = []
            amrs, token, lemma, abstract, targets, options, amr_ids, sentences, raw_amrs = read_file(file)
            res = pool.map(work, zip(amrs, lemma, token), len(amrs) // args.nprocessors)

            file_tag = file[file.index('.txt') - 9 : file.index('.txt')]
            if (file_tag[-2] == '_'):
                if (file_tag[:2] == 'in'):
                    file_tag = 'train_' + file_tag[-1] + '.txt'
                elif (file_tag[:2] == 'st'):
                    file_tag = 'test_' + file_tag[-1] + '.txt'
                elif (file_tag[:2] == 'ev'):
                    file_tag = 'dev_' + file_tag[-1] + '.txt'
            elif (file_tag[-2] != '_'):
                if (file_tag[:1] == 'n'):
                    file_tag = 'train_' + file_tag[-2:] + '.txt'
                elif (file_tag[:1] == 't'):
                    file_tag = 'test_' + file_tag[-2:] + '.txt'
                elif (file_tag[:1] == 'v'):
                    file_tag = 'dev_' + file_tag[-2:] + '.txt'

            with open('wikinet/vocab/'+file_tag, 'w') as f:
                f.write('')
            f.close()

            for gr, to, le, ab, target, option, amr_id, sent, raw_amr in tqdm(zip(res, token, lemma, abstract,
                                                                                 targets,
                                                                                 options, amr_ids, sentences, raw_amrs), total = len(sentences)):

                entity, depth, relation, ARGs = gr

                cnt += 1
                wiki_entities = []
                wiki_depths = []
                wiki_relations = []
                print(f'\nAMR: {cnt} ID: {amr_id}')
                entities = list(set(entity)-set(ARGs))
                if args.entity_seed == 'WikiAMR':
                    wiki_seed = list(set([con for con in ARGs if con in cn]))
                    
                    wikiNet = WikiTag(entities, hops = 5, entity_threshold = 0.2, path_threshold = 0.1, token_index=args.tagme_tokenID, file_tag = file_tag)
                    wiki_seed, entity_graph = wikiNet.output_graph(entity_dict, id_dict, relation_dict, wiki_entity)
                    print(f'\nFinal entity graph: {entity_graph} \n')
                    rel_graph = WikiAMRGraph(raw_amr, entities, entity_graph)
                    _wiki_entities, _wiki_depths, _wiki_relations, _is_connected, g = rel_graph.collect_wikinet_relations()

                else:
                    wiki_seed = []
                    _wiki_entities, _wiki_depths, _wiki_relations, _is_connected, g = [], [], [], [], []

                wiki_entities.append(_wiki_entities)
                wiki_depths.append(_wiki_depths)
                wiki_relations.append(_wiki_relations)

                wiki_entities2 = [wiki_entities[0] for i in range(5)]
                wiki_depths2 = [wiki_depths[0] for i in range(5)]
                wiki_relations2 = [wiki_relations[0] for i in range(5)]

                item = {
                    'entity': entity,
                    'depth': depth,
                    'relation': relation,
                    'wiki_entity': wiki_entities2,
                    'wiki_depth': wiki_depths2,
                    'wiki_relation': wiki_relations2,
                    'token': to,
                    'lemma': le,
                    'abstract': ab,
                    'target': target,
                    'option': option,
                    'id': amr_id,
                    'sentences': sent}

                json.dump(item, json_result)
                json_result.write(',')

            json_result.write(']')
