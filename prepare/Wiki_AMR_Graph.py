# encoding=utf8
import re
import random
import networkx as nx
from typing import List


number_regexp = re.compile(r'^-?(\d)+(\.\d+)?$')
abstract_regexp0 = re.compile(r'^([A-Z]+_)+\d+$')
abstract_regexp1 = re.compile(r'^\d0*$')
discard_regexp = re.compile(r'^n(\d+)?$')

attr_value_set = set(['-', '+', 'interrogative', 'imperative', 'expressive'])


def _is_attr_form(x):
    return (x in attr_value_set or x.endswith('_') or number_regexp.match(x) is not None)


def _is_abs_form(x):
    return (abstract_regexp0.match(x) is not None or abstract_regexp1.match(x) is not None)


def is_attr_or_abs_form(x):
    return _is_attr_form(x) or _is_abs_form(x)


def need_an_instance(x):
    return (not _is_attr_form(x) or (abstract_regexp0.match(x) is not None))


class WikiAMRGraph(object):

    def __init__(self, smatch_amr, seed_word:List, graph:List):
        # transform amr from original smatch format into our own data structure
        instance_triple, attribute_triple, relation_triple = smatch_amr.get_triples()
        self.root = smatch_amr.root
        self.seed_word = seed_word
        self.graph = nx.DiGraph()
        self.name2entity = dict()
        self.entity2name = dict()
        self.entitynet_graph = graph

        # will do some adjustments
        self.abstract_entities = dict()
        for _, name, entity in instance_triple:

            if is_attr_or_abs_form(entity):
                if _is_abs_form(entity):
                    self.abstract_entities[name] = entity
                else:
                    print('bad entity', _, name, entity)
            self.name2entity[name] = entity
            self.graph.add_node(name)
        for rel, entity, value in attribute_triple:
            if rel == 'TOP':
                continue
            # discard some empty names
            if rel == 'name' and discard_regexp.match(value):
                continue
            # abstract entity can't have an attribute
            if entity in self.abstract_entities:
                print(rel, self.abstract_entities[entity], value, "abstract entity cannot have an attribute")
                continue
            name = "%s_attr_%d" % (value, len(self.name2entity))
            if not _is_attr_form(value):
                if _is_abs_form(value):
                    self.abstract_entities[name] = value
                else:
                    print('bad attribute', rel, entity, value)
                    continue
            self.name2entity[name] = value
            self._add_edge(rel, entity, name)


        for rel, head, tail in relation_triple:
            self._add_edge(rel, head, tail)

        # lower entity
        for name in self.name2entity:
            v = self.name2entity[name]
            if not _is_abs_form(v):
                v = v.lower()
            v = v.rstrip('_')
            self.name2entity[name] = v


    def __len__(self):
        return len(self.name2entity)

    def _add_edge(self, rel, src, des):
        self.graph.add_node(src)
        self.graph.add_node(des)
        self.graph.add_edge(src, des, label=rel)
        self.graph.add_edge(des, src, label=rel + '_reverse_')

    def collect_entities_and_relations(self):
        # Make entity2name

        for name in self.name2entity:
            value = self.name2entity[name]
            try:
                del self.name2entity[name]
            except KeyError:
                pass

            if (bool(bool(re.search('[-\d]', value)))) and value not in ['amr-unknown', 'multi-sentence',
                                                                        'have-org-role']:
                try:
                    value = value[:value.index('-')]
                except ValueError:
                    pass
            else:
                pass

            self.entity2name[value] = name
            self.name2entity[name]= value

        for entitynet in self.entitynet_graph:
            if entitynet[0] in self.entity2name:
                src = self.entity2name[entitynet[0]]
            else:
                src = entitynet[0]
                self.name2entity[src] = src
                self.entity2name[src] = src

            if entitynet[2] in self.entity2name:
                des = self.entity2name[entitynet[2]]
            else:
                des = entitynet[2]
                self.name2entity[des] = des
                self.entity2name[des] = des

            self._add_edge(entitynet[1], src, des)

        nodes, depths, is_connected = self.local_bfs(self.graph)
        g = self.graph

        entities = [self.name2entity[n] for n in nodes]
        relations = dict()

        for i, src in enumerate(nodes):
            relations[i] = dict()
            for j, tgt in enumerate(nodes):
                relations[i][j] = list()
                for path in nx.all_shortest_paths(g, src, tgt):

                    info = dict()
                    info['node'] = path[1:-1]
                    info['edge'] = [g[path[i]][path[i + 1]]['label'] for i in range(len(path) - 1)]
                    info['length'] = len(info['edge'])
                    relations[i][j].append(info)
        new_entities = []
        for con in entities:
            if (bool(bool(re.search('[-\d]', con)))) and con not in ['amr-unknown', 'multi-sentence', 'have-org-role']:
                try:
                    con = con[:con.index('-')]

                except ValueError:
                    pass
            else:
                pass
            new_entities.append(con)

        return new_entities, depths, relations, is_connected, g
    
    def collect_wikinet_relations(self):
        # Make entity2name
        for name in self.name2entity:
            value = self.name2entity[name]
            try:
                del self.name2entity[name]
            except KeyError:
                pass

            if (bool(bool(re.search('[-\d]', value)))) and value not in ['amr-unknown', 'multi-sentence']:
                try:
                    value = value[:value.index('-')]
                except ValueError:
                    pass
            else:
                pass

            self.entity2name[value] = name
            self.name2entity[name]= value

        # Adding paths between entities
        print('----------------------------------------------------------\nCreating nodes...\n----------------------------------------------------------')
        
        if self.entitynet_graph is not None:
            for src_entity in self.entitynet_graph:
                for des_entity in self.entitynet_graph[src_entity]:
                    paths = self.entitynet_graph[src_entity][des_entity]
                    # self.entitynet_graph[des_entity][src_entity] = None
                    if paths == None:
                        continue
                    for i in range(len(paths)):
                        entity = paths[i][0]
                        if (i==0):
                            entity = src_entity
                        relation = paths[i][1]
                        if entity in self.entity2name:
                            name = self.entity2name[entity]
                        else:
                            name = entity
                            self.name2entity[name] = entity
                            self.entity2name[entity] = name
                        if i < len(paths) - 1:
                            next_entity = paths[i+1][0]
                            if next_entity in self.entity2name:
                                next_name = self.entity2name[next_entity]
                            else:
                                next_name = next_entity
                                self.name2entity[next_name] = next_entity
                                self.entity2name[next_entity] = next_name
                            self._add_edge(relation, name, next_name)
                        else:
                            next_entity = des_entity
                            if next_entity in self.entity2name:
                                next_name = self.entity2name[next_entity]
                            else:
                                next_name = next_entity
                                self.name2entity[next_name] = next_entity
                                self.entity2name[next_entity] = next_name
                            self._add_edge(relation, name, next_name)
                        print(f'-x- path {i} [{name}] -- [{relation}] -- [{next_name}]')
                        if (next_entity == des_entity):
                            print(f'-x- path between [{src_entity}] and [{des_entity}], Path Length: {len(paths)} Created...')
        print('----------------------------------------------------------\nFinished adding nodes...')

        nodes, depths, is_connected = self.local_bfs(self.graph)
        g = self.graph
        
        # print(f'----------------------------------------------------------\nentity2name {self.entity2name}')
        # print(f'name2entity {self.name2entity}')
        # print(f'----------------------------------------------------------')

        print('\nGraph nodes', g.nodes, '\n')
        # print('\ngraph edges', g.edges)

        entities = [self.name2entity[n] for n in nodes]
        relations = dict()

        for i, src in enumerate(nodes):
            relations[i] = dict()
            for j, tgt in enumerate(nodes):
                relations[i][j] = list()
                for path in nx.all_shortest_paths(g, src, tgt):
                    
                    info = dict()
                    info['node'] = path[1:-1]
                    info['edge'] = [g[path[i]][path[i + 1]]['label'] for i in range(len(path) - 1)]
                    info['length'] = len(info['edge'])
                    relations[i][j].append(info)

        new_entities = []
        for con in entities:
            if (bool(bool(re.search('[-\d]', con)))) and con not in ['amr-unknown', 'multi-sentence']:
                try:
                    con = con[:con.index('-')]

                except ValueError:
                    pass
            else:
                pass
            new_entities.append(con)

        return new_entities, depths, relations, is_connected, g

    def bfs(self):
        g = self.graph
        queue = [self.root]
        depths = [0]
        visited = set(queue)
        step = 0
        while step < len(queue):
            u = queue[step]
            depth = depths[step]
            step += 1
            for v in g.neighbors(u):
                if v not in visited:
                    queue.append(v)
                    depths.append(depth + 1)
                    visited.add(v)
        is_connected = (len(queue) == g.number_of_nodes())
        return queue, depths, is_connected

    def local_bfs(self, g):
        queue = [self.root]
        depths = [0]
        visited = set(queue)
        step = 0
        while step < len(queue):
            u = queue[step]
            depth = depths[step]
            step += 1
            for v in g.neighbors(u):
                if v not in visited:
                    queue.append(v)
                    depths.append(depth+1)
                    visited.add(v)
        is_connected = (len(queue) == g.number_of_nodes())
        return queue, depths, is_connected
    #
    # def make_adjacency(self):
    #     # Make entity2name
    #     for name in self.name2entity:
    #         self.entity2name[self.name2entity[name]] = name
    #
    #     # Let's mix it
    #     for i in self.seed_word:
    #         if i in self.name2entity:
    #             pass
    #         else:
    #             self.name2entity[i] = i
    #
    #     # Let's matching value
    #     for entitynet in self.entitynet_graph:
    #         if entitynet[0] in self.entity2name:
    #             src = self.entity2name[entitynet[0]]
    #         else:
    #             src = entitynet[0]
    #             self.name2entity[src] = src
    #             self.entity2name[src] = src
    #
    #         if entitynet[2] in self.entity2name:
    #             des = self.entity2name[entitynet[2]]
    #         else:
    #             des = entitynet[2]
    #             self.name2entity[des] = des
    #             self.entity2name[des] = des
    #
    #         self._add_edge(entitynet[1], src, des)
    #
    #     g = self.graph
    #     A = nx.adjacency_matrix(g)
    #     print(A.to_dense())
    #
    #     return A
