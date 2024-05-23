# encoding=utf8
import re
import random
import networkx as nx

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

class AMRGraph(object):

    def __init__(self, smatch_amr):
        # transform amr from original smatch format into our own data structure
        instance_triple, attribute_triple, relation_triple = smatch_amr.get_triples()
        self.root = smatch_amr.root
        self.graph = nx.DiGraph()
        self.name2entity = dict()
        self.entity2name = dict()


        # will do some adjustments
        self.abstract_entities = dict()
        for _, name, entity in instance_triple:
            if is_attr_or_abs_form(entity):
                if _is_abs_form(entity):
                    self.abstract_entities[name] = entity
                else:
                    print ('bad entity', _, name, entity)
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
                print (rel, self.abstract_entities[entity], value, "abstract entity cannot have an attribute")
                continue
            name = "%s_attr_%d"%(value, len(self.name2entity))
            if not _is_attr_form(value):
                if _is_abs_form(value):
                    self.abstract_entities[name] = value
                else:
                    print ('bad attribute', rel, entity, value)
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
                    depths.append(depth+1)
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


    def collect_entities_and_relations(self):

        g = self.graph
        nodes, depths, is_connected = self.bfs()
        entities = [self.name2entity[n] for n in nodes]

        relations = dict()
        args = list()
        for i, src in enumerate(nodes):
            relations[i] = dict()
            for j, tgt in enumerate(nodes):
                relations[i][j] = list()
                for path in nx.all_shortest_paths(g, src, tgt):
                    info = dict()

                    info['node'] = path[1:-1]
                    info['edge'] = [g[path[i]][path[i+1]]['label'] for i in range(len(path)-1)]
                    info['length'] = len(info['edge'])
                    relations[i][j].append(info)
                    # ARG0, ARG1으로 연결된 모든 노드 찾기
                    for ban in ['ARG0', 'ARG1','ARG0_reverse_', 'ARG1_reverse_']:
                    # for ban in ['ARG0', 'ARG1','ARG0_reverse_', 'ARG1_reverse_']:
                        if ban in info['edge']:
                            edge_index = [index for index, value in enumerate(info['edge']) if value == ban]
                            # print('info_edge', info['edge'])
                            # print('edge_index', edge_index)
                            # print('path', path)

                            want = [path[value + 1] for value in edge_index]
                            want2 = [path[value] for value in edge_index]
                            # print(want)
                            # print(want2)

                            args.extend([self.name2entity[n] for n in want])
                            args.extend([self.name2entity[n] for n in want2])


        # ARG에서도 -d 연결된 동사들 다 빼기
        args = list(set(args))
        new_args = []
        for ar in args:
            if (bool(bool(re.search('[-\d]', ar)))) and ar not in ['amr-unknown', 'multi-sentence', 'have-org-role']:
                try:
                    ar = ar[:ar.index('-')]

                except ValueError:
                    pass
            else:
                pass
            new_args.append(ar)
        # print(new_args, 'new_args')

        # 모든 entity에서도 -d 연결된 동사들 빼주는데 그 이유는 entityNet에서 찾으려고
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

        return new_entities, depths, relations, is_connected, new_args
