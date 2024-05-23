# Importing tagme and heapq modules
import tagme
import heapq
from tqdm.auto import tqdm

# Defining a class for the concept graph
class WikiTag:
    def __init__(self, cn_seed, hops, entity_threshold, path_threshold, token_index=0, file_tag = 'wikivocab'):
        self.cn_seed = cn_seed
        self.n = hops
        self.entity_threshold = entity_threshold
        self.path_threshold = path_threshold
        self.file_tag = file_tag
        self.concept_graph = {}
        self.entity_dict = {}
        self.id_dict = {}
        self.relation_dict = {}
        self.cn = {}

        tagme_tokens = [
        '5fa6833f-da41-4c7a-8a9f-e6b883be897c-843339462',
        '9f0fefe0-39b0-4338-97ee-29ce50ef2c40-843339462',
        '81b9d4b5-e41a-4dcc-b269-a93a7b96bead-843339462',
        '6c2c5994-6225-4fd0-ac15-2bfc1e2a63e8-843339462',
        'f26b872f-1ee8-4c66-9f39-ffadb627d712-843339462',
        '9265d91c-cb33-4db8-b273-b1e19d0d15dc-843339462', 
        '06b19cd0-8947-4387-b960-8747b62a94eb-843339462',
        'd6b04e41-16f4-4a8c-be25-d0008c6a915a-843339462',
        '8ca9331f-0c2d-4592-ae50-107ed44bff54-843339462',
        'b1b942a1-a251-4bb1-b16a-590f1c33b05e-843339462']

        assert token_index < len(tagme_tokens)

        # Setting the tagme token
        tagme.GCUBE_TOKEN = tagme_tokens[token_index]

    def annotate(self):
        self.entities = []
        self.amr = []
        self.amr_entities = {}
        concept = ""
        for con in self.cn_seed:
            concept += f", {con}"

        annotations = tagme.annotate(concept)

        if (annotations != None):
            for ann in annotations.get_annotations(0.1):
                if ann.entity_title in self.entity_dict and (ann.entity_title not in self.entities \
                                        and ann.mention.split(',')[0] not in self.amr):
                    self.entities.append(ann.entity_title)
                    amr_entity = ann.mention.split(',')[0]
                    self.amr.append(amr_entity)
        # self.entities = list(set(self.entities))
        # self.amr = list(set(self.amr))

        assert len(self.entities) == len(self.amr)

        for i in range(len(self.amr)):
            en = self.entities[i]
            am = self.amr[i]
            self.amr_entities[en] = am
        print('AMR Entities:', self.amr_entities)

    def find_path(self, start, goal):
        # Initial check
        stName, goName = self.id_dict[start][0].lower(), self.id_dict[goal][0].lower()
        relation_path = []
        frontier = []
        nb = []
        heapq.heappush(frontier, (0, 0, [start]))
        explored = set()
        i = 0
        while frontier:
            i += 1
            cost, hops, path = heapq.heappop(frontier)
            current = path[-1]
            print(f'-x- idx: {i} current node: [{self.id_dict[current][0]}] len path: {len(path)}  hops: {hops} out of {self.n}...')

            #if (hops == self.n and current != goal):
            #    return None

            if (current == goal or hops == self.n) and len(path)>1:
                for idx in range(len(path)-1):
                    node_id = path[idx]
                    next_id = path[idx+1]
                    _node = self.id_dict[node_id][0]
                    _next = self.id_dict[next_id][0]
                    if (self.cn[node_id][next_id] in self.relation_dict):
                        rel = self.relation_dict[self.cn[node_id][next_id]]
                    else:
                        rel = 'Related To'
                    relation_path.append([_node, rel, _next])
                print(f'-x- linking [{stName}] with [{goName}] with path {relation_path}')
                print(f'-x- neighbours found: {str(nb)}')
                return relation_path

            explored.add(current)
            # Neighbors are restricted to 5
            adj_neighbor_cnt = 0
            for neighbor in self.cn[current]:

                if (neighbor not in self.id_dict):
                    continue

                adj_neighbor_cnt += 1
                if (adj_neighbor_cnt > 5):
                    break

                node, relation = neighbor, self.cn[current][neighbor]
                if node not in explored:
                    # Convert ids to tagme annotations
                    currName, nodeName = self.id_dict[current][0].lower(), self.id_dict[node][0].lower()
                    optimum = tagme.annotate(goName + ',' + nodeName + ',' + stName)
                    if (optimum == None):
                        continue

                    idx = -1
                    nodeTag = nodeName
                    goTag = goName
                    for ann in optimum.get_annotations(0):
                        idx += 1
                        if (idx == 0):
                            goTag = ann.entity_title
                        elif (idx == 1):
                            nodeTag = ann.entity_title
                        elif (idx == 2):
                            stTag = ann.entity_title

                    rels = tagme.relatedness_title([(stTag, nodeTag)])
                    relg = tagme.relatedness_title([(nodeTag, goTag)]) 
                    if (rels == None or relg == None):
                        continue

                    score1 = rels.relatedness[0].rel # basic cost
                    score2 = relg.relatedness[0].rel # heuristic cost
                    if score1 != None and score2 != None and score1 + score2 >= self.path_threshold:
                        new_cost = cost + self.get_weight(relation)
                        new_path = path + [node]
                        heapq.heappush(frontier, (new_cost, hops+1, new_path))
                        nb.append(adj_neighbor_cnt)
                        break

        return None

    def get_weight(self, relation):
        weights = {
            "memberOf": 1,
            "unsubclassableExampleOf": 2,
            "subClassOf": 3,
            "instanceOf": 4,
            "relatedTo": 5,
            "synonymOf": 6,
            "antonymOf": 7,
            "partOf": 8,
            "hasPart": 9,
            "causes": 10,
            "causedBy": 11,
            "entails": 12,
            "entailedBy": 13,
            "similarTo": 14,
            "differentFrom": 15,
        }
        return weights.get(relation, 100)

    def read_entities_relations(self):
        with open("../wikidata5m/wikidata5m_entity.txt", "r") as f:
            for line in tqdm(f.readlines()):
                parts = line.strip().split('\t')
                e_id, e_names = parts[0], parts[1:]
                for e_name in e_names:
                    self.entity_dict[e_name] = e_id
                    self.id_dict.setdefault(e_id, []).append(e_name)
        print(f'Entities read...')
        with open("../wikidata5m/wikidata5m_relation.txt", "r") as f:
            for line in tqdm(f.readlines()):
                parts = line.strip().split('\t')
                r_id, r_name = parts[0], parts[1]
                self.relation_dict[r_id] = r_name
        print('Relations read...')
        return self.entity_dict, self.id_dict, self.relation_dict

    def read_concepts(self):
        for file in ["../wikidata5m/wikidata5m_transductive_train.txt", "../wikidata5m/wikidata5m_transductive_valid.txt", "../wikidata5m/wikidata5m_transductive_test.txt"]:
            with open(file, "r") as f:
                for line in tqdm(f.readlines()):
                    parts = line.strip().split('\t')
                    e1, rel, e2 = parts[0], parts[1], parts[2]
                    if e1 in self.cn:
                        self.cn[e1][e2] = rel
                    else:
                        self.cn[e1] = {e2: rel}
                    if e2 in self.cn:
                        self.cn[e2][e1] = rel
                    else:
                        self.cn[e2] = {e1: rel}
        print('Paths read...')
        return self.cn

    def build_graph(self):
        """
            Pipeline for entity level filtering
        """
        rel_id = []
        rel_score = []
        idx = -1
        for i in range(len(self.entities)):
            for j in range(i+1, len(self.entities)):
                entity_1 = self.entities[i]
                entity_2 = self.entities[j]
                id_1 = self.entity_dict[entity_1]
                id_2 = self.entity_dict[entity_2]
                rel_id.append(tuple((entity_1, entity_2)))

        rels = tagme.relatedness_title(rel_id)
        for rel in rels.relatedness:
            rel_score.append(rel.rel)
        el = 0
        pl=0
        for i in range(len(self.entities)):
            for j in range(i+1, len(self.entities)):
                idx+=1
                entity_1 = self.entities[i]
                entity_2 = self.entities[j]
                
                if (entity_1 in self.entity_dict and entity_2 in self.entity_dict):
                    id_1 = self.entity_dict[entity_1]
                    id_2 = self.entity_dict[entity_2]
                else:
                    self.concept_graph.setdefault(self.amr_entities[entity_1], {})[self.amr_entities[entity_2]] = None
                    continue

                if id_1 in self.cn and id_2 in self.cn[id_1]:
                    el+=1
                    pl+=2
                    print(f'-x- adding direct relation [{self.relation_dict[self.cn[id_1][id_2]]}] bewteen [{entity_1}] and [{entity_2}]')
                    direct_rel = self.relation_dict[self.cn[id_1][id_2]]
                    self.concept_graph.setdefault(self.amr_entities[entity_1], {})[self.amr_entities[entity_2]] = [[self.amr_entities[entity_1], direct_rel, self.amr_entities[entity_2]]]
                    # self.concept_graph.setdefault(self.amr_entities[entity_2], {})[self.amr_entities[entity_1]] = [[self.amr_entities[entity_2], direct_rel, self.amr_entities[entity_1]]]
                elif id_1 in self.cn:
                    if rel_score[idx] >= self.entity_threshold:
                        el+=1
                        print(f'-x- adding indirect relation between [{entity_1}] and [{entity_2}]')
                        path = self.find_path(id_1, id_2)
                        if path:
                            pl+=len(path)+1
                            path[0][0] = self.amr_entities[entity_1]
                            self.concept_graph.setdefault(self.amr_entities[entity_1], {})[self.amr_entities[entity_2]] = path
                            # self.concept_graph.setdefault(self.amr_entities[entity_2], {})[self.amr_entities[entity_1]] = path[::-1]
                        else:
                            self.concept_graph.setdefault(self.amr_entities[entity_1], {})[self.amr_entities[entity_2]] = None
                            # self.concept_graph.setdefault(self.amr_entities[entity_2], {})[self.amr_entities[entity_1]] = None
                    else:
                        print(f'-x- no significant relation between [{entity_1}] and [{entity_2}], with Score: {rel_score[idx]}')
                        continue

        print(f'entities with ELF threshold: {el}\nentities with CLF threshold: {pl}')

    def output_graph(self, entity_dict, id_dict, relation_dict, wiki_concept):
        self.entity_dict = entity_dict
        self.id_dict = id_dict
        self.relation_dict = relation_dict
        self.cn = wiki_concept

        self.annotate()

        if (len(self.entities) < 2):
            return None, None

        self.build_graph()
        save_dir = 'wikinet/vocab/'
        with open(save_dir+str(self.file_tag), 'a+') as f:
        #print("----------------------------------------------")
        #print("vocab")
            for e1 in self.concept_graph:
                for e2 in self.concept_graph[e1]:
                    paths = self.concept_graph[e1][e2]
                    if paths != None: 
                        for path in paths:
                            #print(str(e1) + '\t' + str(path) + '\n')
                            f.write(str(e1) + '\t' + str(path) + '\n')
        #print("----------------------------------------------")
        f.close()
        return self.entities, self.concept_graph
