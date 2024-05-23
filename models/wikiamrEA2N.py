import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import pickle
nltk.download('stopwords')

from encoder import RelationEncoder, TokenEncoder
from transformer import Transformer, SinusoidalPositionalEmbedding, SelfAttentionMask
from graph_transformer import GraphTransformer
from utils import *

import warnings
warnings.filterwarnings("ignore")

import numpy as np

from joblib import Parallel, delayed

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

np.random.seed(0)


class Reasoning_EA2N_DUAL(nn.Module):
    def __init__(self, vocabs,
                 entity_char_dim, entity_dim,
                 cnn_filters, char2entity_dim,
                 rel_dim, rnn_hidden_size, rnn_num_layers,
                 embed_dim, bert_embed_dim, ff_embed_dim, affective_dim, num_heads, dropout,
                 snt_layer,
                 graph_layers,
                 pretrained_file, device, batch_size,
                 model_type, bert_config, bert_model, bert_tokenizer, bert_max_length,
                 n_answers,
                 model, amr, affFeatures, affective_features):

        super(Reasoning_EA2N_DUAL, self).__init__()
        self.vocabs = vocabs
        self.embed_scale = math.sqrt(embed_dim)
        self.amr = amr
        self.affFeatures = affFeatures
        self.entity_encoder = TokenEncoder(vocabs['concept'], vocabs['entity_char'],
                                            entity_char_dim, entity_dim, embed_dim,
                                            cnn_filters, char2entity_dim, dropout, pretrained_file)
        self.relation_encoder = RelationEncoder(vocabs['relation'], rel_dim, embed_dim, rnn_hidden_size, rnn_num_layers,
                                                dropout)
        self.graph_encoder = GraphTransformer(graph_layers, embed_dim, ff_embed_dim, num_heads, dropout)
        self.transformer = Transformer(snt_layer, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)
        self.c_transformer = Transformer(snt_layer, embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)
        if self.amr and self.affFeatures:
            self.f_transformer = Transformer(snt_layer, bert_embed_dim+embed_dim+affective_dim, ff_embed_dim, num_heads, dropout, with_external=True)
            self.classifier = nn.Linear(bert_embed_dim+embed_dim+affective_dim, 2)
        elif self.amr and self.affFeatures == False:
            self.f_transformer = Transformer(snt_layer, bert_embed_dim+embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)
            self.classifier = nn.Linear(bert_embed_dim+embed_dim, 2)
        elif self.amr == False and self.affFeatures:
            self.f_transformer = Transformer(snt_layer, bert_embed_dim+affective_dim, ff_embed_dim, num_heads, dropout, with_external=True)
            self.classifier = nn.Linear(bert_embed_dim+affective_dim, 2)
        else:
            self.f_transformer = Transformer(snt_layer, bert_embed_dim, ff_embed_dim, num_heads, dropout, with_external=True)
            self.classifier = nn.Linear(bert_embed_dim, 2)

        self.pretrained_file = pretrained_file
        self.embed_dim = embed_dim
        self.entity_dim = entity_dim
        self.embed_scale = math.sqrt(embed_dim)

        self.token_position = SinusoidalPositionalEmbedding(embed_dim, device)
        self.entity_depth = nn.Embedding(32, embed_dim)
        self.token_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.entity_embed_layer_norm = nn.LayerNorm(embed_dim)
        self.self_attn_mask = SelfAttentionMask(device)
        self.dropout = dropout

        self.bert_config = bert_config
        self.bert_model = bert_model
        self.bert_tokenizer = bert_tokenizer
        self.bert_max_length = bert_max_length
        self.answer_len = n_answers
        self.device = device
        self.model_type = model_type
        self.model = model
        self.batch_size = batch_size
        self.affective_dim = affective_dim
        self.affective_features = affective_features

        self.loss_fct = CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.entity_depth.weight, 0.)

    def encoder_attn(self, inp):
        with torch.no_grad():
            entity_repr = self.embed_scale * self.entity_encoder(inp['concept'],
                                                                   inp['entity_char'] + self.entity_depth(
                                                                       inp['entity_depth']))
            entity_repr = self.entity_embed_layer_norm(entity_repr)
            entity_mask = torch.eq(inp['concept'], self.vocabs['concept'].padding_idx)

            relation = self.relation_encoder(inp['relation_bank'], inp['relation_length'])
            relation[0, :] = 0.
            relation = relation[inp['relation']]
            sum_relation = relation.sum(dim=3)
            num_valid_paths = inp['relation'].ne(0).sum(dim=3).clamp_(min=1)
            divisor = (num_valid_paths).unsqueeze(-1).type_as(sum_relation)
            relation = sum_relation / divisor

            attn, attn_weights = self.graph_encoder.get_attn_weights(entity_repr, relation, self_padding_mask=entity_mask)

        return attn

    def encode_wiki_step(self, inp, i, train=True):
        #print(inp['wiki_relation_bank'][i], inp['wiki_relation_length'][i])
        wiki_entity_input = inp['wiki_concept'][i][:,0].unsqueeze(1)
        wiki_entity_char_input = inp['wiki_entity_char'][i][:,0].unsqueeze(1)
        wiki_entity_depth_input = inp['wiki_entity_depth'][i][:,0].unsqueeze(1)
        wiki_relation_bank_input = inp['wiki_relation_bank'][i]
        wiki_relation_length_input = inp['wiki_relation_length'][i]
        wiki_relation_input = inp['wiki_relation'][i][:,:,0].unsqueeze(2)

        entity_repr = self.embed_scale * self.entity_encoder(wiki_entity_input,
                                                               wiki_entity_char_input) + self.entity_depth(
            wiki_entity_depth_input)
        entity_repr = self.entity_embed_layer_norm(entity_repr)
        entity_mask = torch.eq(wiki_entity_input, self.vocabs['concept'].padding_idx)
        relation = self.relation_encoder(wiki_relation_bank_input, wiki_relation_length_input)

        if str(train)=='True':
            relation = relation.index_select(0, wiki_relation_input.reshape(-1)).view(*wiki_relation_input.size(), -1)

        else:
            relation[0, :] = 0. # wiki_relation_length x dim
            relation = relation[wiki_relation_input]  # i x j x bsz x num x dim

            sum_relation = relation.sum(dim=3)  # i x j x bsz x dim
            num_valid_paths = wiki_relation_input.ne(0).sum(dim=3).clamp_(min=1)  # i x j x bsz

            divisor = (num_valid_paths).unsqueeze(-1).type_as(sum_relation) # i x j x bsz x 1
            relation = sum_relation / divisor # i x j x bsz dim
        entity_repr = self.graph_encoder(entity_repr, relation, self_padding_mask=entity_mask)

        return entity_repr

    def encoder_wiki_attn(self, inp, i):
        with torch.no_grad():
            wiki_entity_input = inp['wiki_concept'][i][:, 0].unsqueeze(1)
            wiki_entity_char_input = inp['wiki_entity_char'][i][:, 0].unsqueeze(1)
            wiki_entity_depth_input = inp['wiki_entity_depth'][i][:, 0].unsqueeze(1)

            wiki_relation_bank_input = inp['wiki_relation_bank'][i]
            wiki_relation_length_input = inp['wiki_relation_length'][i]
            wiki_relation_input = inp['wiki_relation'][i][:, :, 0].unsqueeze(2)

            entity_repr = self.embed_scale * self.entity_encoder(wiki_entity_input,
                                                                   wiki_entity_char_input) + self.entity_depth(
                wiki_entity_depth_input)
            entity_repr = self.entity_embed_layer_norm(entity_repr)
            entity_mask = torch.eq(wiki_entity_input, self.vocabs['concept'].padding_idx)

            relation = self.relation_encoder(wiki_relation_bank_input, wiki_relation_length_input) # [211, 512]

            relation[0, :] = 0.  # wiki_relation_length x dim
            relation = relation[wiki_relation_input]  # i x j x bsz x num x dim
            sum_relation = relation.sum(dim=3)  # i x j x bsz x dim
            num_valid_paths = wiki_relation_input.ne(0).sum(dim=3).clamp_(min=1)  # i x j x bsz

            divisor = (num_valid_paths).unsqueeze(-1).type_as(sum_relation)  # i x j x bsz x 1
            relation = sum_relation / divisor  # i x j x bsz x dim

            entity_repr = self.graph_encoder(entity_repr, relation, self_padding_mask=entity_mask)

            attn = self.graph_encoder.get_attn_weights(entity_repr, relation, self_padding_mask=entity_mask)
        return attn

    def convert_batch_to_bert_features(self,
                                       data,
                                       max_seq_length,
                                       tokenizer,
                                       cls_token_at_end=False,
                                       cls_token='[CLS]',
                                       cls_token_segment_id=1,
                                       sep_token='[SEP]',
                                       sequence_a_segment_id=0,
                                       sequence_b_segment_id=1,
                                       sep_token_extra=False,
                                       pad_token_segment_id=0,
                                       pad_on_left=False,
                                       pad_token=0,
                                       mask_padding_with_zero=True):
        features = []
        questions = [" ".join(x for x in sent) for sent in data['token_data']]
        choices_features = []
        stop_words = set(stopwords.words('english'))

        for i, text in enumerate(questions):
            question = ' '.join([word for word in text.split() if word.lower() not in stop_words])
            tokens = tokenizer.tokenize(question)
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if padding_length > 0:
                if pad_on_left:
                    input_ids = ([pad_token] * padding_length) + input_ids
                    input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                else:
                    input_ids = input_ids + ([pad_token] * padding_length)
                    input_mask = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            else:
                input_ids = input_ids[:max_seq_length]
                input_mask = input_ids[:max_seq_length]

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            segment_ids = []
            choices_features.append((input_ids, input_mask, segment_ids))

            features.append(choices_features)
            choices_features = []
        return features


    def prepare_bert_input(self, data, tokenizer):
        move_to_cuda(data, self.device)

        features = self.convert_batch_to_bert_features(data = data,
                                                  max_seq_length=self.bert_max_length,
                                                  tokenizer=tokenizer,
                                                  )

        move_to_cuda(features, self.device)

        all_input_ids = torch.tensor([f[0] for feature in features for f in feature], dtype=torch.long).view(self.batch_size, self.bert_max_length).to(self.device)
        all_input_mask = torch.tensor([f[1] for feature in features for f in feature], dtype=torch.long).view(self.batch_size,  self.bert_max_length).to(self.device)
        all_segment_ids = []
        return all_input_ids, all_input_mask, all_segment_ids


    def prepare_graph_state(self, graph_state, ans_len, entity_dim):
        tot_initial = torch.tensor(1).to(self.device)

        j = 0
        while j < (1*ans_len)-1:
            initial = graph_state[0][j].view(1, -1).to(self.device)
            for i in graph_state[1:]:  # i = [5 x 512] x 7
                com_tensor = i[j + 1].view(1, -1).to(self.device)

                initial = torch.cat([initial, com_tensor], dim=0)
            if j == 0:
                tot_initial = initial.view(1, -1, entity_dim)
            j += 1
            initial = initial.view(1, -1, entity_dim)
            tot_initial = torch.cat([tot_initial, initial], dim=0)
        return tot_initial


    def prepare_affective_features(self, data):
        feature_dict = self.affective_features
        ids = data['id']
        features = []
        for i, id in enumerate(ids):
            if id in feature_dict.keys():
                features.append(feature_dict[id])
            else:
                features.append(np.zeros(240))
        features = np.array(features).reshape(self.batch_size, self.affective_dim)
        features = torch.from_numpy(features).float().to(self.device)
        return features

    def forward(self, data, train):
        answer_len = self.answer_len
        
        if self.amr:
            tot_entity_reprs = []
            for i in range(self.batch_size):
                ## AMR-GTOS
                entity_repr = self.encode_wiki_step(data, i, train=train)  # entity_seq_len x 1 x entity_embed_size

                entity_repr = self.transformer(entity_repr, kv=None)  # res = entity_seq_len x bsz x entity_embed_size


                if entity_repr.size()[1] == 1:
                    entity_repr = entity_repr.squeeze().unsqueeze(0).mean(1).unsqueeze(1)

                else:
                    entity_repr = self.prepare_graph_state(entity_repr, entity_repr.size()[1], self.embed_dim).mean(
                        1).unsqueeze(1)  # re = bsz x 1 x entity_embed_size

                tot_entity_repr = self.c_transformer(entity_repr, kv=None)

                tot_entity_repr = tot_entity_repr.squeeze(1)

                tot_entity_reprs.append(tot_entity_repr)

            tot_entity_reprs = torch.squeeze(torch.stack(tot_entity_reprs), 1)

        if self.affFeatures:
            aff_features = self.prepare_affective_features(data)

        ids = data['id']
        labels = data['target']
        labels = torch.tensor(labels).to(self.device)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)

        ## prepare bert input
        all_input_ids, all_input_mask, all_segment_ids = self.prepare_bert_input(data,
                                                                                 self.bert_tokenizer)
        bert_encoded_features = self.bert_model(
            input_ids=all_input_ids,
            attention_mask=all_input_mask
        )
        bert_final_features = bert_encoded_features[0][:,0,:]
        bsz = len(ids)

        if self.amr and self.affFeatures:
            final_logits = torch.cat([bert_final_features, tot_entity_reprs, aff_features], 1)
        elif self.amr and self.affFeatures == False:
            final_logits = torch.cat([bert_final_features, tot_entity_reprs], 1)
        elif self.amr == False and self.affFeatures:
            final_logits = torch.cat([bert_final_features, aff_features], 1)
        else:
            final_logits = bert_final_features

        final_logits = self.f_transformer(final_logits.unsqueeze(1), kv=None)
        final_logits = self.classifier(final_logits).view(bsz, -1)

        return final_logits, labels, ids