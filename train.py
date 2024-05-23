import time
import math
import copy
import csv
import collections
import torch.distributed as dist
import torch.nn as nn
import warnings
import torch
import logging
import random
import importlib
import numpy as np
import re
import pickle
import transformers
import json
import tensorflow as tf
from tqdm.auto import tqdm, trange
from sklearn.model_selection import KFold
from torch.utils.data import random_split, SubsetRandomSampler, ConcatDataset
from torch.utils.data import Subset, DataLoader as DL
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from optimization import AdamW, WarmupCosineSchedule #, WarmupLinearSchedule
from data import Vocab, DataLoader, STR, END, CLS, SEL, TL, rCLS
from prepare.extract_property import LexicalMap
from utils import move_to_cuda, EarlyStopping

from transformers.models.bert.modeling_bert import BertModel
from transformers.models.electra.modeling_electra import ElectraModel
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.xlnet.modeling_xlnet import XLNetModel
from models.wikiamrEA2N import Reasoning_EA2N_DUAL
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from transformers import (BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer, ElectraConfig, ElectraTokenizer, XLNetConfig, XLNetTokenizer)

from tensorboardX import SummaryWriter

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

warnings.filterwarnings(action='ignore')

MODEL_CLASSES = {
    'EA2N_bert_dual':(BertConfig, BertModel, BertTokenizer),
    'EA2N_electra_dual':(ElectraConfig, ElectraModel, ElectraTokenizer),
    'EA2N_xlnet_dual':(XLNetConfig, XLNetModel, XLNetTokenizer),
    'EA2N_roberta_dual':(RobertaConfig, RobertaModel, RobertaTokenizer)
    }

def set_seed(seed, n_gpu):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if int(n_gpu[0]) > 0:
        torch.cuda.manual_seed_all(seed)

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def read_jsonl(input_file):
    with tf.gfile.Open(input_file, 'r') as f:

        return [json.loads(ln) for ln in f]

class EA2NModel(object):
    def __init__(self, args, local_rank):
        super(EA2NModel, self).__init__()
        self.save_args = args
        self.dataset = args['dataset']
        self.amr = args['amr']
        self.affFeatures = args['affFeatures']
        args['cnn_filters'] = list(zip(args['cnn_filters'][:-1:2], args['cnn_filters'][1::2]))
        args = collections.namedtuple("HParams", sorted(args.keys()))(**args)
        if not os.path.exists(args.ckpt):
            os.mkdir(args.ckpt)
        self.args = args
        self.local_rank = local_rank

    def _build_model(self):
        self.device = torch.device("cuda:"+str(self.args.gpus[0]) if torch.cuda.is_available() else "cpu")

        print("Cuda Version:",torch.cuda.get_device_name(0))
        print(self.device, 'here')
        vocabs, lexical_mappings = [], []
        config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.encoder_type]
        self.bert_config = config_class.from_pretrained(
            self.args.lm_model,
        )
        self.bert_tokenizer = tokenizer_class.from_pretrained(
            self.args.lm_model
        )
        with open(self.args.affective_features, 'rb') as f:
            affective_features = pickle.load(f)

        if self.args.bert_pretrained_file == None:

            self.bert_model = model_class.from_pretrained(
                self.args.lm_model,
                config=self.args.lm_model
            ).to(self.device)

        else:
            self.bert_model = model_class.from_pretrained(
                self.args.bert_pretrained_file,
            ).to(self.device)
            print(self.args.encoder_type, 'pretrained')

        if self.args.encoder_type in ['EA2N_bert_dual', 'EA2N_electra_dual', 'EA2N_xlnet_dual', 'EA2N_roberta_dual']:
            vocabs, lexical_mapping = self._prepare_data()
            self.model = Reasoning_EA2N_DUAL(vocabs,
                                               self.args.concept_char_dim, self.args.concept_dim,
                                               self.args.cnn_filters, self.args.char2concept_dim,
                                               self.args.rel_dim, self.args.rnn_hidden_size, self.args.rnn_num_layers,
                                               self.args.embed_dim, self.args.bert_embed_dim, self.args.ff_embed_dim,
                                               self.args.affective_dim, self.args.num_heads,
                                               self.args.dropout,
                                               self.args.snt_layer,
                                               self.args.graph_layers,
                                               self.args.pretrained_file, self.device, self.args.batch_size,
                                               self.args.lm_model, self.bert_config, self.bert_model, self.bert_tokenizer,
                                               self.args.bert_max_length,
                                               self.args.n_answers,
                                               self.args.encoder_type,
                                               self.amr,
                                               self.affFeatures,
                                               affective_features
                            )

        else:
            pass


        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        return vocabs, lexical_mapping

    def _update_lr(self, optimizer, embed_size, steps, warmup_steps, t_total):
        for param_group in optimizer.param_groups:
            param_group['lr'] = embed_size ** -0.5 * min(steps ** -0.5, steps * (t_total*warmup_steps ** -1.5))

    def _average_gradients(self, model):
        size = float(dist.get_world_size())
        for param in model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= size

    def _prepare_data(self):

        vocabs = dict()
        vocabs['concept'] = Vocab(self.args.concept_vocab, 5, [CLS])
        vocabs['token'] = Vocab(self.args.token_vocab, 5, [STR, END])
        vocabs['token_char'] = Vocab(self.args.token_char_vocab, 100, [STR, END])
        vocabs['concept_char'] = Vocab(self.args.concept_char_vocab, 100, [STR, END])
        vocabs['relation'] = Vocab(self.args.relation_vocab, 5, [CLS, rCLS, SEL, TL])
        lexical_mapping = LexicalMap()

        return vocabs, lexical_mapping
    
    def _file_manage(self):
        # Selecting the encoder type
        # Write Train, Dev, Test Results according to if conditions in self.args.encoder_type
        if (self.args.encoder_type=='EA2N_bert_dual'):
            file1 = 'amr_bert'
        elif (self.args.encoder_type=='EA2N_roberta_dual'):
            file1 = 'amr_roberta'
        elif (self.args.encoder_type=='EA2N_electra_dual'):
            file1 = 'amr_electra'
        elif (self.args.encoder_type=='EA2N_xlnet_dual'):
            file1 = 'amr_xlnet'
        else:
            file1 = 'amr_bert'

        # Selecting the dataset
        file2 = self.args.dataset
        
        # Create train, dev, test txt files and check if directory exists
        if not os.path.exists(f'../results'): 
            os.mkdir('../results')
        if not os.path.exists(f'../results/{file2}'):
            os.mkdir(f'../results/{file2}')
        if not os.path.exists(f'../results/{file2}/{file1}'):
            os.mkdir(f'../results/{file2}/{file1}')

        last_saved_file = [f for f in os.listdir(f'./ea2n_dual_model/') if f.startswith('epoch') and re.search('_last_fold\d+$', f) and (self.args.encoder_type in f)]
        if last_saved_file:
            last_saved_file = max(last_saved_file)
        else:
            last_saved_file = "No pre-trained weights loaded.."
        return file1, file2, last_saved_file
    
    
    def _metrics(self, labels, preds, probs):
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        fpr, tpr, thresholds = metrics.roc_curve(labels, probs, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        return accuracy, precision, recall, f1, auc

    def train(self):

        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        task = self.args.task
        
        dataset = read_jsonl(self.args.train_data_jsonl) + read_jsonl(self.args.dev_data_jsonl) + read_jsonl(self.args.test_data_jsonl)
        train_dataset = read_jsonl(self.args.train_data_jsonl)
        
        total_batches = math.ceil(len(dataset) / self.args.batch_size)

        kf=KFold(n_splits=self.args.kfold, shuffle=True, random_state=self.args.seed)
        all_data_itr = kf.split(np.arange(total_batches))
        
        # Files checking
        file1, file2, last_saved_file = self._file_manage()
        print(last_saved_file)

        
        # Conditions for Checkpoints
        if not os.path.exists(f'./ea2n_dual_model/'+last_saved_file):

            # Emptying the files
            with open(f'../results/{file2}/{file1}/train.txt', 'w') as f:
                f.writelines('')

            with open(f'../results/{file2}/{file1}/dev.txt', 'w') as f:
                f.writelines('')

            with open(f'../results/{file2}/{file1}/test.txt', 'w') as f:
                f.writelines('')
                
            last_epoch = 0
            last_fold = 0

        else:
            vocabs, lexical_mapping = self._build_model()
            # Load the last saved model
            loc = './ea2n_dual_model/'+last_saved_file
            #print(loc)
            self.model.load_state_dict(torch.load(loc, map_location=self.device)['model'])
            self.model = self.model.cuda(self.device)

            last_epoch = int(last_saved_file.split('epoch')[1].split('_')[0])
            last_fold = int(last_saved_file.split('_')[-1][-1])
            print(f'Model Loaded with the last epoch {last_epoch} and last fold {last_fold}')

        for fold, (train_idx,dev_idx) in enumerate(all_data_itr):
            
            if (fold < last_fold):
                continue

            if last_epoch == 0:
                vocabs, lexical_mapping = self._build_model()
            self.model.zero_grad()
            train_sampler = SubsetRandomSampler(train_idx)
            #dev_sampler = SubsetRandomSampler(dev_idx)
            all_data = DataLoader(self.args, vocabs, lexical_mapping, self.args.batch_size, True, train_sampler)
            #dev_data = DataLoader(self.args, vocabs, lexical_mapping, self.args.batch_size, False, dev_sampler)

            all_data.set_unk_rate(self.args.unk_rate)

            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.},
                {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0}
            ]
            gradient_accumulation_steps = self.args.gradient_accumulation_steps
            #t_total = math.ceil((len(train_dataset))/(gradient_accumulation_steps*self.args.batch_size)) * self.args.epochs

            t_total = math.ceil((len(train_dataset))/(self.args.batch_size)) * self.args.epochs

            optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=self.args.adam_epsilon)
            #scheduler = OneCycleLR(optimizer, total_steps=t_total, max_lr = self.args.max_lr_ratio*self.args.lr,epochs = self.args.epochs)
            scheduler = WarmupCosineSchedule(optimizer, warmup_steps=t_total*self.args.warmup_steps, t_total=t_total)
            print(f'total train batches: {t_total}')

            with open(f'../results/{file2}/{file1}/train.txt', 'a') as f:
                f.writelines('Fold ' + str(fold)+'\n')
            with open(f'../results/{file2}/{file1}/dev.txt', 'a') as f:
                f.writelines('Fold ' + str(fold)+'\n')
            with open(f'../results/{file2}/{file1}/test.txt', 'a') as f:
                f.writelines('Fold ' + str(fold)+'\n')
            
            """
            Running the Training Loop
            """

            set_seed(self.args.seed, self.args.gpus)

            batches_acm = 0
            best_acc = 0
            best_model_wts = copy.deepcopy(self.model.state_dict())
            total_steps = 0
            iterations = 0
            last_epoch = 0
            train_losses = []
            val_losses = []

            # initialize the early_stopping object
            early_stopping = EarlyStopping(patience=self.args.patience, verbose=False)
            tot = total_batches
            print(f'FOLD {fold}') 
            
            # Creating Training and Dev batches
            train_batch_data = []
            dev_batch_data = []
            for step, batch in tqdm(enumerate(all_data), total = tot):
                batch = move_to_cuda(batch, self.device)
                if batch['data_flag'] == 'train':
                    train_batch_data.append(batch)
                else :
                    dev_batch_data.append(batch)

            for epoch in range(last_epoch, self.args.epochs):
                with tqdm(total=tot, desc=f"Epoch {epoch + 1}/{self.args.epochs}", unit="batch") as pbar:
                    #pbar = tqdm(train_data, desc=f"Epoch {epoch + 1}/{self.args.epochs}", unit="batch")

                    """ TRAINING MODE """
                    self.model.train()
                    train_loss = 0.0
                    all_labels = []
                    all_preds = []
                    all_probs = []
                    batch_count = self.args.batch_multiplier
                    step = 0
                    
                    for batch in train_batch_data:
                        batch = move_to_cuda(batch, self.device)
                        logits, labels, ans_ids = self.model(batch, train=True)
                        loss = self.criterion(logits, labels)
                        train_loss += loss.item()
                        preds = logits.argmax(dim=1)

                        labels = labels.tolist()
                        preds = preds.tolist()
                        all_labels += labels
                        all_preds += preds
                        all_probs += [F.softmax(logits).cpu().detach().numpy()[i][1] for i in range(len(logits))]
                        #print(all_probs, F.softmax(logits).cpu().detach().numpy(), logits)

                        if batch_count == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            optimizer.step()
                            scheduler.step()
                            total_steps += 1
                            if (total_steps % 100 == 0):
                                print('total_step:', total_steps, 'total_iterations:', iterations)
                            optimizer.zero_grad()
                            self.model.zero_grad()

                            batch_count = self.args.batch_multiplier

                        loss.backward()
                        batch_count -= self.args.batch_size

                        # Clearing cache
                        if (batches_acm % (self.args.batch_multiplier*self.args.batch_size) == 0) & (batches_acm != 0) & (step != 0):
                            torch.cuda.empty_cache()
                        batches_acm += 1
                        step += 1
                        iterations += 1
                        pbar.update(1)
                       
                        
                    
                    train_loss = train_loss / batches_acm
                    train_losses.append(train_loss)

                    # Calculate Accuracy, Precision, Recall, F1 score
                    tA, tP, tR, tF, tAuc = self._metrics(all_labels, all_preds, all_probs)

                    """ EVALUATION MODE """
                    self.model.eval()
                    val_loss = 0.0
                    all_labels = []
                    all_preds = []
                    all_probs = []
                    batch_count = self.args.batch_multiplier
                    
                    with torch.no_grad():
                        for batch in dev_batch_data:
                            #batch = move_to_cuda(batch, self.device)
                            logits, labels, ans_ids = self.model(batch, train=False)

                            loss = self.criterion(logits, labels)
                            val_loss += loss.item()
                            preds = logits.argmax(dim=1)

                            labels = labels.tolist()
                            preds = preds.tolist()
                            all_labels += labels
                            all_preds += preds
                            all_probs += [F.softmax(logits).cpu().detach().numpy()[i][1] for i in range(len(logits))]
                            pbar.update(1)

                    val_loss = val_loss / batches_acm
                    val_losses.append(val_loss)

                    # Calculate Accuracy, Precision, Recall, F1 score
                    vA, vP, vR, vF, vAuc = self._metrics(all_labels, all_preds, all_probs)

                    pbar.set_postfix({"Train Loss": f"{train_loss:.4f} Train Accuracy: {tA:.4f} Val Loss: {val_loss:.4f} Val Accuracy: {vA:.4f}"})
                    pbar.refresh()

                """
                Writing Results and Saving Weights
                """

                # Write the epoch results
                with open(f'../results/{file2}/{file1}/train.txt', 'a') as f:
                    f.writelines('Epoch ' + str(epoch) + ': Loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, AUC: {:.4f}\n'.format(train_loss, tA, tP, tR, tF, tAuc))
                    
                with open(f'../results/{file2}/{file1}/dev.txt', 'a') as f:
                    f.writelines('Epoch ' + str(epoch) + ': Loss: {:.4f}, Accuracy: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, F1: {:.4f}, AUC: {:.4f}\n'.format(val_loss, vA, vP, vR, vF, vAuc))

                # Save only best accuracy model on dev set
                if vA > best_acc:
                    if epoch > 0:
                        best_saved_file = [f for f in os.listdir(f'ea2n_dual_model/') if f.startswith('epoch') and f.endswith('_best_fold'+str(fold))  and (self.args.encoder_type in f)][0]
                        best_epoch = int(best_saved_file.split('epoch')[1].split('_')[0])
                        os.remove(f'{self.args.ckpt}/epoch{best_epoch}_encoder{self.args.encoder_type}_model_best_fold'+str(fold))
                    best_acc = vA
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    torch.save({'args': self.save_args, 'model': best_model_wts},
                    '%s/epoch%d_encoder%s_model_best_fold%s' % (self.args.ckpt, epoch, self.args.encoder_type, str(fold)))
                    best_epoch = epoch

                if epoch > 0:
                    last_saved_file = [f for f in os.listdir(f'ea2n_dual_model/') if f.startswith('epoch') and f.endswith('_last_fold'+str(fold))  and (self.args.encoder_type in f)][0]
                    last_epoch = int(last_saved_file.split('epoch')[1].split('_')[0])
                    os.remove(f'{self.args.ckpt}/epoch{last_epoch}_encoder{self.args.encoder_type}_model_last_fold'+str(fold))
                    
                # Saving the last model
                torch.save({'args': self.save_args, 'model': best_model_wts},
                    '%s/epoch%d_encoder%s_model_last_fold%s' % (self.args.ckpt, epoch, self.args.encoder_type, str(fold)))
                
                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(val_loss, self.model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            last_epoch=0
    
        """
        Finalizing the weights and log files
        """

        # Move the saved model to the ea2n_dual_model_final folder
        if not os.path.exists(f'ea2n_dual_model_final/'):
            os.mkdir(f'ea2n_dual_model_final/')
        if not os.path.exists(f'ea2n_dual_model_final/{self.args.encoder_type}'):
            os.mkdir(f'ea2n_dual_model_final/{self.args.encoder_type}')

        for fold in range(self.args.kfold):
            # Locate best saved file such that its name ends with _best
            last_saved_file = [f for f in os.listdir(f'ea2n_dual_model/') if f.startswith('epoch') and f.endswith('_last_fold'+str(fold))  and (self.args.encoder_type in f)][0]
            best_saved_file = [f for f in os.listdir(f'ea2n_dual_model/') if f.startswith('epoch') and f.endswith('_best_fold'+str(fold)) and (self.args.encoder_type in f)][0]

            os.rename(f'ea2n_dual_model/{last_saved_file}', f'ea2n_dual_model_final/{self.args.encoder_type}/weights_last_fold'+str(fold))
            os.rename(f'ea2n_dual_model/{best_saved_file}', f'ea2n_dual_model_final/{self.args.encoder_type}/weights_best_fold'+str(fold))
    
    def evaluate_model(self, eval_file, gpus):
        self.device = torch.device("cuda:" + str(gpus) if torch.cuda.is_available() else "cpu")
        print('device', self.device)
        test_models = []
        if os.path.isdir(eval_file):
            for file in os.listdir(eval_file):
                fname = os.path.join(eval_file, file)
                if os.path.isfile(fname):
                    test_models.append(fname)
            model_args = torch.load(fname, map_location=self.device)['args']
        else:
            test_models.append(eval_file)
            model_args = torch.load(eval_file, map_location=self.device)['args']

        from data import Vocab, DataLoader, STR, END, CLS, SEL, TL, rCLS
        model_args = collections.namedtuple("HParams", sorted(model_args.keys()))(**model_args)
        vocabs = dict()
        vocabs['concept'] = Vocab(model_args.concept_vocab, 5, [CLS])
        vocabs['token'] = Vocab(model_args.token_vocab, 5, [STR, END])
        vocabs['token_char'] = Vocab(model_args.token_char_vocab, 100, [STR, END])
        vocabs['concept_char'] = Vocab(model_args.concept_char_vocab, 100, [STR, END])
        vocabs['relation'] = Vocab(model_args.relation_vocab, 5, [CLS, rCLS, SEL, TL])
        lexical_mapping = LexicalMap()

        if self.args.encoder_type:
            vocabs, lexical_mapping = self._prepare_data()
            config_class, model_class, tokenizer_class = MODEL_CLASSES[self.args.encoder_type]

            bert_config = config_class.from_pretrained(
                self.args.lm_model,
            )
            bert_tokenizer = tokenizer_class.from_pretrained(
                self.args.lm_model
            )
            bert_model = model_class.from_pretrained(
                self.args.lm_model,
                from_tf=bool(".ckpt" in self.args.lm_model),
                config=self.args.lm_model,
            ).to(self.device)

            eval_model = Reasoning_EA2N_DUAL(vocabs,
                                               model_args.concept_char_dim, model_args.concept_dim,
                                               model_args.cnn_filters, model_args.char2concept_dim,
                                               model_args.rel_dim, model_args.rnn_hidden_size, model_args.rnn_num_layers,
                                               model_args.embed_dim, model_args.bert_embed_dim, model_args.ff_embed_dim,
                                               model_args.affective_dim, model_args.num_heads,
                                               model_args.dropout,
                                               model_args.snt_layer,
                                               model_args.graph_layers,
                                               model_args.pretrained_file, self.device, model_args.batch_size,
                                               model_args.lm_model, bert_config, bert_model, bert_tokenizer, model_args.bert_max_length,
                                               model_args.n_answers,
                                               model_args.encoder_type,
                                               model_args.gcn_concept_dim, model_args.gcn_hidden_dim, model_args.gcn_output_dim,
                                               model_args.amr,
                                               model_args.affFeatures
            )

        else:
            eval_model = ''

        train_data = DataLoader(self.args, vocabs, lexical_mapping, self.args.batch_size,
                                for_train=True)
        val_data = DataLoader(self.args, vocabs, lexical_mapping, self.args.batch_size,
                              for_train=False)
        test_data = DataLoader(self.args, vocabs, lexical_mapping, self.args.test_data, self.args.batch_size,
                               for_train=False)

        self.model.eval()
        with torch.no_grad():
            train_logits, train_labels, _ = self.model(train_data, Train=False)
            val_logits, val_labels, _ = self.model(val_data, Train=False)
            test_logits, test_labels, _ = self.model(test_data, Train=False)

        train_preds = train_logits.argmax(dim=1)
        val_preds = val_logits.argmax(dim=1)
        test_preds = test_logits.argmax(dim=1)

        train_accuracy = accuracy_score(train_labels, train_preds)
        val_accuracy = accuracy_score(val_labels, val_preds)
        test_accuracy = accuracy_score(test_labels, test_preds)

        print(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
