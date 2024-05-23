import logging
import warnings
import argparse

from model_params import *
from train import EA2NModel

warnings.filterwarnings(action='ignore')

PARAMS_MAP = {
    'EA2N_bert_dual':EA2N_BERT_PARAMS,
    'EA2N_electra_dual':EA2N_ELECTRA_PARAMS,
    'EA2N_xlnet_dual':EA2N_XLNET_PARAMS,
    'EA2N_roberta_dual':EA2N_ROBERTA_PARAMS,
}

def train_model(args, local_rank):
    args_save = args
    args = PARAMS_MAP[args.encoder_type]
    if 'google' in args['lm_model']:
        lm_args = args['lm_model'][args['lm_model'].index('/')+1:]
    else:
        lm_args = args['lm_model']
    args['prefix'] = str(args['encoder_type']) + '_' + args['task'] + '_lr' + str(
        args['lr']) + '_' + str(args['batch_multiplier']) + '_' + lm_args

    assert len(args['cnn_filters']) % 2 == 0

    # Setting Path for the dataset
    file =  args_save.dataset
    args['dataset'] = file

    # Setting if No AMR or AMR
    args['amr'] = True if args_save.amr == 'True' else False
    args['affFeatures'] = True if args_save.affFeatures == 'True' else False

    curr_gpu = args_save.gpus
    args['gpus'] = [0,1] if curr_gpu == 0 else [1,0]

    # Vocabulary Updation
    args['token_vocab'] = f'./mnt/wiki_data_{file}/amr_2.0/ea2n/token_vocab'
    args['entity_vocab'] = f'./mnt/wiki_data_{file}/amr_2.0/ea2n/entity_vocab'
    args['token_char_vocab'] = f'./mnt/wiki_data_{file}/amr_2.0/ea2n/token_char_vocab'
    args['entity_char_vocab'] = f'./mnt/wiki_data_{file}/amr_2.0/ea2n/entity_char_vocab'
    args['relation_vocab'] = f'./mnt/wiki_data_{file}/amr_2.0/ea2n/relation_vocab'

    # Dataset Path Updation
    args['train_data'] = f'./mnt/wiki_data_{file}/amr_2.0/ea2n/train_pred_wiki_extended_real_final.json'
    args['dev_data'] = f'./mnt/wiki_data_{file}/amr_2.0/ea2n/dev_pred_wiki_extended_real_final.json'
    args['test_data'] = f'./mnt/wiki_data_{file}/amr_2.0/ea2n/test_pred_wiki_extended_real_final.json'
    args['train_data_jsonl'] = f'./mnt/wiki_data_{file}/amr_2.0/ea2n/train_rand_split.jsonl'
    args['dev_data_jsonl'] = f'./mnt/wiki_data_{file}/amr_2.0/ea2n/dev_rand_split.jsonl'
    args['test_data_jsonl'] = f'./mnt/wiki_data_{file}/amr_2.0/ea2n/test_rand_split.jsonl'

    model = EA2NModel(args, local_rank)
    model.train()

def evaluate_model(args, local_rank):
    eval_file = args.eval_file
    gpus = args.gpus
    args_save = args
    args = PARAMS_MAP[args.encoder_type]
    if 'google' in args['lm_model']:
        lm_args = args['lm_model'][args['lm_model'].index('/')+1:]
    else:
        lm_args = args['lm_model']
    args['prefix'] = str(args['encoder_type']) + '_' + args['task'] + '_lr' + str(
        args['lr']) + '_' + str(args['batch_multiplier']) + '_' + lm_args

    assert len(args['cnn_filters']) % 2 == 0

    # Setting Path for the dataset
    file =  args_save.dataset
    args['dataset'] = file

    # Setting if No AMR or AMR
    args['amr'] = True if args_save.amr == 'True' else False
    args['affFeatures'] = True if args_save.affFeatures == 'True' else False

    curr_gpu = args_save.gpus
    args['gpus'] = [0,1] if curr_gpu == 0 else [1,0]

    model = EA2NModel(args, local_rank)
    model.evaluate_model(eval_file, gpus)

def get_params():
    # Training settings
    parser = argparse.ArgumentParser(description='gct_dual')
    parser.add_argument('--mode', dest="mode", type=str, default='eval')
    parser.add_argument("--encoder_type", dest="encoder_type", type=str, default=None,
                            help="Model Name")
    parser.add_argument("--eval_file", dest='eval_file', type=str, default=None)
    parser.add_argument("--lan_encoder", dest="lan_encoder", type=str, default='bert')
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--dataset", type=str, default='proposed')
    parser.add_argument("--amr", type=str, default='False')
    parser.add_argument("--affFeatures", type=str, default='False')
    args, _ = parser.parse_known_args()
    return args

if __name__ == "__main__":
    args = get_params()

    if args.mode == 'train':
        train_model(args, 0)
    else:
        evaluate_model(args, 0)