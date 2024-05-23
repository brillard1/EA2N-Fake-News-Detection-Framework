from collections import defaultdict

BASE_PARAMS = defaultdict(

    # model_condition
    encoder_type='lm',
    #lm_model = 'bert-base-cased',
    bert_pretrained_file=None,
    prefix='model',

    # bert
    bert_embed_dim = 768,
    bert_max_length=256,

    # Fake_Flow
    affective_dim = 240,

    # train
    kfold = 5,
    epochs=30,
    max_lr_ratio=2,
    lr=5e-5,
    adam_epsilon=1e-8,
    warmup_steps=0.1,
    seed=128,
    gradient_accumulation_steps =1,
    batch_size = 1,
    batch_multiplier =1,
    patience = 500,
    # dropout/unk
    dropout=0.2,
    unk_rate=0.33,
    
    ckpt='/home/mnt/ea2n_dual_model',

    # IO
    task = 'cqa',
    n_answers=2,
    pretrained_file=None,
    block_size=100,

    # Dataset
    affective_features = './affFeatures/affective_embedding/politifact_affective_features.pkl',
    eval_file = './{}' # enter the input file
)

EA2N_BERT_PARAMS = BASE_PARAMS.copy()
EA2N_BERT_PARAMS.update(
    encoder_type='EA2N_bert_dual',
    ###### LN Param ##
    lm_model='bert-base-uncased',

    # concept/token encoders
    concept_char_dim = 32,
    concept_dim = 300,
    max_concept_len = 100,
    snt_layer = 1,

    # char-cnn
    cnn_filters = [3,256],
    char2concept_dim = 128,

    # relation encoder
    rel_dim = 100,
    rnn_hidden_size = 256,
    rnn_num_layers = 2,

    # core architecture
    embed_dim = 256,
    ff_embed_dim = 1024,
    num_heads = 8,
    graph_layers = 4,
    n_lstm_layers = 2,
    


    ##### BERT #####
    bert_embed_dim = 768,

    ##### TRAIN #####
    task = 'csqa', ############# 이거 꼭 바꾸기!!!!
    n_answers=2,
    bert_pretrained_file = None,
    prefix = '',
    omcs= False,

    ##### IO #####
    pretrained_file = './glove/glove.840B.300d.txt',
    gpus = [0,1],

    ckpt='./ea2n_dual_model'

)

EA2N_ELECTRA_PARAMS = BASE_PARAMS.copy()
EA2N_ELECTRA_PARAMS.update(
    encoder_type='EA2N_electra_dual',

    ###### LN Param ##
    lm_model='google/electra-base-discriminator',
    
    # concept/token encoders
    concept_char_dim = 32,
    concept_dim = 300,
    max_concept_len = 100,
    snt_layer = 1,

    # char-cnn
    cnn_filters = [3,256],
    char2concept_dim = 128,

    # relation encoder
    rel_dim = 100,
    rnn_hidden_size = 256,
    rnn_num_layers = 2,

    # core architecture
    embed_dim = 512,
    ff_embed_dim = 1024,
    num_heads = 8,
    graph_layers = 4,
    n_lstm_layers = 2,

    

    ##### BERT #####
    bert_embed_dim = 768,

    ##### TRAIN #####
    task = 'csqa', ############# 이거 꼭 바꾸기!!!!
    n_answers=2,
    bert_pretrained_file = None,
    prefix = '',
    omcs= False,

    ##### IO #####
    pretrained_file = './glove/glove.840B.300d.txt',
    gpus = [0,1],

    ckpt='./ea2n_dual_model'

)

EA2N_XLNET_PARAMS = BASE_PARAMS.copy()
EA2N_XLNET_PARAMS.update(
    encoder_type='EA2N_xlnet_dual',

    ###### LN Param ##
    lm_model='xlnet-base-cased',

    # concept/token encoders
    concept_char_dim = 32,
    concept_dim = 300,
    max_concept_len = 100,
    snt_layer = 1,

    # char-cnn
    cnn_filters = [3,256],
    char2concept_dim = 128,

    # relation encoder
    rel_dim = 100,
    rnn_hidden_size = 256,
    rnn_num_layers = 2,

    # core architecture
    embed_dim = 512,
    ff_embed_dim = 1024,
    num_heads = 8,
    graph_layers = 4,
    n_lstm_layers = 2,
    

    ##### BERT #####
    bert_embed_dim = 768,

    ##### TRAIN #####
    task = 'csqa', ############# 이거 꼭 바꾸기!!!!
    n_answers=2,
    bert_pretrained_file = None,
    prefix = '',
    omcs= False,

    ##### IO #####
    pretrained_file = './glove/glove.840B.300d.txt',
    gpus = [0,1],

    ckpt='./ea2n_dual_model'

)

EA2N_ROBERTA_PARAMS = BASE_PARAMS.copy()
EA2N_ROBERTA_PARAMS.update(
    encoder_type='EA2N_roberta_dual',

    ###### LN Param ##
    lm_model='roberta-base',

    # concept/token encoders
    concept_char_dim = 32,
    concept_dim = 300,
    max_concept_len = 100,
    snt_layer = 1,

    # char-cnn
    cnn_filters = [3,256],
    char2concept_dim = 128,

    # relation encoder
    rel_dim = 100,
    rnn_hidden_size = 256,
    rnn_num_layers = 2,

    # core architecture
    embed_dim = 512,
    ff_embed_dim = 1024,
    num_heads = 8,
    graph_layers = 4,
    n_lstm_layers = 2,
    

    ##### BERT #####
    bert_embed_dim = 768,

    ##### TRAIN #####
    task = 'csqa', ############# 이거 꼭 바꾸기!!!!
    n_answers=2,
    bert_pretrained_file = None,
    prefix = '',
    omcs= False,

    ##### IO #####
    pretrained_file = './glove/glove.840B.300d.txt',
    gpus = [0,1],

    ckpt='./ea2n_dual_model'

)

