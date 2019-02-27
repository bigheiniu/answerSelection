import torch
class config_data_preprocess:

    #max length of a post
    max_len = 100
    is_classification = True
    #remove word that frequecy less than
    min_word_count = 5

    #split data into train and test data
    train_per = 0.75
    test_per = 0.25

    # location store the raw data
    raw_data ="data/v3.2" if is_classification else "/home/yichuan/course/data/math"

    # store preprocessed data
    save_data = "data/math_remove.torchpickle" if not is_classification else "data/store_SemEval.torchpickle"
    log_file = "text_log/preprocess.log"
    logger_name = "preprocess"
    max_love_count = 10


class config_model:
    #in Debug mode or not
    DEBUG = False
    cuda = False
    device = torch.device('cuda' if cuda else 'cpu')
    is_classification = True
    log_file = "text_log/model_train.log"
    logger_name = "model_train"


    #dataset
    neg_size = 1



    #basic setting
    # epoch =
    epoch = 60
    log = None
    batch_size = 64
    num_class = 2 if is_classification else 1



    #path to store data
    # data = "data/store_stackoverflow.torchpickle" if not is_classification else "data/store_SemEval.torchpickle"
    # data = "data/store_stackoverflow_datascience.torchpickle"
    # data = "/home/yichuan/course/induceiveAnswer/data/store_SemEval.torchpickle"
    data = "data/store_SemEval.torchpickle"

    #=====================
    #content_data setting
    #====================

    #max length of question
    max_q_len = 100
    #max length of answer
    max_a_len = 100
    #max length of user context
    max_u_len = 200



    # learning rate
    lr = 0.0001

    #======================
    #word embedding setting
    #======================
    embed_fileName="data/glove/glove.6B.100d.txt"
    #vocabulary size
    vocab_size = 30000
    #word to vector embed size
    embed_size = 100

    #================
    #LSTM settingw
    #================

    #128
    lstm_hidden_size = 128
    lstm_num_layers = 2
    drop_out_lstm = 0.5
    bidirectional = False
    bidire_layer = 2 if bidirectional else 1


    #============
    #evaluate settings
    #============
    precesion_at_k = 1
    ndcg_k = 2

    #diversity setting
    use_dpp = False
    div_topK = 3
    dpp_early_stop = 0.00001

    #coverage test model setting
    lda_topic = 20
    #whether the coverage test model is already trained or not
    cov_pretrain = False
    # location to store or load model
    cov_model_path = "result"

    # #Rank evaluate setting
    # ndcg_k = 2

    #hinge loss margin
    margin = 0.1

    #===================================================== Model Specific ======================================================
    #=============
    #CNTN model
    #==============
    cntn_cnn_layers = 3
    cntn_kernel_size = [3,3,3]
    #k max_pooling last layer
    k_max_s = 30

    cntn_last_max_pool_size = k_max_s
    cntn_feature_r = 5

    # ==============
    # Multi Hop Attention Model
    # ==============

    attention_layers = 3

    #==============
    # GraphSage model
    #==============
    neighbor_number_list = [2,3]
    graphsage_depth = len(neighbor_number_list)
    max_degree = 6

    #==============
    #ARMNL smooth
    #==============
    follow_smooth = 0.1


    #=============
    #Hybrid Attention
    #=============
    hy_in_channels = 1

