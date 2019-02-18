import torch
class config_data_preprocess:

    #max length of a post
    max_len = 100
    is_classification = True
    #remove word that frequecy less than
    min_word_count = 5

    #split data into train and test data
    train_size = 0.6
    test_size = 0.4

    # location store the raw data
    raw_data ="data/v3.2" if is_classification else "/home/yichuan/course/data"

    # store preprocessed data
    save_data = "data/store_stackoverflow.torchpickle" if not is_classification else "data/store_SemEval.torchpickle"
    log_file = "text_log/preprocess.log"
    logger_name = "preprocess"
    max_love_count = 10


class config_model:
    #in Debug mode or not
    DEBUG = True
    is_classification = True
    log_file = "text_log/model_train.log"
    logger_name = "model_train"



    #basic setting
    epoch = 60
    log = None
    batch_size = 34
    num_class = 2 if is_classification else 1
    model_name = "CNTN"
    cuda = True
    device = torch.device('cuda' if cuda else 'cpu' )
    #path to store data
    data = "data/store_stackoverflow.torchpickle" if not is_classification else "data/store_SemEval.torchpickle"

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
    lr = 0.001

    #======================
    #word embedding setting
    #======================
    embed_fileName="data/glove/glove.6B.100d.txt"
    #vocabulary size
    vocab_size = 30000
    #word to vector embed size
    embed_size = 100

    #================
    #LSTM setting
    #================
    lstm_hidden_size = 128
    lstm_num_layers = 1
    drop_out_lstm = 0.3
    bidirectional = False
    bidire_layer = 2 if bidirectional else 1


    #============
    #evaluate settings
    #============
    # rank or classification


    #diversity setting
    use_dpp = True
    div_topK = 1
    dpp_early_stop = 0.0001

    #coverage test model setting
    lda_topic = 20
    #whether the coverage test model is already trained or not
    cov_pretrain = True
    # location to store or load model
    cov_model_path = "result"

    #Rank evaluate setting
    ndcg_k = 2

    #hinge loss margin
    margin = 0.1

    #===================================================== Model Specific ======================================================
    #=============
    #CNTN model
    #==============
    cntn_cnn_layers = 3
    cntn_kernel_size = [3,3,3]
    #k max_pooling last layer
    k_max_s = 100

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
    max_degree=6

    #==============
    #ARMNL smooth
    #==============
    follow_smooth = 0.1
