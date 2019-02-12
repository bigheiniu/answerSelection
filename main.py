import argparse
from tqdm import tqdm
#pytorch import
from Util import  *
from Layer.Layer import PairWiseHingeLoss
from Layer import Layer, Model
from DataSet.dataset import clasifyDataSet
from Layer.DPP import *
from Metric.coverage_metric import *
from Metric.rank_metrics import ndcg_at_k, mean_average_precision_scikit, Accuracy, precision_at_k, mean_reciprocal_rank
import itertools




#grid search for paramter
from sklearn.model_selection import ParameterGrid
from Visualization.logger import Logger

info = {}
logger = Logger('./logs_map')
i_flag = 0
train_epoch_count = 0
eval_epoch_count = 0

def prepare_dataloaders(data, args):
    # ========= Preparing DataLoader =========#



    train_loader = torch.utils.data.DataLoader(
        clasifyDataSet(G=data['G'],
                       user_count = data['user_count'],
                       args=args,
                       is_classification=args.is_classification
                       ),
        num_workers=2,
        batch_size=args.batch_size,
        shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        clasifyDataSet(
            G=data['G'],
            user_count=data['user_count'],
            args=args,
            is_classification=args.is_classification,
            is_training=False
        ),
        num_workers=2,
        batch_size=args.batch_size,
        shuffle=True)

    return train_loader, val_loader




def train_epoch(model, data, optimizer, args, train_epoch_count):
    model.train()
    loss_fn = nn.NLLLoss()
    loss_hinge = Layer.PairWiseHingeLoss(args.margin)
    for batch in tqdm(
        data, mininterval=2, desc=' --(training)--',leave=True
    ):

        if args.is_classification:
            q_iter, a_iter, u_iter, gt_iter, _ = map(lambda x: x.to(args.device), batch)
            args.batch_size = q_iter.shape[0]
            optimizer.zero_grad()
            result, predit = model(q_iter, a_iter, u_iter)
            loss = loss_fn(result, gt_iter)
            logger.scalar_summary("train_loss",loss.item(),1)
            loss.backward()
            optimizer.step()
        else:
            question_list, answer_pos_list, user_pos_list, score_pos_list, answer_neg_list, user_neg_list, score_neg_list, count_list = map(lambda x: x.to(args.device), batch)
            args.batch_size = question_list.shape[0]
            optimizer.zero_grad()
            score_pos = model(question_list, answer_pos_list, user_pos_list)
            score_neg = model(question_list, answer_neg_list, user_neg_list)
            t = 0
            result = 0
            for i in count_list:
                result += loss_hinge(score_pos[t: t + i], score_neg[t : t + i])
                t += i
            result.backward()
            optimizer.step()

    train_epoch_count += 1



    for tag, value in model.named_parameters():
        if value.grad is None:
            continue
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.cpu().detach().numpy(), train_epoch_count)
        logger.histo_summary(tag + '/grad', value.grad.cpu().numpy(),train_epoch_count)



def eval_epoch(model, data, args, eval_epoch_count):
    model.eval()
    pred_label = []
    pred_score = []
    true_label = []
    label_score_order = []
    diversity_answer_recommendation = []
    val_answer_list = []
    question_list = []
    info_test = {}
    loss_fn = nn.NLLLoss()
    loss = 0
    ndcg_loss = 0
    query_count = 0
    with torch.no_grad():
        for batch in tqdm(
            data, mininterval=2, desc="  ----(validation)----  ", leave=True
        ):
            if args.is_classification:
                q_val, a_val, u_val, gt_val, count = map(lambda x: x.to(args.device), batch)
                args.batch_size = gt_val.shape[0]
                result, predict, feature_matrix = model(q_val, a_val, u_val, True)
                loss += loss_fn(result, gt_val)


                pred_label.append(tensorTonumpy(predict, args.cuda))
                true_label.append(tensorTonumpy(gt_val, args.cuda))

                count = tensorTonumpy(count, args.cuda)
                relevance_score = tensorTonumpy(result[:,1], args.cuda)
                feature_matrix = tensorTonumpy(feature_matrix, args.cuda)
                pred_score.append(relevance_score)
                temp = 0
                question_list.append(tensorTonumpy(q_val, args.cuda))

                for i in count:
                    score_ = relevance_score[temp:temp + i]
                    #label order based on predicted score
                    label = true_label[-1][temp:temp+i]
                    sorted_index = np.argsort(-score_)
                    label = label[sorted_index]
                    label_score_order.append(label)

                    #coverage metric
                    #index -> [0-k]
                    top_answer_index = diversity(feature_matrix, score_, sorted_index, args.dpp_early_stop)
                    #id -> [10990, 12334, 1351]
                    top_answer_id = tensorTonumpy(a_val[temp:temp+i][top_answer_index], args.cuda)
                    val_answer = tensorTonumpy(a_val[temp:temp+i], args.cuda)
                    val_answer_list.append(val_answer)
                    diversity_answer_recommendation.append(top_answer_id)
                    temp += i
            else:
                q_val, a_val, u_val, gt_val, count = map(lambda x:x.to(args.device), batch)
                args.batch_size = gt_val.shape[0]
                relevance_score, feature_matrix = model(q_val, a_val, u_val, True)
                count = tensorTonumpy(count, args.cuda)
                relevance_score = tensorTonumpy(relevance_score, args.cuda)
                temp = 0
                feature_matrix = tensorTonumpy(feature_matrix, args.cuda)
                gt_val = tensorTonumpy(gt_val, args.cuda)
                question_list.append(tensorTonumpy(q_val, args.gpu))
                for i in count:
                    score_ = relevance_score[temp:temp+i]
                    sorted_index = np.argsort(-score_)
                    # ground truth sorted based on generated score order
                    score_sorted = gt_val[sorted_index]
                    ndcg_loss += ndcg_at_k(score_sorted, args.ndcg_k)
                    temp += i
                    query_count += 1

                    # coverage metric
                    # index -> [0-k]
                    top_answer_index = diversity(feature_matrix, score_, sorted_index,
                                                       args.dpp_early_stop)
                    # id -> [10990, 12334, 1351]
                    top_answer_id = tensorTonumpy(a_val[temp:temp + i][top_answer_index], args.cuda)
                    diversity_answer_recommendation.append(top_answer_id)
                    temp += i



    if args.is_classification:
        pred_label_flatt = list(itertools.chain.from_iterable(pred_label))
        true_label_flatt = list(itertools.chain.from_iterable(true_label))
        score_list_flatt = list(itertools.chain.from_iterable(pred_score))

        accuracy, zero_count, one_count = Accuracy(pred_label_flatt, true_label_flatt)
        mAP = mean_average_precision_scikit(true_label_flatt, score_list_flatt)
        pat1 = precision_at_k(label_score_order, 1)
        mpr = mean_reciprocal_rank(label_score_order)

        # visualize the data
        info_test['eval_loss'] = loss.item()
        info_test['eval_accuracy'] = accuracy
        info_test['zero_count'] = zero_count
        info_test['one_count'] = one_count
        info_test['mAP'] = mAP
        info_test['P@1'] = pat1
        info_test['mPR'] = mpr

        print("[Info] Accuacy: {}; {} samples, {} correct prediction".format(accuracy, len(pred_label), len(pred_label) * accuracy))
        print("[Info] mAP: {}\n".format(mAP))
        eval_epoch_count += 1
    else:
        mean_ndcgg = ndcg_loss * 1.0 / query_count
        info_test['nDCGG'] = mean_ndcgg

    #coverage metric



    for tag, value in info_test.items():
        logger.scalar_summary(tag, value, eval_epoch_count)

    return diversity_answer_recommendation, val_answer_list


    # diversity_recommendation(answer_id_dic,relevance_dic, content=content, early_stop=0.00001, topN=3)



def diversity_evaluation(diversity_answer_recommendation, background_list, content, topK, tfidf, lda):
    #init evaluate class
    background_data = []
    for content_id in background_list:
        background_data.append(content[content_id])
    # tfidf = TFIDFSimilar(background_data, coverage_metric_model_pretrain, coverage_metric_model_path)
    # lda = LDAsimilarity(background_data, coverage_metric_model_pretrain, coverage_metric_model_path, lda_topic)
    tf_idf_score = 0
    lda_score = 0
    question_count = len(diversity_answer_recommendation)
    for candidate_answer_list in diversity_answer_recommendation:
        candidate_word_space = []
        temp_tfidf_score = 0
        temp_lda_score = 0
        for answer in candidate_answer_list:
            answer_content = content[answer]
            candidate_word_space += answer_content
        for top_answer in candidate_answer_list[:topK]:
            top_answer_content = content[top_answer]
            temp_tfidf_score += tfidf.simiarity(candidate_word_space, top_answer_content)
            temp_lda_score += lda.similarity(candidate_word_space, top_answer_content)


        tf_idf_score += temp_tfidf_score
        lda_score += temp_lda_score
    return (tf_idf_score * 1.0) / question_count, (lda_score * 1.0) / question_count












def grid_search(params_dic):
    '''
    :param params_dic: similar to {"conv_size":[0,1,2], "lstm_hiden_size":[1,2,3]}
    :return: iter {"conv_size":1, "lstm_hidden_size":1}
    '''
    grid_parameter = ParameterGrid(params_dic)
    parameter_list = []
    for params in grid_parameter:
        params_dic_result = {}
        for key in params_dic.keys():
            params_dic_result[key] = params[key]
        parameter_list.append(params_dic_result)
    return parameter_list



def train(args, train_data, val_data, user_count ,pre_trained_word2vec, G, content):
    adj, adj_edge, _ = Adjance(G, args.max_degree)
    adj = adj.to(args.device)
    adj_edge = adj_edge.to(args.device)
    model = Model.InducieveLearningQA(args, user_count, adj, adj_edge, content, pre_trained_word2vec).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)


    #load coverage model
    tfidf = TFIDFSimilar(content, args.cov_pretrain, args.cov_model_path)
    lda = LDAsimilarity(content, args.cov_pretrain, args.cov_model_path, args.lda_topic)
    info_val = {}
    if args.cuda:
        _content = content.cpu().numpy()
    else:
        _content = content.numpy()

    for epoch_i in range(args.epoch):
        train_epoch(model, train_data, optimizer, args, epoch_i)

        diversity_answer_recommendation, background_list = eval_epoch(model, val_data, args, eval_epoch_count)
        tfidf_cov, lda_cov = diversity_evaluation(diversity_answer_recommendation, background_list, content, args.div_topK, tfidf, lda)

        info_val['tfidf_cov'] = tfidf_cov
        info_val['lda_cov'] = lda_cov
        for tag, value in info_val.items():
            logger.scalar_summary(tag, value, eval_epoch_count)
        # print("[Info] Val Loss: {}, accuracy: {}".format(val_loss, accuracy_val))

        # test_loss, accuracy_test = eval_epoch(model, test_data, args, epoch_i)
        # print("[Info] Test Loss: {}, accuracy: {}".format(test_loss, accuracy_test))







def main():
    ''' setting '''
    parser = argparse.ArgumentParser()

    parser.add_argument("-epoch",type=int, default=60)
    parser.add_argument("-log", default=None)
    # load data
    # parser.add_argument("-data",required=True)
    parser.add_argument("-no_cuda", action="store_false")
    parser.add_argument("-lr", type=float, default=0.3)

    # induceive arguments
    parser.add_argument("-max_degree", type=int, default=6)
    parser.add_argument("-num_class", type=int, default=2)
    # 1-UIA-LSTM-CNN; 2-CNTN
    parser.add_argument("-model",type=int,default=1)
    parser.add_argument("-max_q_len", type=int, default=60)
    parser.add_argument("-max_a_len", type=int, default=60)
    parser.add_argument("-max_u_len", type=int, default=200)
    parser.add_argument("-vocab_size", type=int, default=30000)
    parser.add_argument("-embed_size", type=int, default=100)
    parser.add_argument("-lstm_hidden_size",type=int, default=128)
    parser.add_argument("-bidirectional", action="store_true")
    parser.add_argument("-class_kind", type=int, default=2)
    parser.add_argument("-embed_fileName",default="data/glove/glove.6B.100d.txt")
    parser.add_argument("-batch_size", type=int, default=64)
    parser.add_argument("-lstm_nulrm_layers", type=int, default=1)
    parser.add_argument("-drop_out_lstm", type=float, default=0.3)
    # conv parameter
    parser.add_argument("-in_channels", type=int, default=1)
    parser.add_argument("-out_channels", type=int, default=20)
    parser.add_argument("-kernel_size", type=int, default=3)

    #coverage model path
    parser.add_argument("-cov_model_path", default="result")


    args = parser.parse_args()
    #===========Load DataSet=============#


    args.data="data/store.torchpickle"
    #===========Prepare model============#
    args.cuda =  args.no_cuda
    args.device = torch.device('cuda' if args.cuda else 'cpu')

    print("cuda : {}".format(args.cuda))
    args.DEBUG=False
    args.neighbor_number_list = [2,3]
    args.depth = len(args.neighbor_number_list)
    data = torch.load(args.data)
    word2ix = data['dict']
    G = data['G']
    user_count = data['user_count']
    content = torch.LongTensor(data['content']).to(args.device)
    train_data, val_data= prepare_dataloaders(data, args)
    pre_trained_word2vec = loadEmbed(args.embed_fileName, args.embed_size, args.vocab_size, word2ix, args.DEBUG).to(args.device)
    #grid search
    # if args.model == 1:
    paragram_dic = {"lstm_hidden_size":[32, 64, 128, 256, 512],
                   "lstm_num_layers":[2,3,4],
                   "kernel_size":[3,4, 5],
                   "drop_out_lstm":[0.5],
                    "lr":[1e-4, 1e-3, 1e-2]
                    }
    pragram_list = grid_search(paragram_dic)
    args_dic = vars(args)
    for paragram in pragram_list:
        for key, value in paragram.items():
            print("Key: {}, Value: {}".format(key, value))
            args_dic[key] = value
        args.out_channels = args.lstm_hidden_size
        args.user_embed_dim = args.lstm_hidden_size
        train(args, train_data, val_data, user_count, pre_trained_word2vec, G, content)
if __name__ == '__main__':
    main()
