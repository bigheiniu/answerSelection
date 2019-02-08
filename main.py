import argparse
from tqdm import tqdm
#pytorch import
from Util import  *
from Layer import Model
from DataSet.dataset import clasifyDataSet
from Layer.DPP import *
from CoverageMetric.Similarity import *
import itertools




#grid search for paramter
from sklearn.model_selection import ParameterGrid
from Visualization.logger import Logger

info = {}
logger = Logger('./logs_map')
i_flag = 0

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




def train_epoch(model, data, optimizer, args, epoch):
    model.train()
    loss_fn = nn.NLLLoss()
    for batch in tqdm(
        data, mininterval=2, desc=' --(training)--',leave=True
    ):
        if args.is_classification:
            q_iter, a_iter, u_iter, gt_iter = map(lambda x: x.to(args.device), batch)
            args.batch_size = q_iter.shape[0]
            optimizer.zero_grad()
            result, predit, _ = model(q_iter, a_iter, u_iter)
            loss = loss_fn(result, gt_iter)
            logger.scalar_summary("train_loss",loss.item(),1)
            loss.backward()
            optimizer.step()
        else:
            q_iter, a_pos_iter, u_pos_iter, score_pos_iter, a_neg_iter, u_pos_iter, score_neg_iter = map(lambda x: x.to(args.device), batch)



    for tag, value in model.named_parameters():
        if value.grad is None:
            continue
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.cpu().detach().numpy(), 1)
        logger.histo_summary(tag + '/grad', value.grad.cpu().numpy(),1)



def eval_epoch(model, data, args, epoch, content, user_count):
    model.eval()
    pred_label = []
    pred_score = []
    true_label = []
    answer_id_dic = {}
    relevance_dic = {}
    loss_fn = nn.NLLLoss()
    loss = 0
    with torch.no_grad():
        for batch in tqdm(
            data, mininterval=2, desc="  ----(validation)----  ", leave=True
        ):
            # q_iter, a_iter, u_iter, gt_iter = map(lambda x: x.to(args.device), batch)
            q_val, a_val, u_val, gt_val = map(lambda x: x.to(args.device), batch)
            args.batch_size = gt_val.shape[0]
            result, predict, relevance_score = model(q_val, a_val, u_val)
            loss += loss_fn(result, gt_val)
            pred_label.append(predict)
            true_label.append(gt_val)
            pred_score.append(result[:,1])
            # if args.cuda:
            #     question_id_list = q_val.cpu().numpy()
            #     answer_id_list = a_val.cpu().numpy()
            #     relevance_score = relevance_score.cpu().numpy()
            # else:
            #     question_id_list = q_val.numpy()
            #     answer_id_list = a_val.numpy()
            #     relevance_score = relevance_score.numpy()
            #
            # for index, question_id in enumerate(question_id_list):
            #     question_id = question_id - user_count
            #     if question_id in answer_id_dic:
            #
            #         answer_id_dic[question_id].append(answer_id_list[index] - user_count)
            #         relevance_dic[question_id].append(relevance_score[index])
            #     else:
            #         answer_id_dic[question_id] = [answer_id_list[index] - user_count]
            #         relevance_dic[question_id] = [relevance_score[index]]


    # diversity_recommendation(answer_id_dic,relevance_dic, content=content, early_stop=0.00001, topN=3)





    pred_label = torch.cat(pred_label)
    true_label = torch.cat(true_label)
    pred_score = torch.cat(pred_score)
    accuracy, zero_count, one_count = Accuracy(pred_label, true_label)
    mean_average_precesion = mAP(true_label,pred_score)
    # precesion_at_one = Precesion_At_One(true_label, pred_score, question_id_list)
    info['eval_loss'] = loss.item()
    info['eval_accuracy'] = accuracy
    info['zero_count'] = zero_count
    info['one_count'] = one_count
    info['mAP'] = mean_average_precesion
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch)
    print("[Info] Accuacy: {}; {} samples, {} correct prediction".format(accuracy, len(pred_label), len(pred_label) * accuracy))
    return loss, accuracy



def diversity_recommendation(answe_id_dic, relevance_dic, content, early_stop, topN):
    #init evaluate class
    background_data = []
    recommend_list = []
    for question_id, answer_id_list in answe_id_dic.items():
        answer_content = content[answer_id_list]
        question_content = content[question_id]

        background_data.append(question_content)
        background_data += (answer_content.tolist())
        relevance_score = relevance_dic[question_id]
        S = diversity(answer_content, relevance_score, list(range(len(answer_content))), early_stop)
        recommend_list.append(S)


        #evaluate recommendation

    tfidf = TFIDFSimilar(background_data)
    # lda = LDAsimilarity(model_path=lda_model_path, background_data=background_data, topic_count=topic_count)

    tfidf_score = 0
    for recommend in recommend_list:
        recommend = itertools.chain.from_iterable(recommend[:topN])
        compare = itertools.chain.from_iterable(recommend)
        tfidf_score += tfidf.simiarity(recommend, compare)
    tfidf_score /= (len(recommend_list) * 1.0)

    # info['tfidf_score']  = tfidf_score
    logger.scalar_summary("tfidf_score", tfidf_score, 0)
    print("[INFO] coverage ratio: {}".format(tfidf_score))











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
    model = Model.InducieveLearning(args, user_count,adj, adj_edge, content, pre_trained_word2vec).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.cuda:
        _content = content.cpu().numpy()
    else:
        _content = content.numpy()

    #TODO: Early stopping
    for epoch_i in range(args.epoch):
        train_epoch(model, train_data, optimizer, args, epoch_i)
        eval_epoch(model, val_data, args, epoch_i, content=_content, user_count=user_count)


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


    args = parser.parse_args()
    #===========Load DataSet=============#


    args.data="data/store.torchpickle"
    #===========Prepare model============#
    args.cuda =  args.no_cuda
    args.device = torch.device('cuda' if  args.cuda else 'cpu')

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
