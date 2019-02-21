import argparse
from tqdm import tqdm
#pytorch import
from Util import *
from GraphSAGEDiv import Model as Inducive_Model
from GraphSAGEDiv import Layer as Inducive_Layer
from HybridAttention import Model as Hybrid_Model

from DataSet.dataset import clasifyDataSet, rankDataSetUserContext, context_collect_fn_train, context_collect_fn_test, classify_collect_fn
from GraphSAGEDiv.DPP import *
from Metric.coverage_metric import *
from Metric.rank_metrics import ndcg_at_k, mean_average_precision, Accuracy, precision_at_k, mean_reciprocal_rank
import itertools
from Config import config_model




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
    user_context = data['user_context']
    user_count = data['user_count']
    question_count = data['question_count']
    content_embed = ContentEmbed(data['content'])
    if args.is_classification:

        train_loader = torch.utils.data.DataLoader(
            clasifyDataSet(G=data['G'],
                           args=args,
                        question_list=train_question,
                           user_context=user_context,
                           content=content_embed,
                           user_count=user_count
                       ),
        num_workers=0,
        batch_size=args.batch_size,
        collate_fn=classify_collect_fn,
        shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
        clasifyDataSet(
            G=data['G'],
            args=args,
            question_list=test_question,
            user_context=user_context,
            content=content_embed,
            user_count=user_count
        ),
        num_workers=0,
        batch_size=args.batch_size,
        collate_fn=classify_collect_fn,
        shuffle=True)
    else:
        train_data = data['question_answer_user_train']
        test_data = data['question_answer_user_test']
        answer_score = data['vote_sort']
        answer_user_dic = data['answer_user_dic']
        train_loader = torch.utils.data.DataLoader(
            rankDataSetUserContext(
                args=args,
                question_answer_user_vote=train_data,
                content=content_embed,
                user_context=user_context,
                question_count=question_count,
                user_count=user_count,
                is_training=True,
                answer_score=answer_score,
                answer_user_dic=answer_user_dic
            ),
            num_workers=4,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn= context_collect_fn_train
        )

        val_loader = torch.utils.data.DataLoader(
            rankDataSetUserContext(
                args=args,
                question_answer_user_vote=test_data,
                is_training=False,
                user_context=user_context,
                content=content_embed,
                user_count=user_count

            ),
            num_workers=4,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=context_collect_fn_test

        )

    return train_loader, val_loader




def train_epoch(model, data, optimizer, args, train_epoch_count):
    model.train()
    loss_fn = nn.NLLLoss() if args.is_classification else Inducive_Layer.PairWiseHingeLoss(args.margin)
    flag_i = 0
    flag_j = 0
    for batch in tqdm(
        data, mininterval=2, desc=' --(training)--',leave=True
    ):
        if args.is_classification:
            q_iter, a_iter, u_iter, gt_iter, _ = map(lambda x: x.to(args.device), batch)
            args.batch_size = q_iter.shape[0]

            optimizer.zero_grad()
            result = model(q_iter, a_iter, u_iter)[0]
            loss = loss_fn(result, gt_iter)
            logger.scalar_summary("train_loss",loss.item(),1)
            loss.backward()
            optimizer.step()
        else:
            flag_j += 1
            question_list, answer_pos_list, user_pos_list, score_pos_list, answer_neg_list, user_neg_list = map(lambda x: x.to(args.device), batch)
            args.batch_size = question_list.shape[0]

            # print("batch size {}".format(args.batch_size))
            optimizer.zero_grad()
            score_pos = model(question_list, answer_pos_list, user_pos_list)[0]
            score_neg = model(question_list, answer_neg_list, user_neg_list)[0]
            result = loss_fn(score_pos, score_neg)
            result.backward()
            optimizer.step()

    train_epoch_count += 1
    print("pass train_data {}, actually train_data {}".format(flag_j - flag_i, flag_i))



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
    question_answer_label_dic = {}
    diversity_answer_recommendation = []
    val_answer_list = []
    question_list = []
    info_test = {}
    loss_fn = nn.NLLLoss()
    loss = 0
    ndcg_loss = 0
    query_count = 0
    pat1_count = 0
    with torch.no_grad():
        for batch in tqdm(
            data, mininterval=2, desc="  ----(validation)----  ", leave=True
        ):

            if args.is_classification:
                q_val, a_val, u_val, gt_val, count = map(lambda x: x.to(args.device), batch)
                args.batch_size = gt_val.shape[0]
                result, score, predict, feature_matrix = model(q_val, a_val, u_val, True)
                loss += loss_fn(result, gt_val)
                pred_label.append(tensorTonumpy(predict, args.cuda))
                true_label.append(tensorTonumpy(gt_val, args.cuda))

                count = tensorTonumpy(count, args.cuda)
                relevance_score = tensorTonumpy(score[:,1], args.cuda)
                feature_matrix = tensorTonumpy(feature_matrix, args.cuda)
                pred_score.append(relevance_score)
                temp = 0
                question_list.append(tensorTonumpy(q_val, args.cuda))

                for i in count:
                    score_slice = relevance_score[temp:temp + i]
                    feature_matrix_slice = feature_matrix[temp:temp+i]
                    #label order based on predicted score
                    label = true_label[-1][temp:temp+i]
                    sorted_index = np.argsort(-score_slice)
                    label = label[sorted_index]
                    label_score_order.append(label)

                    #coverage metric
                    #index -> [0-k]
                    if args.use_dpp:
                        top_answer_index = diversity(feature_matrix_slice, score_slice, sorted_index, args.dpp_early_stop)
                    else:
                        top_answer_index = list(range(2)) if i > 1 else [0]
                    #id -> [10990, 12334, 1351]
                    top_answer_id = tensorTonumpy(a_val[temp:temp+i][top_answer_index], args.cuda)
                    val_answer = tensorTonumpy(a_val[temp:temp+i], args.cuda)
                    val_answer_list.append(val_answer)
                    diversity_answer_recommendation.append(top_answer_id)
                    temp += i
            else:
                q_val, a_val, u_val, gt_val, question_id = map(lambda x:x.to(args.device), batch)
                args.batch_size = gt_val.shape[0]
                # if len(gt_val.shape) == 1 or args.batch_size <= 1:
                #     continue
                relevance_score = model(q_val, a_val, u_val, False)[0]
                relevance_score = tensorTonumpy(relevance_score, args.cuda)
                gt_val = tensorTonumpy(gt_val, args.cuda)
                a_val = tensorTonumpy(a_val, args.cuda)
                question_id = tensorTonumpy(question_id, args.cuda)
                for q_id, a_feature, vote, pred_score in zip(question_id, a_val, gt_val, relevance_score):
                    if q_id in question_answer_label_dic:
                        question_answer_label_dic[q_id].append([a_feature, vote, pred_score[0]])
                    else:
                        question_answer_label_dic[q_id] = [[a_feature, vote, pred_score[0]]]



    if args.is_classification:
        pred_label_flatt = list(itertools.chain.from_iterable(pred_label))
        true_label_flatt = list(itertools.chain.from_iterable(true_label))
        score_list_flatt = list(itertools.chain.from_iterable(pred_score))

        accuracy, zero_count, one_count = Accuracy(true_label_flatt, pred_label_flatt)
        mAP = mean_average_precision(label_score_order)
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

        print("[Info] Accuacy: {}; One Count {}".format(accuracy, one_count))
        print("[Info] mAP: {}".format(mAP))
        eval_epoch_count += 1
    else:
        # rank the score based on predice_score
        pat1_count = 0
        ndcg_loss = []
        for value in question_answer_label_dic.values():
            sorted_value = sorted(value, key=lambda x: -x[-1])
            maxVote = np.argmax([line[-2] for line in value])
            maxPredic = np.argmax([line[-1] for line in value])
            if maxPredic == maxVote:
                pat1_count += 1

            order_vote = []
            feature_matrix = []
            for line in sorted_value:
                order_vote.append(line[-2])
                feature_matrix.append(line[0])
            diversity_answer_recommendation.append(feature_matrix)
            ndcg_loss.append(ndcg_at_k(order_vote, 3))
        mean_pat1 = pat1_count * 1.0 / len(ndcg_loss)
        ndcg_loss = np.mean(ndcg_loss)
        info_test['nDCGG'] = ndcg_loss
        info_test['P@1'] = mean_pat1
        print("[INFO] Ranking Porblem nDCGG: {}, p@1: {}".format(ndcg_loss, mean_pat1))

    #coverage metric



    for tag, value in info_test.items():
        logger.scalar_summary(tag, value, eval_epoch_count)

    return diversity_answer_recommendation, val_answer_list


    # diversity_recommendation(answer_id_dic,relevance_dic, content=content, early_stop=0.00001, topN=3)



def diversity_evaluation(diversity_answer_recommendation, topK, tfidf, lda):
    #init evaluate class
    tf_idf_score = 0
    lda_score = 0
    question_count = len(diversity_answer_recommendation)
    for candidate_answer_list in diversity_answer_recommendation:
        candidate_word_space = []
        temp_tfidf_score = 0
        temp_lda_score = 0
        for answer in candidate_answer_list:
            candidate_word_space += answer.tolist()
        for top_answer in candidate_answer_list[:topK]:
            temp_tfidf_score += tfidf.simiarity(candidate_word_space, top_answer)
            temp_lda_score += lda.similarity(candidate_word_space, top_answer)


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



def train(args, train_data, val_data, user_count ,pre_trained_word2vec, content_numpy):
    model = Hybrid_Model.HybridAttentionModel(args, pre_trained_word2vec,user_count)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.to(args.device)
    #load coverage model
    tfidf = TFIDFSimilar(content_numpy, args.cov_pretrain, args.cov_model_path)
    lda = LDAsimilarity(content_numpy, args.lda_topic, args.cov_pretrain, args.cov_model_path)
    if args.cov_pretrain is False:
        args.cov_pretrain = True
    info_val = {}

    for epoch_i in range(args.epoch):

        train_epoch(model, train_data, optimizer, args, epoch_i)

        diversity_answer_recommendation, _ = eval_epoch(model, val_data, args, eval_epoch_count)
        tfidf_cov, lda_cov = diversity_evaluation(diversity_answer_recommendation, args.div_topK, tfidf, lda)

        info_val['tfidf_cov'] = tfidf_cov
        info_val['lda_cov'] = lda_cov
        print("[INFO] tfidf coverage {}, lda coverage {}".format(tfidf_cov, lda_cov))
        for tag, value in info_val.items():
            logger.scalar_summary(tag, value, eval_epoch_count)

        # test_loss, accuracy_test = eval_epoch(model, test_data, args, epoch_i)
        # print("[Info] Test Loss: {}, accuracy: {}".format(test_loss, accuracy_test))







def main():

    #===========Load DataSet=============#
    args = config_model
    print("cuda : {}".format(args.cuda))
    datafoler = "data/"
    datasetname = ["apple.torchpickle", "tex.torchpickle" ,"math.torchpickle"]
    for datan in datasetname:
        args.data = datafoler + datan
        data = torch.load(args.data)
        word2ix = data['dict']
        user_count = data['user_count']
        content = np.array(data['content'])
        train_data, val_data= prepare_dataloaders(data, args)
        pre_trained_word2vec = loadEmbed(args.embed_fileName, args.embed_size, args.vocab_size, word2ix, args.DEBUG).to(args.device)
        # model_name = args.model_name
    #grid search
    # # if args.model == 1:
    # paragram_dic = {"lstm_hidden_size":[128, 256],
    #                "lstm_num_layers":[1,2,3,4],
    #                "drop_out_lstm":[0.5],
    #                 "lr" : [1e-4, 1e-3, 1e-2],
    #                 "margin" : [0.1, 0.2, 0.3]
    #                 }
    # pragram_list = grid_search(paragram_dic)
    # for paragram in pragram_list:
    #     for key, value in paragram.items():
    #         print("Key: {}, Value: {}".format(key, value))
    #         setattr(args, key, value)
        train(args, train_data, val_data, user_count, pre_trained_word2vec, content)
if __name__ == '__main__':
    main()
