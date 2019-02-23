import argparse
from tqdm import tqdm
#pytorch import
from Util import *

from DataSet.dataset import classifyDataSetEdge, rankDataSetEdge, classify_collect_fn_hybrid, my_collect_fn_test_hybrid, my_collect_fn_test
from GraphSAGEDiv.DPP import *
from Metric.coverage_metric import *
from Metric.rank_metrics import ndcg_at_k, average_precision, precision_at_k, mean_reciprocal_rank, Accuracy
from Config import config_model
from GraphSAGEDiv.Model import InducieveLearningQA
from Visualization.logger import Logger

info = {}
logger = Logger('./logs_map')
i_flag = 0
train_epoch_count = 0
eval_epoch_count = 0

def prepare_dataloaders(data, args):
    # ========= Preparing DataLoader =========#
    G = data['G']
    answer_score = data['vote_sort']
    answer_user_dic = data['answer_user_dic']
    question_count = data['question_count']
    user_count = data['user_count']
    if args.is_classification:

        train_loader = torch.utils.data.DataLoader(
            classifyDataSetEdge(
                args = args,
                G = G,
                is_training=True
            ),
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
        classifyDataSetEdge(
            G = G,
            args= args,
            is_training = False
        ),
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            rankDataSetEdge(
                G=G,
                args=args,
                is_training=True,
                question_count=question_count,
                user_count=user_count,
                answer_user_dic=answer_user_dic,
                answer_score=answer_score
            ),
            num_workers=0,
            batch_size=args.batch_size,
            shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
            rankDataSetEdge(
                G=data['G'],
                args=args,
                is_training=False,
                question_count=question_count,
                user_count=user_count,
                answer_user_dic=answer_user_dic,
                answer_score=answer_score
            ),
            num_workers=0,
            batch_size=args.batch_size,
            shuffle=True

        )

    return train_loader, val_loader




def train_epoch(model, data, optimizer, args, train_epoch_count):
    model.train()
    loss_fn = nn.NLLLoss() if args.is_classification else PairWiseHingeLoss(args.margin)

    for batch in tqdm(
        data, mininterval=2, desc=' --(training)--',leave=True
    ):
        if args.is_classification:
            q_iter, a_iter, u_iter, gt_iter = map(lambda x: x.to(args.device), batch)
            args.batch_size = q_iter.shape[0]
            optimizer.zero_grad()
            result = model(q_iter, a_iter, u_iter)[0]
            loss = loss_fn(result, gt_iter)
            logger.scalar_summary("train_loss",loss.item(),1)
            loss.backward()
            optimizer.step()
        else:
            question_list, answer_pos_list, user_pos_list, score_pos_list, answer_neg_list, user_neg_list = map(lambda x: x.to(args.device), batch)
            args.batch_size = question_list.shape[0]
            optimizer.zero_grad()
            score_pos = model(question_list, answer_pos_list, user_pos_list)[0]
            score_neg = model(question_list, answer_neg_list, user_neg_list)[0]
            result = torch.sum(loss_fn(score_pos, score_neg))
            result.backward()
            optimizer.step()




    for tag, value in model.named_parameters():
        if value.grad is None:
            continue
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.cpu().detach().numpy(), train_epoch_count)
        logger.histo_summary(tag + '/grad', value.grad.cpu().numpy(),train_epoch_count)



def eval_epoch(model, data, args, eval_epoch_count):
    model.eval()
    diversity_answer_recommendation = []
    questionid_answer_score_gt_dic = {}
    info_test = {}
    accuracy = 0
    one_count = 0
    zero_count = 0
    line_count = 0
    with torch.no_grad():
        for batch in tqdm(
            data, mininterval=2, desc="  ----(validation)----  ", leave=True
        ):

            if args.is_classification:
                q_val, a_val, u_val, gt_val = map(lambda x: x.to(args.device), batch)
                args.batch_size = gt_val.shape[0]
                line_count += args.batch_size
                assert args.batch_size == gt_val.shape[0], "batch size is not eqaul {} != {}".format(args.batch_size, gt_val.shape[0])

                score, predic = model(q_val, a_val, u_val)
                score = tensorTonumpy(score, args.cuda)
                gt_val = tensorTonumpy(gt_val, args.cuda)
                question_id_list = tensorTonumpy(q_val, args.cuda)
                # biggest in the beginning
                accuracy_temp, zero_count_temp, one_count_temp = Accuracy(gt_val, predic)
                accuracy += accuracy_temp
                zero_count += zero_count_temp
                one_count += one_count_temp
                assert one_count_temp + zero_count_temp == args.batch_size,"one count + zero count is not eqaul to batch size{} != {}".format(one_count_temp + zero_count_temp, args.batch_size)

                for questionid, gt, pred_score in zip(question_id_list, gt_val, score):
                    if questionid in questionid_answer_score_gt_dic:
                        questionid_answer_score_gt_dic[questionid].append([pred_score, gt])
                    else:
                        questionid_answer_score_gt_dic[questionid] = [[pred_score, gt]]


            else:
                q_val, a_val, u_val, gt_val = map(lambda x: x.to(args.device), batch)
                args.batch_size = gt_val.shape[0]
                line_count += args.batch_size
                assert args.batch_size == gt_val.shape[0], "batch size is not eqaul {} != {}".format(args.batch_size, gt_val.shape[0])
                score, answer_feature = model(q_val, a_val, u_val,need_feature=True)
                score = tensorTonumpy(score, args.cuda)
                gt_val = tensorTonumpy(gt_val, args.cuda)
                question_id_list = tensorTonumpy(q_val, args.cuda)
                a_val = tensorTonumpy(answer_feature, args.cuda)
                for questionid, answer_content, gt, pred_score in zip(question_id_list, a_val, gt_val, score):
                    if questionid in questionid_answer_score_gt_dic:
                        questionid_answer_score_gt_dic[questionid].append([answer_content, pred_score, gt])
                    else:
                        questionid_answer_score_gt_dic[questionid] = [[answer_content, pred_score, gt]]

    mAP = 0
    mRP = 0
    p_at_one = 0
    ndcg_loss = 0

    for _, values in questionid_answer_score_gt_dic.items():
        #biggest in the begining
        answer_score_gt_reorder = sorted(values, key=lambda x: -x[-2])
        rank_gt = [line[-1] for line in answer_score_gt_reorder]
        if args.is_classification:
           mAP += average_precision(rank_gt)
           mRP += mean_reciprocal_rank([rank_gt])
           p_at_one += precision_at_k([rank_gt], args.precesion_at_k)
        else:

            rank_answer_content = np.array([line[0] for line in answer_score_gt_reorder])
            rank_relevance_score = [line[-1] for line in answer_score_gt_reorder]
            rank_list = list(range(len(rank_relevance_score)))
            rank_order = diversity(featureMatrix=rank_answer_content, rankList=rank_list, relevanceScore=rank_relevance_score, early_stop=args.dpp_early_stop)
            diversity_answer_recommendation.append(rank_order)
            ndcg_loss += ndcg_at_k(rank_gt, args.ndcg_k)
            if np.argmax(rank_gt) == 0:
                p_at_one += 1
    question_count = len(questionid_answer_score_gt_dic)


    if args.is_classification:
        mAP = mAP* 1.0 / question_count
        p_at_one = p_at_one * 1.0 / question_count
        mRP = mRP * 1.0 / question_count
        accuracy = accuracy * 1.0 / line_count
        # visualize the data
        info_test['mAP'] = mAP
        info_test['P@1'] = p_at_one
        info_test['mRP'] = mRP
        info_test['accuracy'] = accuracy
        info_test['one_count'] = one_count
        info_test['zero_count'] = zero_count
        print("[Info] mAP: {}, P@1: {}, mRP: {}, Accuracy: {}, one_count: {}, zero_count: {}".format(mAP, p_at_one, mRP, accuracy, one_count, zero_count))

    else:
        p_at_one = p_at_one * 1.0 / question_count
        ndcg_loss = ndcg_loss * 1.0 / question_count
        info_test['nDCGG'] = ndcg_loss
        info_test['P@1'] = p_at_one
        print("[INFO] Ranking Porblem nDCGG: {}, p@1 is {}".format(ndcg_loss, p_at_one))


    for tag, value in info_test.items():
        logger.scalar_summary(tag, value, eval_epoch_count)

    return diversity_answer_recommendation


def train(args, train_data, val_data, user_count, pre_trained_word2vec, G, content_numpy):
    content_embed = ContentEmbed(torch.LongTensor(content_numpy).to(args.device))
    content_numpy_embed = ContentEmbed(content_numpy)
    adj, adj_edge, _ = Adjance(G, args.max_degree)
    adj = adj.to(args.device)
    adj_edge = adj_edge.to(args.device)
    model = InducieveLearningQA(args, user_count, adj, adj_edge, content_embed, pre_trained_word2vec)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.to(args.device)
    #load coverage model
    model_path = args.cov_model_path + "Graph"
    tfidf = TFIDFSimilar(content_numpy, False, model_path)
    lda = LDAsimilarity(content_numpy, args.lda_topic, False, model_path)
    info_val = {}

    for epoch_i in range(args.epoch):

        train_epoch(model, train_data, optimizer, args, epoch_i)

        diversity_answer_recommendation = eval_epoch(model, val_data, args, eval_epoch_count)


        if args.is_classification is False:
            lda_cov = 0
            tfidf_cov = 0
            for answer_id_list in diversity_answer_recommendation:
                answer_content = content_numpy_embed.content_embed(answer_id_list)
                temp_lda, temp_tfidf = diversity_evaluation(answer_content, args.div_topK, tfidf, lda)
                lda_cov += temp_lda
                tfidf_cov += temp_tfidf
            tfidf_cov = tfidf_cov / len(diversity_answer_recommendation)
            lda_cov = lda_cov / len(diversity_answer_recommendation)

            info_val['tfidf'] = tfidf_cov
            info_val['lda'] = lda_cov
            print("[INFO] lda coverage: {}, tfidf coverage: {}".format(lda_cov, tfidf_cov))

        for tag, value in info_val.items():
            logger.scalar_summary(tag, value, eval_epoch_count)









def main():

    #===========Load DataSet=============#
    args = config_model
    print("cuda : {}".format(args.cuda))
    data = torch.load(args.data)
    word2ix = data['dict']
    G = data['G']
    user_count = data['user_count']
    content = data['content']
    train_data, val_data= prepare_dataloaders(data, args)
    pre_trained_word2vec = loadEmbed(args.embed_fileName, args.embed_size, args.vocab_size, word2ix, args.DEBUG).to(args.device)
    #grid search
    # if args.model == 1:
    paragram_dic = {"lstm_hidden_size":[4, 128, 256],
                   "lstm_num_layers":[1,2,3,4],
                   "drop_out_lstm":[0.5],
                    "lr":[1e-4, 1e-3, 1e-2],
                    "margin":[0.1, 0.2, 0.3]
                    }
    pragram_list = grid_search(paragram_dic)
    for paragram in pragram_list:
        for key, value in paragram.items():
            print("Key: {}, Value: {}".format(key, value))
            setattr(args, key, value)
        train(args=args, train_data=train_data, val_data=val_data,
              user_count=user_count, G=G, content_numpy=content, pre_trained_word2vec=pre_trained_word2vec)

if __name__ == '__main__':
    main()
