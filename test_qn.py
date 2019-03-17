import argparse
from tqdm import tqdm
#pytorch import
from Util import *

from DataSet.dataset import classifyDataNei, classify_collect_fn_Nei
from Metric.rank_metrics import ndcg_at_k, average_precision, precision_at_k, mean_reciprocal_rank, Accuracy, marcoF1
from Config import config_model
from SimQues.Layer import *
from SimQues.Model import QuestinGenerate
from Visualization.logger import Logger
import torch

info = {}
log_filename = "./logs_qsim_embed_1e4"
if os.path.isdir(log_filename) is False:
    os.mkdir(log_filename)
filelist = [ f for f in os.listdir(log_filename)]
for f in filelist:
    os.remove(os.path.join(log_filename, f))
logger = Logger(log_filename)
i_flag = 0
train_epoch_count = 0
eval_epoch_count = 0

def prepare_dataloaders(data, ques_finder, args):
    # ========= Preparing DataLoader =========#
    train_data = data['question_answer_user_train']
    test_data = data['question_answer_user_test']
    user_count = data['user_count']
    # if args.is_classification:

    train_loader = torch.utils.data.DataLoader(
            classifyDataNei(
                args = args,
                question_neighbor_finder=ques_finder,
                user_count=user_count,
                question_answer_user_vote=train_data

            ),
            num_workers=4,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn = classify_collect_fn_Nei
        )

    val_loader = torch.utils.data.DataLoader(
        classifyDataNei(
            args= args,
            question_neighbor_finder=ques_finder,
            user_count=user_count,
            question_answer_user_vote=test_data
        ),
        num_workers=4,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn = classify_collect_fn_Nei
    )
    # else:
        # train_loader = torch.utils.data.DataLoader(
        #     rankDataSetEdge(
        #         G=G,
        #         args=args,
        #         is_training=True,
        #         question_count=question_count,
        #         user_count=user_count,
        #         answer_user_dic=answer_user_dic,
        #         answer_score=answer_score
        #     ),
        #     num_workers=4,
        #     batch_size=args.batch_size,
        #     shuffle=True
        # )
        #
        # val_loader = torch.utils.data.DataLoader(
        #     rankDataSetEdge(
        #         G=data['G'],
        #         args=args,
        #         is_training=False,
        #         question_count=question_count,
        #         user_count=user_count,
        #         answer_user_dic=answer_user_dic,
        #         answer_score=answer_score
        #     ),
        #     num_workers=4,
        #     batch_size=args.batch_size,
        #     shuffle=True
        #
        # )

    return train_loader, val_loader




def train_epoch(model, data, optimizer, args, train_epoch_count):
    model.train()
    loss_fn = nn.NLLLoss() if args.is_classification else PairWiseHingeLoss(args.margin)
    loss1 = 0
    line_count = 0
    for batch in tqdm(
        data, mininterval=2, desc=' --(training)--',leave=True
    ):

        if args.is_classification:
            q_iter, question_key, question_neighbor, question_neighbor_key, a_iter, u_iter, gt_iter = map(lambda x: x.to(args.device), batch)
            args.batch_size = q_iter.shape[0]
            line_count += args.batch_size
            optimizer.zero_grad()
            result = model(q_iter, question_key, question_neighbor, question_neighbor_key, a_iter, u_iter)
            loss = loss_fn(result, gt_iter)
            loss1 += loss.item()
            loss.backward()
            optimizer.step()
        else:
            question_list, answer_pos_list, user_pos_list, score_pos_list, answer_neg_list, user_neg_list = map(lambda x: x.to(args.device), batch)
            args.batch_size = question_list.shape[0]
            optimizer.zero_grad()
            line_count += args.batch_size
            score_pos = model(question_list, answer_pos_list, user_pos_list)[0]
            score_neg = model(question_list, answer_neg_list, user_neg_list)[0]
            result = torch.sum(loss_fn(score_pos, score_neg))
            result.backward()
            optimizer.step()

    logger.scalar_summary("train_loss", loss1 / line_count, train_epoch_count)
    for tag, value in model.named_parameters():
        if value.grad is None:
            continue
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.cpu().detach().numpy(), train_epoch_count)
        logger.histo_summary(tag + '/grad', value.grad.cpu().numpy(),train_epoch_count)



def eval_epoch(model, data, args, eval_epoch_count):
    model.eval()
    # 0.5917252146760343　random init
    diversity_answer_recommendation = []
    questionid_answer_score_gt_dic = {}
    info_test = {}
    accuracy = 0
    one_count = 0
    zero_count = 0
    line_count = 0
    loss_fn = nn.NLLLoss()
    loss = 0
    gt_list = []
    predic_list = []
    with torch.no_grad():
        for batch in tqdm(
            data, mininterval=2, desc="  ----(validation)----  ", leave=True
        ):

            if args.is_classification:
                q_val, question_key, question_neighbor, question_neighbor_feature, a_val, u_val, gt_val = map(lambda x: x.to(args.device), batch)
                args.batch_size = gt_val.shape[0]
                line_count += args.batch_size
                assert args.batch_size == gt_val.shape[0], "batch size is not eqaul {} != {}".format(args.batch_size, gt_val.shape[0])

                score = model(q_val, question_key, question_neighbor, question_neighbor_feature, a_val, u_val, gt_val)
                loss += loss_fn(score, gt_val)
                predic = torch.argmax(score, dim=-1)
                score = tensorTonumpy(score, args.cuda)
                score = score[:,1]
                gt_val = tensorTonumpy(gt_val, args.cuda)
                predic = tensorTonumpy(predic, args.cuda)
                gt_list += gt_val.tolist()
                predic_list += predic.tolist()
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
                score = model(q_val, a_val, u_val,need_feature=True)[0]
                score = tensorTonumpy(score, args.cuda)
                gt_val = tensorTonumpy(gt_val, args.cuda)
                question_id_list = tensorTonumpy(q_val, args.cuda)
                # a_val = tensorTonumpy(answer_feature, args.cuda)
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
        answer_score_gt_reorder = sorted(values,  key=lambda x: -x[-2])
        rank_gt = [line[-1] for line in answer_score_gt_reorder]
        if args.is_classification:
           mAP += average_precision(rank_gt)
           mRP += mean_reciprocal_rank([rank_gt])
           p_at_one += precision_at_k([rank_gt], args.precesion_at_k)
        else:

            # rank_answer_content = np.array([line[0] for line in answer_score_gt_reorder])
            # rank_relevance_score = [line[-1] for line in answer_score_gt_reorder]
            # rank_list = list(range(len(rank_relevance_score)))
            # rank_order = diversity(featureMatrix=rank_answer_content, rankList=rank_list, relevanceScore=rank_relevance_score, early_stop=args.dpp_early_stop)
            # diversity_answer_recommendation.append(rank_order)
            ndcg_loss += ndcg_at_k(rank_gt, args.ndcg_k)
            if np.argmax(rank_gt) == 0:
                p_at_one += 1
    question_count = len(questionid_answer_score_gt_dic)


    if args.is_classification:
        mAP = mAP* 1.0 / question_count
        p_at_one = p_at_one * 1.0 / question_count
        mRP = mRP * 1.0 / question_count
        accuracy = accuracy * 1.0 / line_count
        loss = loss / line_count
        marco_f1 = marcoF1(y_gt=gt_list, y_pred=predic_list)
        # visualize the data
        info_test['mAP'] = mAP
        info_test['P@1'] = p_at_one
        info_test['mRP'] = mRP
        info_test['accuracy'] = accuracy
        info_test['marco_f1'] = marco_f1
        info_test['one_count'] = one_count
        info_test['zero_count'] = zero_count
        info_test['eval_loss'] = loss
        print("[Info] p@1: {}, ACC: {}, mAP: {}, MRP: {}, f1: {}, loss: {}, one_count: {}, zero_count: {}".format(p_at_one, accuracy, mAP, mRP, marco_f1, loss,one_count, zero_count))

    else:
        p_at_one = p_at_one * 1.0 / question_count
        ndcg_loss = ndcg_loss * 1.0 / question_count
        info_test['nDCGG'] = ndcg_loss
        info_test['P@1'] = p_at_one
        print("[INFO] Ranking Porblem nDCGG: {}, p@1 is {}".format(ndcg_loss, p_at_one))


    for tag, value in info_test.items():
        logger.scalar_summary(tag, value, eval_epoch_count)

    return diversity_answer_recommendation


def train(args, train_data, val_data, user_count, question_count, pre_trained_word2vec, content_numpy, context, epoch_count):
    content_embed = ContentEmbed(torch.LongTensor(content_numpy).to(args.device))
    content_numpy_embed = ContentEmbed(content_numpy)
    user_embed_model = UserContextEmbed(content_numpy_embed, args, user_count)
    user_embed_matrix = user_embed_model.buildUserContextEmbed(context)
    user_embed_matrix = ContentEmbed(torch.LongTensor(user_embed_matrix).to(args.device))
    model = QuestinGenerate(args, user_count, question_count, content_embed, user_embed_matrix,
                            pre_trained_word2vec)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.to(args.device)
    #load coverage model
    # model_path = args.cov_model_path + "Graph"
    # tfidf = TFIDFSimilar(content_numpy, False, model_path)
    # lda = LDAsimilarity(content_numpy, args.lda_topic, False, model_path)
    info_val = {}
    for epoch_i in range(args.epoch):

        train_epoch(model, train_data, optimizer, args, epoch_count)

        eval_epoch(model, val_data, args, epoch_count)

        epoch_count += 1
        # if args.is_classification is False:
        #     lda_cov = 0
        #     tfidf_cov = 0
        #     for answer_id_list in diversity_answer_recommendation:
        #         answer_content = content_numpy_embed.content_embed(answer_id_list)
        #         temp_lda, temp_tfidf = diversity_evaluation(answer_content, args.div_topK, tfidf, lda)
        #         lda_cov += temp_lda
        #         tfidf_cov += temp_tfidf
        #     tfidf_cov = tfidf_cov / len(diversity_answer_recommendation)
        #     lda_cov = lda_cov / len(diversity_answer_recommendation)
        #
        #     info_val['tfidf'] = tfidf_cov
        #     info_val['lda'] = lda_cov
        #     print("[INFO] lda coverage: {}, tfidf coverage: {}".format(lda_cov, tfidf_cov))

        for tag, value in info_val.items():
            logger.scalar_summary(tag, value, eval_epoch_count)

    return epoch_count



def buildAnnoyIndex(question_count, content, topic_count):
    lda_model = LDAKey(content, topic_count=topic_count, model_path='./data/lda_model')
    lda_vec = np.zeros((question_count, topic_count))
    for i in range(question_count):
        lda_output = lda_model.lda_by_index(i)
        for key, value in lda_output:
            lda_vec[i, key] = value


    # lda = LDAKey(content, topic_count, load_pretrain=False, model_path="./data/lda_model")
    # lda_vec_list = [lda.lda_by_index(i) for i in range(question_count)]
    que_neighbor = NeighborLocation(topic_count, file_path="./data/annoy_index", key_vector_list=lda_vec)
    return que_neighbor

def loadAnnoyIndex(topic_count):
    que_neighbor = NeighborLocation(topic_count, file_path="./data/annoy_index", load_file=True)
    return que_neighbor


def main():

    #===========Load DataSet=============#
    args = config_model
    print("cuda : {}".format(args.cuda))
    data = torch.load(args.data)
    user_count = data['user_count']
    content = data['content']
    word2ix = data['dict']
    context = data['user_context']
    question_count = data['question_count']
    if False:
        pre_trained_word2vec = loadEmbed(args.embed_fileName, args.embed_size, args.vocab_size, word2ix, args.DEBUG).to(args.device)
        torch.save(pre_trained_word2vec,"./word_vec_class.fuck")
        que_neighbor = buildAnnoyIndex(question_count, content, args.lda_topic)
    else:
        pre_trained_word2vec = torch.load("./word_vec_class.fuck")
        que_neighbor = loadAnnoyIndex(args.lda_topic)

    train_data, val_data= prepare_dataloaders(data, que_neighbor, args)


    #grid search
    # if args.model == 1:
    paragram_dic = {"lstm_hidden_size":[128],
                   "lstm_num_layers":[2,3,4,5],
                   "drop_out_lstm":[0.3],
                    "lr":[1e-4],
                    "neighbor_number_list": [[2], [2, 5], [5,2],[5, 5]]
                    # "margin":[0.1, 0.2, 0.3]
                    }
    pragram_list = grid_search(paragram_dic)
    epoch_count = 0
    for paragram in pragram_list:
        for key, value in paragram.items():
            print("Key: {}, Value: {}".format(key, value))
            setattr(args, key, value)
        epoch_count = train(epoch_count=epoch_count,
            args=args, train_data=train_data, val_data=val_data,
              user_count=user_count, question_count=question_count, content_numpy=content, pre_trained_word2vec=pre_trained_word2vec,context=context)

if __name__ == '__main__':
    main()
