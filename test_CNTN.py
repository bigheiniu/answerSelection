import argparse
from tqdm import tqdm
#pytorch import
from Util import *

from CNTN import Model as CNTN_Model

from DataSet.dataset import rankDataOrdinary, classifyDataOrdinary, my_collect_fn_test, my_clloect_fn_train, classify_collect_fn
from Metric.coverage_metric import *
from Metric.rank_metrics import ndcg_at_k, average_precision, precision_at_k, mean_reciprocal_rank, Accuracy, marcoF1
from Config import config_model
import os
os.chdir("/home/yichuan/course/induceiveAnswer")




#grid search for paramter
from Visualization.logger import Logger

info = {}
logger = Logger('./logs_cntn')
i_flag = 0
train_epoch_count = 0
eval_epoch_count = 0

#WARNNING: cannot output label, can only output score
def prepare_dataloaders(data, args):
    # ========= Preparing DataLoader =========#
    train_data = data['question_answer_user_train']
    test_data = data['question_answer_user_test']
    question_count = data['question_count']
    user_count = data['user_count']
    content_embed = ContentEmbed(data['content'])
    answer_score = data['vote_sort']

    if args.is_classification:

        train_loader = torch.utils.data.DataLoader(
            classifyDataOrdinary(
                           args=args,
                        question_answer_user_vote=train_data,
                        content=content_embed,
                        user_count=user_count
                       ),
        num_workers=4,
        batch_size=args.batch_size,
        collate_fn=classify_collect_fn,
        shuffle=True
        )

        val_loader = torch.utils.data.DataLoader(
        classifyDataOrdinary(
            args=args,
            question_answer_user_vote=test_data,
            content=content_embed,
            user_count=user_count
        ),
        num_workers=4,
        batch_size=args.batch_size,
        collate_fn=classify_collect_fn,
        shuffle=True
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            rankDataOrdinary(
                args=args,
                question_answer_user_vote=train_data,
                is_training=True,
                user_count=user_count,
                answer_score=answer_score,
                content_embed= content_embed,
                question_count=question_count
            ),
            num_workers=4,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn= my_clloect_fn_train
        )

        val_loader = torch.utils.data.DataLoader(
            rankDataOrdinary(
                args=args,
                question_answer_user_vote=test_data,
                is_training=False,
                content_embed=content_embed,
                user_count=user_count,
                question_count=question_count,

            ),
            num_workers=4,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=my_collect_fn_test
        )

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

            q_iter, a_iter, gt_iter,_ = map(lambda x: x.to(args.device), batch)
            line_count += q_iter.shape[0]
            args.batch_size = q_iter.shape[0]
            optimizer.zero_grad()
            result = model(q_iter, a_iter)[0]
            loss = loss_fn(result, gt_iter)

        else:
            question_list, answer_pos_list, answer_neg_list = map(lambda x: x.to(args.device), batch)
            args.batch_size = question_list.shape[0]
            optimizer.zero_grad()
            score_pos = model(question_list, answer_pos_list)[0]
            score_neg = model(question_list, answer_neg_list)[0]
            loss = torch.sum(loss_fn(score_pos, score_neg))

        # logger.scalar_summary("train_loss", loss.item(), train_epoch_count)
        loss.backward()
        loss1 += loss.item()
        optimizer.step()




    for tag, value in model.named_parameters():
        if value.grad is None:
            continue
        tag = tag.replace('.', '/')
        logger.histo_summary(tag, value.cpu().detach().numpy(), train_epoch_count)
        logger.histo_summary(tag + '/grad', value.grad.cpu().numpy(),train_epoch_count)
    loss = loss / line_count
    logger.scalar_summary("train_loss", loss, train_epoch_count)


def eval_epoch(model, data, args, eval_epoch_count, tfidf_model, lda_model):
    model.eval()
    diversity_answer_recommendation = []
    questionid_answer_score_gt_dic = {}
    info_test = {}
    accuracy = 0
    one_count = 0
    zero_count = 0
    line_count = 0
    loss_fn = nn.NLLLoss()
    loss = 0
    predic_list = []
    gt_list = []
    with torch.no_grad():
        for batch in tqdm(
            data, mininterval=2, desc="  ----(validation)----  ", leave=True
        ):

            if args.is_classification:
                q_val, a_val, gt_val, question_id_list = map(lambda x: x.to(args.device), batch)
                args.batch_size = gt_val.shape[0]
                line_count += args.batch_size
                assert args.batch_size == gt_val.shape[0], "batch size is not eqaul {} != {}".format(args.batch_size, gt_val.shape[0])

                log_softmax, score, predic = model(q_val, a_val)
                loss += loss_fn(log_softmax, gt_val).item()

                score = tensorTonumpy(score, args.cuda)
                gt_val = tensorTonumpy(gt_val, args.cuda)
                question_id_list = tensorTonumpy(question_id_list, args.cuda)
                predic = tensorTonumpy(predic, args.cuda)
                # biggest in the beginning
                accuracy_temp, zero_count_temp, one_count_temp = Accuracy(gt_val, predic)
                predic_list += predic.tolist()
                gt_list += gt_val.tolist()
                accuracy += accuracy_temp
                zero_count += zero_count_temp
                one_count += one_count_temp
                assert one_count_temp + zero_count_temp == args.batch_size,"one count + zero count is not eqaul to batch size {} != {}".format(one_count_temp + zero_count_temp, args.batch_size)

                for questionid, gt, pred_score in zip(question_id_list, gt_val, score):
                    if questionid in questionid_answer_score_gt_dic:
                        questionid_answer_score_gt_dic[questionid].append([pred_score, gt])
                    else:
                        questionid_answer_score_gt_dic[questionid] = [[pred_score, gt]]


            else:
                q_val, a_val, gt_val, question_id_list = map(lambda x: x.to(args.device), batch)
                args.batch_size = gt_val.shape[0]
                line_count += args.batch_size
                assert args.batch_size == gt_val.shape[0], "batch size is not eqaul {} != {}".format(args.batch_size, gt_val.shape[0])
                score = model(q_val, a_val)[0]
                score = tensorTonumpy(score, args.cuda)
                gt_val = tensorTonumpy(gt_val, args.cuda)
                question_id_list = tensorTonumpy(question_id_list, args.cuda)
                a_val = tensorTonumpy(a_val, args.cuda)
                for questionid, answer_content, gt, pred_score in zip(question_id_list, a_val, gt_val, score):
                    if questionid in questionid_answer_score_gt_dic:
                        questionid_answer_score_gt_dic[questionid].append([answer_content, pred_score, gt])
                    else:
                        questionid_answer_score_gt_dic[questionid] = [[answer_content, pred_score, gt]]

    mAP = 0
    mRP = 0
    p_at_one = 0
    ndcg_loss = 0
    tf_idf_score = 0
    lda_score = 0
    for _, values in questionid_answer_score_gt_dic.items():
        #biggest in the begining
        answer_score_gt_reorder = sorted(values, key=lambda x: -x[-2])
        rank_gt = [line[-1] for line in answer_score_gt_reorder]
        if args.is_classification:
           mAP += average_precision(rank_gt)
           mRP += mean_reciprocal_rank([rank_gt])
           p_at_one += precision_at_k([rank_gt], args.precesion_at_k)
        else:
            rank_answer_content = [line[0] for line in answer_score_gt_reorder]
            tfidf_cov, lda_cov = diversity_evaluation([rank_answer_content], args.div_topK, tfidf_model, lda_model)
            tf_idf_score += tfidf_cov
            lda_score += lda_cov
            ndcg_loss += ndcg_at_k(rank_gt, args.ndcg_k)
            if np.argmax(rank_gt) == 0:
                p_at_one += 1
    question_count = len(questionid_answer_score_gt_dic)


    if args.is_classification:
        mAP = mAP*1.0 / question_count
        p_at_one = p_at_one*1.0 / question_count
        mRP = mRP *1.0 / question_count
        accuracy = accuracy * 1.0 / line_count
        loss = loss / question_count
        f1_score = marcoF1(y_gt = gt_list, y_pred=predic_list)
        # visualize the data
        info_test['mAP'] = mAP
        info_test['P@1'] = p_at_one
        info_test['mRP'] = mRP
        info_test['accuracy'] = accuracy
        info_test['f1_score'] = f1_score
        info_test['eval_loss'] = loss
        info_test['one_count'] = one_count
        info_test['zero_count'] = zero_count

        print("[Info] mAP: {}, P@1: {}, mRP: {}, Accuracy: {}, f1_score {}, loss {},one_count: {}, zero_count: {}".format(mAP, p_at_one, mRP, accuracy, f1_score, loss,one_count, zero_count))

    else:
        p_at_one = p_at_one * 1.0 / question_count
        ndcg_loss = ndcg_loss * 1.0 / question_count
        lda_score = lda_score * 1.0 / question_count
        tf_idf_score = tf_idf_score * 1.0 / question_count
        info_test['nDCGG'] = ndcg_loss
        info_test['P@1'] = p_at_one
        info_test['lda'] = lda_score
        info_test['tfidf'] = tf_idf_score

        print("[INFO] Ranking Porblem nDCGG: {}, p@1 is {}, lda score is {}, tf_idf score is {}".format(ndcg_loss, p_at_one, lda_score, tf_idf_score))



    for tag, value in info_test.items():
        logger.scalar_summary(tag, value, eval_epoch_count)

    return diversity_answer_recommendation


def train(args, train_data, val_data ,pre_trained_word2vec, content):
    model = CNTN_Model.CNTN(args, pre_trained_word2vec)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.to(args.device)
    #load coverage model
    lda=None
    tfidf=None
    if args.is_classification is False:
        cov_model_path = args.cov_model_path + "CNTN"
        tfidf = TFIDFSimilar(content, args.cov_pretrain, cov_model_path)
        lda = LDAsimilarity(content, args.lda_topic, args.cov_pretrain, cov_model_path)
    info_val = {}

    for epoch_i in range(args.epoch):

        train_epoch(model, train_data, optimizer, args, epoch_i)

        eval_epoch(model, val_data, args, epoch_i, tfidf_model=tfidf, lda_model=lda)
        # for tag, value in info_val.items():
        #     logger.scalar_summary(tag, value, epoch_i)



def main():

    #===========Load DataSet=============#
    datafoler = "data/"
    #"store_SemEval.torchpickle","tex.torchpickle", "apple.torchpickle",
    #,"tex.torchpickle", "apple.torchpickle","math.torchpickle"
    datasetname = [ "store_SemEval.torchpickle"]
    args = config_model
    for datan in datasetname:
        args.is_classification = True if "SemEval" in datan else False
        args.num_class = 2 if args.is_classification else 1
        args.data = datafoler + datan
        print("[FILE] Data file {}".format(datan))
        print("cuda : {}".format(args.cuda))
        data = torch.load(args.data)
        word2ix = data['dict']
        content = data['content']
        train_data, val_data = prepare_dataloaders(data, args)
        pre_trained_word2vec = loadEmbed(args.embed_fileName, args.embed_size, args.vocab_size, word2ix, args.DEBUG).to(args.device)
        train(args, train_data, val_data, pre_trained_word2vec, content)
        args.cov_pretrain = False
if __name__ == '__main__':
    main()
