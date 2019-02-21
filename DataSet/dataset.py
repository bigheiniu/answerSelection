
import torch.utils.data as data
import numpy as np
import torch
from Constants import Constants
from random import shuffle
import itertools
from scipy.stats import rankdata



def my_clloect_fn_train(batch):
    # btach = list(zip(*batch))
    # return batch
    try:
        batch = [item for item in batch if item[-1][0] > 5 and item[5][0] >= 0]
    except:
        batch = [item for item in batch if item[-1][0] > 5 and len(item[5][0]) >= 0]
    question_list = torch.LongTensor(list(itertools.chain.from_iterable([item[0]for item in batch])))
    answer_pos_list = torch.LongTensor(list(itertools.chain.from_iterable([item[1] for item in batch])))
    user_pos_list = torch.LongTensor([x for item in batch for x in item[2]])
    score_pos_list = torch.FloatTensor(list(itertools.chain.from_iterable([item[3] for item in batch])))
    answer_neg_list = torch.LongTensor(list(itertools.chain.from_iterable([item[4] for item in batch])))
    user_neg_list = torch.LongTensor([x for item in batch for x in item[5]])
    score_neg_list = torch.FloatTensor(list(itertools.chain.from_iterable([item[6] for item in batch])))
    count_list = torch.IntTensor(list(itertools.chain.from_iterable([item[7] for item in batch])))
    return question_list, answer_pos_list, user_pos_list, score_pos_list, answer_neg_list, user_neg_list, score_neg_list, count_list

def my_collect_fn_test(batch):

    batch = [item for item in batch if item[-1][0] > 5]

    question_list = torch.LongTensor(list(itertools.chain.from_iterable([item[0]for item in batch])))
    answer_list = torch.LongTensor(list(itertools.chain.from_iterable([item[1] for item in batch])))
    user_list = torch.LongTensor([x for item in batch for x in item[2]])
    score_list = torch.FloatTensor(list(itertools.chain.from_iterable([item[3] for item in batch])))
    count_list = torch.IntTensor(list(itertools.chain.from_iterable([item[4] for item in batch])))
    return question_list, answer_list, user_list, score_list, count_list

def classify_collect_fn(batch):
    question_list = torch.LongTensor(list(itertools.chain.from_iterable([item[0] for item in batch])))
    answer_list = torch.LongTensor(list(itertools.chain.from_iterable([item[1] for item in batch])))
    user_list = torch.LongTensor([x for item in batch for x in item[2]])
    label_list = torch.LongTensor(list(itertools.chain.from_iterable([item[3] for item in batch])))
    count_list = torch.IntTensor(list(itertools.chain.from_iterable([item[4] for item in batch])))
    return question_list, answer_list, user_list, label_list, count_list



class rankDataSetEdge(data.Dataset):
    def __init__(self,
                 G,
                 args,
                 answer_score,
                 question_id_list,
                 question_count=None,
                 user_context=None,
                 content=None,
                 user_count=None,
                 is_training=True,
                 answer_user_dic = None
                 ):
        self.G = G
        self.args = args
        self.question_id_list = question_id_list
        self.user_context = user_context
        self.content = content
        self.is_training = is_training
        self.user_count = user_count
        self.question_count = question_count
        self.answer_score = answer_score
        self.answer_user_dic = answer_user_dic
        self.rank_score, self.rank_index = self.rankAnswer(answer_score)
        self.edges = self.edgeGenre()

    def edgeGenre(self):
        edges = []
        for edge in self.G.edges(data=True):
            if self.is_training and ~edge['train_removed']:
                edges.append(edge)
            elif ~self.is_training and edge['train_removed']:
                edges.append(edge)
        return edges

    def rankAnswer(self, answer_score):
        rank_score = [int(i) for i in rankdata(answer_score, method='ordinal')]
        rank_dic = {rank:index for index, rank in enumerate(rank_score) }
        rank_index = [ index for _, index in  sorted(rank_dic.items(), key=lambda x: x[0])]
        return rank_score, rank_index

    def negative_sampling(self, answerid):
        rank = self.rank_score[answerid - self.user_count - self.question_count]
        negative_answer_candidate = self.rank_index[:rank]
        negative_score = self.answer_score[self.rank_index[:rank]]
        negative_pro = negative_score / np.sum(negative_score)
        if len(negative_answer_candidate) > self.args.neg_size:
            negative_answer = np.random.choice(negative_answer_candidate, self.args.neg_size, replace=False, p=negative_pro)
        else:
            negative_answer = np.random.choice(list(range(len(self.answer_score))), self.args.neg_size, replace=False, p=self.answer_score/np.sum(self.answer_score))

        return negative_answer + self.user_count + self.question_count

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, edgeidx):
        edge = self.edges[edgeidx]
        question = edge[0] if edge[0] > edge[1] else edge[1]
        user = edge[1] if edge[0] > edge[1] else edge[0]

        answer = edge['a_id']
        score = edge['score']
        if self.is_training:
            neg_ans = self.negative_sampling(answer)
            neg_user = self.answer_user_dic[neg_ans]

            return question, user, score, neg_ans, neg_user

        else:
            return question, user, score


class rankDataSetUserContext(data.Dataset):
    def __init__(self,
                 args,
                 answer_score,
                 question_answer_user_vote,
                 question_count=None,
                 user_context=None,
                 content=None,
                 user_count=None,
                 is_training=True,
                 answer_user_dic=None
                 ):
        self.args = args
        self.user_context = user_context
        self.content = content
        self.is_training = is_training
        self.user_count = user_count
        self.question_count = question_count
        self.question_answer_user_vote = question_answer_user_vote
        self.answer_user_dic = answer_user_dic
        self.answer_score = answer_score
        self.rank_score, self.rank_index = self.rankAnswer(answer_score)



    def rankAnswer(self, answer_score):
        rank_score = [int(i) for i in rankdata(answer_score, method='ordinal')]
        rank_dic = {rank:index for index, rank in enumerate(rank_score) }
        rank_index = [ index for _, index in  sorted(rank_dic.items(), key=lambda x: x[0])]
        return rank_score, rank_index

    def negative_sampling(self, answerid):
        rank = self.rank_score[answerid - self.user_count - self.question_count]
        negative_answer_candidate = self.rank_index[:rank]
        negative_score = self.answer_score[self.rank_index[:rank]]
        negative_pro = negative_score / np.sum(negative_score)
        if len(negative_answer_candidate) > self.args.neg_size:
            negative_answer = np.random.choice(negative_answer_candidate, self.args.neg_size, replace=False, p=negative_pro)
        else:
            negative_answer = np.random.choice(list(range(len(self.answer_score))), self.args.neg_size, replace=False, p=self.answer_score/np.sum(self.answer_score))

        return negative_answer + self.user_count + self.question_count

    def get_user_context(self, userid):
        document = []
        for answer_id in self.user_context[userid]:
            document += self.content.content_embed(answer_id - self.user_count)
            if len(document) > self.args.max_u_len:
                document = document[:self.args.max_u_len]
                break
        if len(document) < self.args.max_u_len:
            pad_word = [Constants.PAD] * (self.args.max_u_len - len(document))
            document += pad_word
        return document


    def __len__(self):
        return len(self.question_answer_user_vote)

    def __getitem__(self, index):
        question_answer_vote_line = self.question_answer_user_vote[index]
        question = question_answer_vote_line[0]
        answer = question_answer_vote_line[1]
        user_context = self.get_user_context(question_answer_vote_line[2])
        score = question_answer_vote_line[3]
        if self.is_training:
            neg_ans = self.negative_sampling(answer)
            neg_user_context = self.get_user_context(self.answer_user_dic[neg_ans])

            return question, answer, user_context, score, neg_ans, neg_user_context
        else:
            return question, answer, user_context, score










class rankData(data.Dataset):
    def __init__(self,
                 G,
                 args,
                 answer_score,
                 question_answer_user_vote,
                 question_count=None,
                 user_context=None,
                 content=None,
                 user_count=None,
                 is_training=True,
                 answer_user_dic=None
                 ):
        self.args = args
        self.user_context = user_context
        self.content = content
        self.is_training = is_training
        self.user_count = user_count
        self.question_count = question_count
        self.question_answer_user_vote = question_answer_user_vote
        self.answer_user_dic = answer_user_dic
        self.answer_score = answer_score
        self.rank_score, self.rank_index = self.rankAnswer(answer_score)



    def rankAnswer(self, answer_score):
        rank_score = [int(i) for i in rankdata(answer_score, method='ordinal')]
        rank_dic = {rank:index for index, rank in enumerate(rank_score) }
        rank_index = [ index for _, index in  sorted(rank_dic.items(), key=lambda x: x[0])]
        return rank_score, rank_index

    def negative_sampling(self, answerid):
        rank = self.rank_score[answerid - self.user_count - self.question_count]
        negative_answer_candidate = self.rank_index[:rank]
        negative_score = self.answer_score[self.rank_index[:rank]]
        negative_pro = negative_score / np.sum(negative_score)
        if len(negative_answer_candidate) > self.args.neg_size:
            negative_answer = np.random.choice(negative_answer_candidate, self.args.neg_size, replace=False, p=negative_pro)
        else:
            negative_answer = np.random.choice(list(range(len(self.answer_score))), self.args.neg_size, replace=False, p=self.answer_score/np.sum(self.answer_score))

        return negative_answer + self.user_count + self.question_count

    def random_negative(self, questionId, userId, score):

        user_list = [user for user in self.G.neighbors(questionId) if self.G[questionId][user]['score'] < score]
        if len(user_list) < 1:
            return -1
        while (user_list[0] == userId):
            shuffle(user_list)

        return user_list[0]





    def __len__(self):

        return len(self.question_answer_user_vote)

    def __getitem__(self, idx):
        question_list = []
        answer_pos_list = []
        user_pos_list = []
        score_pos_list = []

        answer_neg_list = []
        user_neg_list = []
        score_neg_list = []
        #1
        question = self.question_id_list[idx]
        # k * 1
        user_list = list(self.G.neighbors(question))

        #generate negative answer
        for user_pos in user_list:
            #ATTENTION: people answer the same question for multiple times
            answer_pos = self.G[question][user_pos]['a_id']
            score_pos = self.G[question][user_pos]['score']


            if self.is_training:
                user_neg = self.random_negative(question, user_pos, score_pos)
                if(user_neg == -1):
                    continue
                answer_n = self.G[question][user_neg]['a_id']
                score_neg = self.G[question][user_neg]['score']
                answer_neg_list.append(answer_n)
                score_neg_list.append(score_neg)
                if self.user_context is not None:
                    document = []
                    for neg_id in self.user_context[user_neg]:
                        document += self.content.content_embed(neg_id - self.user_count)
                        if len(document) > self.args.max_u_len:
                            document = document[:self.args.max_u_len]
                            break
                    if len(document) < self.args.max_u_len:
                        pad_word = [Constants.PAD] * (self.args.max_u_len - len(document))
                        document += pad_word
                    user_neg_list.append(document)
                else:
                    user_neg_list.append(user_neg)

            question_list.append(question)
            answer_pos_list.append(answer_pos)
            if self.user_context is not None:
                document = []
                for post_id in self.user_context[user_pos]:
                    document += self.content.content_embed(post_id - self.user_count)
                    if len(document) > self.args.max_u_len:
                        document = document[:self.args.max_u_len]
                        break
                if len(document) < self.args.max_u_len:
                    pad_word = [Constants.PAD] * (self.args.max_u_len - len(document))
                    document += pad_word
                user_pos_list.append(document)
            else:
                user_pos_list.append(user_pos)
            score_pos_list.append(score_pos)

        if self.is_training:
            return question_list, answer_pos_list, user_pos_list, score_pos_list, answer_neg_list, user_neg_list, score_neg_list,[len(question_list)]
        else:
            return question_list, answer_pos_list, user_pos_list, score_pos_list, [len(question_list)]



class clasifyDataSet(data.Dataset):

    def __init__(self,
                 G,
                 args,
                 question_list,
                 user_context=None,
                 content=None,
                 user_count = None
                 ):
        self.G = G
        self.args = args
        self.question_used = question_list
        self.user_context = user_context
        self.content = content
        self.user_count = user_count



    def __len__(self):
        return len(self.question_used)

    def __getitem__(self,   idx):
        question_id = self.question_used[idx]
        users = self.G.neighbors(question_id)

        question_list = []
        user_list = []
        label_list = []
        answer_list = []
        for user in users:
            answer = self.G[question_id][user]['a_id']
            label = self.G[question_id][user]['score']
            question_list.append(question_id)
            if self.user_context is not None:
                document = []
                for post_id in self.user_context[user]:
                    document += self.content.content_embed(post_id - self.user_count)
                    if len(document) > self.args.max_u_len:
                        document = document[:self.args.max_u_len]
                        break
                if len(document) < self.args.max_u_len:
                    pad_word = [Constants.PAD] * (self.args.max_u_len - len(document))
                    document += pad_word
                user_list.append(document)
            else:
                user_list.append(user)
            answer_list.append(answer)
            label_list.append(label)

        return question_list, answer_list, user_list, label_list, [len(question_list)]