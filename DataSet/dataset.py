
import torch.utils.data as data
import numpy as np
import torch
from Constants import Constants
from random import shuffle
import itertools



def my_clloect_fn_train(batch):
    # btach = list(zip(*batch))
    # return batch
    batch = [item for item in batch if item[-1][0] > 0 and item[5][0] >= 0]
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
    batch = [item for item in batch if item[-1][0] > 0]
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


class rankDataSet(data.Dataset):
    def __init__(self,
                 G,
                 args,
                 question_id_list,
                 user_context=None,
                 is_training=True
                 ):
        self.G = G
        self.args = args
        self.question_id_list = question_id_list
        self.user_context = user_context
        self.is_training = is_training

    def random_negative(self, questionId, userId, score):

        user_list = [user for user in self.G.neighbors(questionId) if self.G[questionId][user]['score'] < score]
        if len(user_list) < 1:
            return -1
        while (user_list[0] == userId):
            shuffle(user_list)

        return user_list[0]

    def __len__(self):

        return len(self.question_id_list)

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
                    user_neg_list.append(self.user_context[user_neg])
                else:
                    user_neg_list.append(user_neg)

            question_list.append(question)
            answer_pos_list.append(answer_pos)
            if self.user_context is not None:
                user_pos_list.append(self.user_context[user_pos])
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
                 user_context=None
                 ):
        self.G = G
        self.args = args
        self.question_list = question_list
        self.user_context = user_context



    def __len__(self):
        return len(self.question_list)

    def __getitem__(self,   idx):
        question_id = self.question_list[idx]
        users = self.G.neighbors(question_id)
        question_list = []
        user_list = []
        label_list = []
        answer_list = []
        for user in users:
            answer = self.G[question_id][user]['a_id']
            label = self.G[question_id][user]['score']
            question_list.append(question_id)
            if self.user_context is None:
                user_list.append(user)
            else:
                user_list.append(self.user_context[user])
            answer_list.append(answer)
            label_list.append(label)

        return question_list, answer_list, user_list, label_list, [len(question_list)]