
import torch.utils.data as data
import numpy as np
import torch
from Constants import Constants
from random import shuffle
import itertools



def my_clloect_fn(batch):
    return tuple(torch.LongTensor(list(itertools.chain.from_iterable(i))) for i in batch)


class rankDataSet(data.Dataset):
    def __init__(self,
                 G,
                 args,
                 question_id_list,
                 is_training=True
                 ):
        self.G = G
        self.args = args
        self.question_id_list = question_id_list
        self.is_training = is_training

    def random_negative(self, questionId, userId, score):

        user_list = [self.G[questionId][user]['score'] < score for user in self.G.neighbors(questionId)]
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
        user_list = self.G.neighbors(question)


        #generate negative answer
        for user_pos in user_list:
            answer_pos = self.G[question][user_pos]['a_id']
            score_pos = self.G[question][user_pos]['score']
            user_neg = self.random_negative(question, user_pos, score_pos)

            if self.is_training:
                if(user_neg == -1):
                    continue
                answer_n = self.G[question][user_neg]['a_id']
                score_neg = self.G[question][user_neg]['score']
                answer_neg_list.append(answer_n)
                score_neg_list.append(score_neg)
                user_neg_list.append(user_neg)



            question_list.append(question)
            answer_pos_list.append(answer_pos)
            user_pos_list.append(user_pos)
            score_pos_list.append(score_pos)

        if self.is_training:
            return question_list, answer_pos_list, user_pos_list, score_pos_list, answer_neg_list, user_neg_list, score_neg_list, [len(question_list)]
        else:
            # index = np.argsort(-1 * np.array(score_pos_list))
            # answer_pos_list = list(np.array(answer_pos_list)[index])
            # user_pos_list = list(np.array(user_pos_list)[index])
            # score_pos_list = list(np.array(score_pos_list)[index])
            return question_list, answer_pos_list, user_pos_list, score_pos_list, [len(question_list)]



class clasifyDataSet(data.Dataset):

    def __init__(self,
                 G,
                 args,
                 question_list
                 ):
        self.G = G
        self.args = args
        self.question_list = question_list




    def __len__(self):
        return len(self.question_list)

    def __getitem__(self, idx):
        question_id = self.question_list[idx]
        users = self.G.neighbors[question_id]
        question_list = []
        user_list = []
        label_list = []
        answer_list = []
        for user in users:
            answer = self.G[question_id][user]['a_id']
            label = self.G[question_id][user]['score']
            question_list.append(question_id)
            user_list.append(user)
            answer_list.append(answer)
            label_list.append(label)

        return question_list, answer_list, label_list, user_list, [len(question_list)]