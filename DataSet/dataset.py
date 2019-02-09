
import torch.utils.data as data
import numpy as np
import torch
from Constants import Constants
from random import shuffle





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
            return question_list, answer_pos_list, user_pos_list, score_pos_list, answer_neg_list, user_neg_list, score_neg_list
        else:
            return question_list, answer_pos_list, user_pos_list, score_pos_list



class clasifyDataSet(data.Dataset):

    def __init__(self,
                 G, user_count,
                 args,
                 question_count = -1,
                 is_classification=True,
                 is_training=True
                 ):
        self.G = G
        self.args = args
        self.user_count = user_count
        self.edges = self.train_edge() if is_training else self.val_edge()
        self.is_classification = is_classification
        self.question_count = question_count

    def train_edge(self):
        return [e for e in self.G.edges(data=True) if not self.G[e[0]][e[1]]['train_removed']]

    def val_edge(self):
        return [e for e in self.G.edges(data=True) if self.G[e[0]][e[1]]['train_removed']]



    def __len__(self):

        return len(self.edges)

    def __getitem__(self, idx):
        edge = self.edges[idx]
        question = max(edge[0:2])
        user_p = min(edge[0:2])
        answer_p = self.G[edge[0]][edge[1]]['a_id']
        score_p = self.G[edge[0]][edge[1]]['score']

        return question, answer_p, user_p, score_p