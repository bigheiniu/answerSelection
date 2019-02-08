
import torch.utils.data as data
import numpy as np
import torch
from Constants import Constants
from random import shuffle






class clasifyDataSet(data.Dataset):

    def __init__(self,
                 G, user_count,
                 args,
                 is_classification=True,
                 is_training=True
                 ):
        self.G = G
        self.args = args
        self.user_count = user_count
        self.edges = self.train_edge() if is_training else self.val_edge()
        self.is_classification = is_classification

    def train_edge(self):
        return [e for e in self.G.edges(data=True) if not self.G[e[0]][e[1]]['train_removed']]

    def val_edge(self):
        return [e for e in self.G.edges(data=True) if self.G[e[0]][e[1]]['train_removed']]


    def random_negative(self, questionId, userId, score):

        user_list = [self.G[questionId][user]['score'] < score for user in self.G.neighbors(questionId)]

        while (user_list[0] == userId):
            shuffle(user_list)

        return user_list[0]


    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        edge = self.edges[idx]
        question = max(edge[0:2])
        user_p = min(edge[0:2])
        answer_p = self.G[edge[0]][edge[1]]['a_id']
        score_p = self.G[edge[0]][edge[1]]['score']

        if self.is_classification:
            return question, answer_p, user_p, score_p

        else:
            user_n = self.random_negative(question, user_p, score_p)
            answer_n = self.G[question][user_n]['a_id']
            score_n = self.G[question][user_n]['score']

            # For pairwise hinge loss function
            return question, answer_p, user_p, score_p, user_n, answer_n, score_n

