
import torch.utils.data as data
import numpy as np
import torch
from Constants import Constants





class clasifyDataSet(data.Dataset):

    def __init__(self,
                 G, user_count,
                 args,
                 Istraining=True
                 ):
        self.G = G
        self.args = args
        self.user_count = user_count
        self.edges = self.train_edge() if Istraining else self.val_edge()

    def train_edge(self):
        return [e for e in self.G.edges(data=True) if not self.G[e[0]][e[1]]['train_removed']]

    def val_edge(self):
        return [e for e in self.G.edges(data=True) if self.G[e[0]][e[1]]['train_removed']]

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        edge = self.edges[idx]
        question = max(edge[0:2])
        user = min(edge[0:2])
        answer = self.G[edge[0]][edge[1]]['a_id']
        score = self.G[edge[0]][edge[1]]['score']

        return question, answer, user, score

