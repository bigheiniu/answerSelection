
import  torch.utils.data as data
import numpy as np
import torch
from Constants import Constants


# def paired_collate_fn(insts):
#     question_content, answer_content, users, label = list(zip(*insts)
# )
#     question_content = collate_fn(question_content)
#     answer_content = collate_fn(answer_content)
#     users = users
#     label = collate_fn(label, label=True)
#     return question_content, answer_content, user_contex_list, label
#
# def collate_fn(insts, label=False):
#     ''' Pad the instance to the max seq length in batch '''
#     if (label):
#         return torch.LongTensor(insts)
#     max_len = max(len(inst) for inst in insts)
#
#     batch_seq = np.array([
#         inst + [Constants.PAD] * (max_len - len(inst))
#         for inst in insts
#     ])
#
#
#     batch_seq = torch.LongTensor(batch_seq)
    # batch_pos = torch.LongTensor(batch_pos)

    # return batch_seq




class clasifyDataSet(data.Dataset):

    def __init__(self,
                 G, user_count,
                 args,
                 Istraining=True
                 ):
        super(clasifyDataSet, self).__init__()
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

