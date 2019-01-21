import torch
import torch.nn as nn
import torch.nn.functional as F

class Aggregate(torch.nn.Module):
    def __init__(self, input1_dim,
                 input2_dim, bilinear_output_dim
                 ):
        super(Aggregate, self).__init__()
        self.biliear = nn.Bilinear(input1_dim, input2_dim, bilinear_output_dim)

    def forward(self, neighbors, edges):
        middle = self.biliear(neighbors, edges)
        middle_act = F.relu(middle)
        #max-pooling
        result = torch.max(middle_act, dim=1)
        return result


class NodeGenerate(torch.nn.Module):
    def __init__(self, input_dim):
        super(NodeGenerate, self).__init__()
        self.linear = nn.Linear(2*input_dim, input_dim)
        self.normal = nn.BatchNorm1d(input_dim)


    def forward(self, item, neighbor_agg):
        '''

        :param item: batch * feature
        :param neighbor_feature: batch * feature
        :return:
        '''
        concat = torch.cat((item, neighbor_agg), dim = -1)
        result = self.linear(concat)
        nn.ReLU(result, True)
        result = self.normal(result)
        return result

class EdgeGenerate(torch.nn.Module):
    def __init__(self):
        super(EdgeGenerate, self).__init__()

    def forward(self, edges, questions, users):
        return edges



