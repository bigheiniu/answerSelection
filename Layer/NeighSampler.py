
import torch
import torch.nn as nn

import numpy as np

class UniformNeighborSampler(torch.nn.Module):
    '''
    Uniformly sample neighbors.
    TODO: Assume that adj lists are padded with random re-sampling
    '''
    def __init__(self, adj_neighbor:torch.LongTensor, adj_edge: torch.LongTensor):
        super(UniformNeighborSampler, self).__init__()
        self.neighbor_embed = nn.Embedding.from_pretrained(adj_neighbor)
        self.edge_embed = nn.Embedding.from_pretrained(adj_edge)

    def forward(self, batch_ids, number):
        '''

        :param batch_ids: batch of node ids
        :param number: how many number should be count
        :return: return next layer neighbors and the edge connecting them
        '''

        neighbor_next_layer = self.neighbor_embed(batch_ids)[:,:number]
        edge_next_layer = self.edge_embed(batch_ids)[:,:number]

        return neighbor_next_layer, edge_next_layer
