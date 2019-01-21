import torch
import torch.nn as nn
import numpy as np

class contentEmbed(nn.Module):
    def __init__(self, content:torch.LongTensor):
        super(contentEmbed, self).__init__()
        self.contentEmbed = nn.Embedding.from_pretrained(content)


    def forward(self, batch_id):
        return self.contentEmbed(batch_id)


def Adjance(G, max_degree, user_count):
    nodes_count = len(G.nodes())
    adj = np.zeros((nodes_count + 1, max_degree), dtype=np.int64)
    adj_edge = np.zeros_like(adj, dtype=np.int64)
    adj_score = np.zeros_like(adj, dtype=np.float)
    adj_degree = np.zeros_like(nodes_count,)

    for node in G.nodes():
        assert isinstance(node, int), "[ERROR] Argument of node should be int"
        neighbors = np.array([neighbor for neighbor in G.neighbors[node]])
        adj_degree[node] = len(neighbors)

        if len(neighbors) == 0:
            continue
        if len(neighbors) < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)
        else:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)

        for index, neigh_node in enumerate(neighbors):
            adj_edge[node, index] = G[node][neigh_node]['a_id']
            adj_score[node, index] = G[node][neigh_node]['score']

        adj[node, :] = neighbors
    adj = torch.LongTensor(adj)
    adj_edge = torch.LongTensor(adj_edge)
    adj_score = torch.FloatTensor(adj_score)
    #all the return are tensor
    return adj, adj_edge, adj_score