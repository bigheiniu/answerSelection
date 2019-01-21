import torch
import torch.nn as nn
from Layer.NeighSampler import UniformNeighborSampler
from Layer.Aggregate import *

class InducieveLearning(nn.Module):
    def __init__(self, args,
                 user_count,
                 adj,
                 adj_edge,
                 adj_score
                 ):
        super(InducieveLearning, self).__init__()
        self.args = args
        self.adj = adj
        self.adj_edge = adj_edge

        ##############
        #  network structure init
        ##############
        self.user_embed = nn.Embedding(user_count, args.user_embed_dim)
        self.lstm = nn.LSTM(args.lstm_input_dim, args.lstm_output_dim, batch_first=True)

        self.sampler = UniformNeighborSampler(self.adj, self.adj_edge)
        self.q_aggregate = Aggregate(args.lstm_output_dim, args.lstm_output_dim, args.lstm_output_dim)
        self.u_aggregate = Aggregate(args.lstm_output_dim, args.lstm_output_dim, args.lstm_output_dim)
        self.q_node_generate = NodeGenerate(args.lstm_output_dim)
        self.u_node_generate = NodeGenerate(args.lstm_output_dim)
        self.a_edge_generate = EdgeGenerate()

        self.w_q = nn.Linear(args.lstm_output_dim, args.lstm_output_dim)
        self.w_a = nn.Linear(args.lstm_output_dimm, args.lstm_output_dim)
        self.w_u = nn.Linear(args.lstm_output_dim, args.lstm_output_dim)
        self.w_final = nn.Linear(args.lstm_output_dim, args.lstm_output_dim)



    def neighbor_agg(self, item_feature, type):
        neighbor_node, neighbor_edge = self.sampler(item_feature)
        if type == "q":
            neighbor_feature = self.q_aggregate(neighbor_node, neighbor_edge)
        else:
            neighbor_feature = self.u_aggregate(neighbor_node, neighbor_edge)

        return neighbor_feature


    def neighbor_sample(self, item, depth, neighbor_number_list):
        neighbor_node = []
        neighbor_edge = []
        current = item
        for i in range(depth):
            neighbor_node_layer, neighbor_edge_layer = self.sampler(current, neighbor_number_list[i])
            neighbor_node.append(neighbor_node_layer)
            neighbor_edge.append(neighbor_edge_layer)
            current = neighbor_node_layer

        return neighbor_node, neighbor_edge



    def lstm_init(self):
        h_0_size_1 = 1
        if self.args.bidirectional:
            h_0_size_1 *= 2
        h_0_size_1 *= self.args.lstm_num_layers
        hiddena = torch.zeros((h_0_size_1, self.args.batch_size, self.args.lstm_hidden_size),
                              dtype=torch.float, device=self.args.device)
        hiddenb = torch.zeros((h_0_size_1, self.args.batch_size, self.args.lstm_hidden_size),
                              dtype=torch.float, device=self.args.device)
        return hiddena, hiddenb


    def forward(self, question, answer_edge, user):
        question_neighbors, question_neighbors_edge = self.neighbor_sample(question, self.args.depth, self.args.neighbor_number_list)
        user_neighbors, user_neigbor_edge = self.neighbor_sample(user, self.args.depth, self.args.neighbor_number_list)

        return


