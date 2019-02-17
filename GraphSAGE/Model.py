from GraphSAGE.NeighSampler import UniformNeighborSampler
from GraphSAGE.layer import *
from Util import *
from GraphSAGE import DPP

class InducieveLearningQA(nn.Module):
    def __init__(self, args,
                 user_count,
                 adj,
                 adj_edge,
                 content,
                 word2vec,
                 need_diversity=False
                 ):
        super(InducieveLearningQA, self).__init__()
        self.args = args
        self.adj = adj
        self.adj_edge = adj_edge
        self.user_count = user_count
        self.need_diversity = need_diversity

        ##############
        #  network structure init
        ##############
        self.user_embed = nn.Embedding(user_count, args.user_embed_dim)
        self.content = content
        self.word2vec_embed = nn.Embedding.from_pretrained(word2vec)

        self.lstm = LSTM(args)

        self.sampler = UniformNeighborSampler(self.adj, self.adj_edge)
        self.q_aggregate = AttentionAggregate(args.lstm_hidden_size, args.lstm_hidden_size, args.lstm_hidden_size)
        self.u_aggregate = AttentionAggregate(args.lstm_hidden_size, args.lstm_hidden_size, args.lstm_hidden_size)
        self.q_node_generate = NodeGenerate(args.lstm_hidden_size)
        self.u_node_generate = NodeGenerate(args.lstm_hidden_size)
        self.a_edge_generate = EdgeGenerate()

        self.w_q = nn.Linear(args.lstm_hidden_size, args.lstm_hidden_size)
        self.w_a = nn.Linear(args.lstm_hidden_size, args.lstm_hidden_size)
        self.w_u = nn.Linear(args.lstm_hidden_size, args.lstm_hidden_size)


        if self.args.is_classification:
            self.w_final = nn.Linear(args.lstm_hidden_size, args.num_class)
        else:
            self.w_final = nn.Linear(args.lstm_hidden_size, 1)




    def content_embed(self, batch_id):
        shape = [*batch_id.shape]
        shape.append(len(self.content[0]))
        shape = tuple(shape)
        batch_id = batch_id.view(-1, )
        content = self.content[batch_id]
        content = content.view(shape)
        return content



    def neighbor_sample(self, item, depth, neighbor_count_list):
        neighbor_node = []
        neighbor_edge = []
        neighbor_node.append(item)

        for i in range(depth):
            neighbor_node_layer, neighbor_edge_layer = self.sampler.sample(neighbor_node[i], neighbor_count_list[i])
            neighbor_node.append(neighbor_node_layer)
            neighbor_edge.append(neighbor_edge_layer)

        return neighbor_node, neighbor_edge


    def diversity_recomend(self, relevance_score_list, feature_matrix):
        relevance_score_list = tensorTonumpy(relevance_score_list, self.args.cuda)
        rankList = np.argsort(-relevance_score_list)
        feature_matrix = tensorTonumpy(feature_matrix, self.args.cuda)
        candidate_answer = DPP.diversity(feature_matrix, relevance_score_list, rankList, self.args.dpp_early_stop)
        return candidate_answer



    def forward(self, question, answer_edge, user, need_feature=False):
        #sample neighbors
        # q <- a -> u <- a -> u
        # u <- a -> q <- a -> q
        question_neighbors, question_neighbors_edge = self.neighbor_sample(question, self.args.depth, self.args.neighbor_number_list)
        user_neighbors, user_neigbor_edge = self.neighbor_sample(user, self.args.depth, self.args.neighbor_number_list)

        depth = len(question_neighbors)
        #load embedding
        for i in range(depth):
            if i % 2 == 0:
                question_embed = self.content_embed(question_neighbors[i] - self.user_count)
                question_embed_word2vec = self.word2vec_embed(question_embed)
                question_lstm_embed = self.lstm(question_embed_word2vec)
                question_neighbors[i] = question_lstm_embed


                user_neighbors[i] = self.user_embed(user_neighbors[i])
            else:
                question_neighbors[i] = self.user_embed(question_neighbors[i])

                question_embed = self.content_embed(user_neighbors[i] - self.user_count)
                question_embed_word2vec = self.word2vec_embed(question_embed)
                question_lstm_embed = self.lstm(question_embed_word2vec)
                user_neighbors[i] = question_lstm_embed

        for i in range(depth-1):
            question_edge_embed = self.content_embed(question_neighbors_edge[i] - self.user_count)
            question_edge_word2vec = self.word2vec_embed(question_edge_embed)
            question_edge_lstm = self.lstm(question_edge_word2vec)
            question_neighbors_edge[i] = question_edge_lstm

            user_edge_embed = self.content_embed(user_neigbor_edge[i] - self.user_count)
            user_edge_word2vec = self.word2vec_embed(user_edge_embed)
            user_edge_lstm = self.lstm(user_edge_word2vec)
            user_neigbor_edge[i] = user_edge_lstm

        answer_embed_layer = self.content_embed(answer_edge - self.user_count)
        answer_embed_word2vec = self.word2vec_embed(answer_embed_layer)
        answer_lstm_embed = self.lstm(answer_embed_word2vec)
        answer_edge_feaure = answer_lstm_embed




        #aggregate
        for i in range(depth - 1):
            layer_no = depth - i - 1
            if layer_no % 2 == 0:
                question_layer = question_neighbors[layer_no]
                question_edge = question_neighbors_edge[layer_no - 1]
                question_edge = self.a_edge_generate(question_edge, question_layer, question_neighbors[layer_no - 1])
                question_neighbor_feature = self.u_aggregate(question_layer, question_edge, question_neighbors[layer_no - 1])
                question_neighbors[layer_no - 1] = self.u_node_generate(question_neighbors[layer_no - 1], question_neighbor_feature)

                user_layer = user_neighbors[layer_no]
                user_edge = user_neigbor_edge[layer_no - 1]
                # update the edge based on two sides of nodes
                user_edge = self.a_edge_generate(user_edge, user_layer, user_neighbors[layer_no - 1])
                user_neigbor_feature = self.q_aggregate(user_layer, user_edge, user_neighbors[layer_no - 1])
                user_neighbors[layer_no - 1] = self.q_node_generate(user_neighbors[layer_no - 1], user_neigbor_feature)

            else:
                user_layer = question_neighbors[layer_no]
                user_edge = question_neighbors_edge[layer_no - 1]
                user_edge = self.a_edge_generate(user_edge, user_layer, question_neighbors[layer_no - 1])
                user_neighbor_feature = self.q_aggregate(user_layer, user_edge, question_neighbors[layer_no-1])
                question_neighbors[layer_no - 1] = self.q_node_generate(question_neighbors[layer_no - 1],
                                                                    user_neighbor_feature)

                question_layer = user_neighbors[layer_no]
                question_edge = user_neigbor_edge[layer_no - 1]
                question_edge= self.a_edge_generate(question_edge, question_layer, user_neighbors[layer_no - 1])
                question_neigbor_feature = self.q_aggregate(question_layer, question_edge, user_neighbors[layer_no-1])

                user_neighbors[layer_no - 1] = self.q_node_generate(user_neighbors[layer_no - 1], question_neigbor_feature)

        #score edge strength
        score = torch.tanh(self.w_a(answer_edge_feaure) + self.w_q(question_neighbors[0]) + self.w_u(user_neighbors[0]))
        if self.args.is_classification:
            score = F.log_softmax(self.w_final(score), dim=-1)
            predic = torch.argmax(score, dim=-1)
            return_list = [score, predic]
        else:
            score = self.w_final(score).view(-1,)
            return_list = [score]
        if need_feature:
            feature_matrix = self.w_a(answer_edge_feaure) + self.w_q(question_neighbors[0]) + self.w_u(user_neighbors[0])
            return_list.append(feature_matrix)

        return tuple(return_list)


        #
        # if self.training or not self.need_diversity:
        #     if self.args.is_classification:
        #         if predic == -1:
        #             exit(-1)
        #         return score, predic
        #     else:
        #         return score
        # else:
        #     feature_matrix = self.w_a(answer_edge_feaure) + self.w_q(question_neighbors[0]) + self.w_u(user_neighbors[0])
        #     candidate_answer_list = []
        #     temp = 0
        #     for i in count_list:
        #         feature_sub_matrix = feature_matrix[temp:temp + i]
        #         relevance_sub_list = score[temp:temp+i]
        #         candidate_answer_index = self.diversity_recomend(feature_sub_matrix, relevance_sub_list)
        #         candidate_answer_id_order = answer_edge[candidate_answer_index]
        #         candidate_answer_list.append(candidate_answer_id_order)
        #     if self.args.is_classification:
        #         #tensor, tensor, numpy, numpy
        #         return score, predic, tensorTonumpy(question,self.args.gpu), candidate_answer_list



