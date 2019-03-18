from GraphSAGEDiv.NeighSampler import UniformNeighborSampler
from GraphSAGEDiv.LayerKey import *
from Util import *

class QuestinGenerate(nn.Module):
    def __init__(self, args,
                 user_count,
                 question_count,
                 content_embed,
                 user_embed,
                 word2vec
                 ):
        super(QuestinGenerate, self).__init__()
        self.args = args
        self.user_count = user_count
        self.hidden_state_size = self.args.lstm_hidden_size
        ##############
        #  network structure init
        ##############
        self.user_embed = user_embed
        # self.user_embed = nn.Embedding(user_count, self.hidden_state_size)
        self.content_embed = content_embed
        self.word2vec_embed = nn.Embedding.from_pretrained(word2vec)
        self.question_key_embedding = nn.Embedding(question_count, self.args.value_dim)

        #AGG
        self.agg = AttentionAggregate_Cos()
        self.que_generate = NodeGenerate_GRU_Forget_Gate(self.hidden_state_size, self.hidden_state_size)
        if self.args.is_classification:
            self.w_final = nn.Linear(self.hidden_state_size, args.num_class)
        else:
            self.w_final = nn.Linear(self.hidden_state_size, 1)
        self.cnn_lr = nn.Conv2d(1, self.hidden_state_size, (3, args.embed_size))
        self.bn = nn.BatchNorm1d(self.args.lstm_hidden_size)
        self.dropout = nn.Dropout(args.drop_out_lstm)

        #score function
        self.score_fn = ARMNLScore(self.hidden_state_size) if self.args.is_classification is False else WeightScore(self.hidden_state_size, self.args.num_class)

    def content_cnn(self, content):
        shape = content.shape
        content = content.view(-1, 1, shape[-2], shape[-1])
        content_cnn, _ = torch.max(torch.relu(self.cnn_lr(content)), dim=-2)
        content_cnn = self.bn(content_cnn)
        content_cnn = self.dropout(content_cnn)
        shape1 = []
        for i in range(len(shape) - 2):
            shape1.append(shape[i])
        shape1.append(self.hidden_state_size)
        content_cnn = content_cnn.view(shape1)
        return content_cnn





    def forward(self, question, question_key, question_neighbors_index, question_neighbors_key, answer_edge,user, need_feature=False):
        #sample neighbors
        # q <- a -> u <- a -> u
        # u <- a -> q <- a -> q

        question_value = self.content_cnn(self.word2vec_embed(self.content_embed.content_embed(question - self.user_count)))
        answer_value = self.content_cnn(self.word2vec_embed(self.content_embed.content_embed(answer_edge - self.user_count)))
        user_value = self.content_cnn(self.word2vec_embed(self.user_embed.content_embed(user)))
        # q_key = self.question_key_embedding(question - self.user_count)
        # q_nei_key = self.question_key_embedding(question_neighbors_index - self.user_count)
        question_neighbor_value = self.content_cnn(self.word2vec_embed(self.content_embed.content_embed(question_neighbors_index)))
        #
        agg_neighbor_feature = self.agg(question_neighbors_key, question_key, question_neighbor_value)
        question_value_new = self.que_generate(question_value, agg_neighbor_feature)

        score = self.score_fn(question_value_new, answer_value, user_value)
        # score = self.score_fn(question_value, answer_value, user_value)

        score = torch.softmax(score, dim=-1)

        return score
