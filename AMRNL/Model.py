import torch
import torch.nn.functional as F
import torch.nn as nn
from Util import LSTM


'''
Community-Based Question Answering via Asymmetric Multi-Faceted Ranking Network Learning
'''
class AMRNL(nn.Module):
    def __init__(self, args,
                 user_count,
                 word2vec,
                 ContentEmbed,
                 user_adjance=None,
                 user_love_degree=None
                 ):
        super(AMRNL, self).__init__()
        hidden_size = (2 if args.bidirectional else 1) * args.lstm_hidden_size
        self.lstm = LSTM(args)
        self.args = args
        self.user_count = user_count
        self.user_embed = nn.Embedding(self.user_count + 2, hidden_size, padding_idx= self.user_count)
        self.content_matrix = ContentEmbed
        self.word2vec = nn.Embedding.from_pretrained(word2vec)
        #already normalized
        self.user_adjance_embed = user_adjance
        self.user_love_degree_embed = user_love_degree

        # f_M(q_i, u_j, a_k) = s_M(q_i, a_k)s(q_i, u_j)
        # s_M(q_i, a_k) = q_i * M * a_k => batch_size * 1 => batch of question answer match score

        self.smantic_meache_bilinear = nn.Bilinear(hidden_size, hidden_size, 1)


    def forward(self,
                question_list,
                answer_list,
                user_list,
                need_feature=False
                ):


        question_embed_feature = self.word2vec(self.content_matrix.content_embed(question_list- self.user_count))
        answer_embed_feature = self.word2vec(self.content_matrix.content_embed(answer_list - self.user_count))
        user_embed_feature = self.user_embed(user_list)

        question_lstm = self.lstm(question_embed_feature)
        answer_lstm = self.lstm(answer_embed_feature)

        match_score = self.smantic_meache_bilinear(question_lstm, answer_lstm)
        #ATTENTION: In ARMNL they use (q_i).T * u_j as similarity between question and answer
        relevance_score = F.cosine_similarity(question_lstm, user_embed_feature, dim=-1)
        relevance_score.unsqueeze_(-1)
        result = match_score * relevance_score
        #l2 norm
        user_neighbor_feature = torch.sum(self.user_embed(self.user_adjance_embed.content_embed(user_list)), dim=-2) / (self.user_love_degree_embed.content_embed(user_list) + self.args.follow_smooth)
        regular = F.normalize(user_embed_feature - user_neighbor_feature, 2, dim=-1)
        return_list = [result, regular]
        if need_feature:
            answer_vec = answer_lstm.detach()
            return_list.append(answer_vec)
        return tuple(return_list)






