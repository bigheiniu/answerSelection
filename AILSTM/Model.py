import torch
import torch.nn as nn
from MultihopAttention.Layer import LSTM_Hidden_List

class AILSTM(nn.Module):
    def __init__(self,
                 args,
                 pre_trained_word2vec):
        super(AILSTM, self).__init__()
        self.args = args
        self.wordEmbedd = nn.Embedding.from_pretrained(pre_trained_word2vec)
        self.LSTM = LSTM_Hidden_List(args)
        self.weight_a = nn.Linear(2 * args.lstm_hidden_size, args.lstm_hidden_size)
        self.a_feedfroward_question = nn.Linear(2 * args.lstm_hidden_size , 1)
        self.a_feedfroward_answer = nn.Linear(2*args.lstm_hidden_size, 1)
        self.weight_last = nn.Linear(2 * args.lstm_hidden_size, args.num_class)


    def forward(self, question_content, answer_content):
        question_embed = self.wordEmbedd(question_content)
        answer_embed = self.wordEmbedd(answer_content)
        question_lstm = self.LSTM(question_embed)
        answer_lstm = self.LSTM(answer_embed)

        # batch * question_length * answer_length * lstm_hidden_size
        question_length = question_lstm.shape[-2]
        answer_length = answer_lstm.shape[-2]
        question_lstm_expand = question_lstm.unsqueeze(2).expand(-1, -1, answer_length, -1)
        answer_lstm_expand = answer_lstm.unsqueeze(1).expand(-1, question_length, -1, -1)
        #ATTENTION: activate function is tanh
        A = torch.tanh(self.weight_a(torch.cat((question_lstm_expand, answer_lstm_expand), dim=-1)))
        r_q, _ = torch.max(A, dim=-3)
        r_a, _ = torch.max(A, dim=-2)
        cat_q = torch.cat((question_lstm, r_q), dim=-1)
        cat_a = torch.cat((answer_lstm, r_a), dim=-1)
        alpha_q = torch.softmax(self.a_feedfroward_question(cat_q), dim=-2)
        alpha_a = torch.softmax(self.a_feedfroward_answer(cat_a), dim=-2)
        feature_q = torch.sum(alpha_q * r_q, dim=-2)
        feature_a = torch.sum(alpha_a * r_a, dim=-2)
        feature_last = torch.cat((feature_q, feature_a), dim=-1)

        if self.args.is_classification:
            score = torch.log_softmax(self.weight_last(feature_last),dim = -1)
            predict = score.max(dim=-1)[1]
            return score, predict
        else:
            #ATTENTION: act function is the tanh
            score = torch.tanh(self.weight_last(feature_last))

            return score

