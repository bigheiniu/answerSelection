import torch
import torch.nn as nn
import torch.nn.functional as F
from MultihopAttention.Layer import LSTM_Hidden_List

class BiLstMCNN(torch.nn.Module):
    def __init__(self,
                 args, embedding):
        super(BiLstMCNN, self).__init__()
        self.args = args
        self.word_embed = nn.Embedding.from_pretrained(embedding)
        self.lstm_bi = LSTM_Hidden_List(args)
        self.atten_weight_a = nn.Linear(args.lstm_hidden_size, args.lstm_hidden_size)
        self.atten_weight_q = nn.Linear(args.lstm_hidden_size, args.lstm_hidden_size)
        self.atten_last = nn.Linear(args.lstm_hidden_size, 1)

    def forward(self, question, good_answer, is_training=True):
        question_embed = self.word_embed(question)
        good_answer_emebed = self.word_embed(good_answer)


        question_lstm = self.lstm_bi(question_embed)
        good_answer_lstm = self.lstm_bi(good_answer_emebed)

        #mean pooling
        output_q = torch.mean(question_lstm, dim=-2, keepdim=True)

        m_good = torch.tanh(self.atten_weight_a(good_answer_lstm) + self.atten_weight_q(output_q))

        #m_good: batch * sequen_length * hidden_size
        s_good = torch.softmax(self.atten_last(m_good), dim=-2)


        # Dropout operation is performed on the QA representations before cosine similarity matching.
        atten_good_answer = s_good * good_answer_lstm

        #mean pooling
        pool_good_answer = torch.mean(atten_good_answer, dim=-2)

        drop_atten_good_answer = F.dropout(pool_good_answer, 0.5)

        output_q1 = output_q.squeeze()
        drop_output_q1 = F.dropout(output_q1, 0.5)
        good_score = F.cosine_similarity(drop_output_q1, drop_atten_good_answer, dim=-1)
        return good_score


