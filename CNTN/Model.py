import torch
import torch.nn as nn
import torch.nn.functional as F
from .Layer import kmax_pooling, dynamic_k_cal, matrix2vec_max_pooling

'''
Convolutional Neural Tensor Network Architecture for Community-based Question Answering

'''
class CNTN(nn.Module):
    def __init__(self, args, word2_vec, Content_embed, user_count):
        super(CNTN, self).__init__()
        self.args = args
        self.word_embedding = nn.Embedding.from_pretrained(word2_vec)
        # input channels and output channels are the same
        self.cnn_lr = nn.Conv2d(1, args.k_max_s, (3, args.embed_size))
        # self.cnn_list = [nn.Conv2d(1, 1, kernel_size).to(self.args.device) for kernel_size in self.args.cntn_kernel_size]
        self.Content_emebd = Content_embed
        self.user_count = user_count

        self.bilinear_M = nn.Bilinear(self.args.k_max_s, self.args.k_max_s, self.args.cntn_feature_r)
        self.linear_V = nn.Linear(2 * args.k_max_s, self.args.cntn_feature_r, bias=False)
        self.linear_U = nn.Linear(self.args.cntn_feature_r, self.args.num_class, bias=False)

        # nn.init.xavier_normal_(self.bilinear_M.weight)
        # nn.init.xavier_normal_(self.linear_V.weight)
        # nn.init.xavier_normal_(self.linear_U.weight)

    def forward(self, question, answer_list, _, need_feature=False):

        question_embed = self.word_embedding(self.Content_emebd.content_embed(question - self.user_count))
        question_embed.unsqueeze_(1)
        answer_embed = self.word_embedding(self.Content_emebd.content_embed(answer_list - self.user_count))
        answer_embed.unsqueeze_(1)
        question_length = question_embed.shape[-2]
        answer_length = answer_embed.shape[-2]

        #
        #
        question_cnn, _ = torch.max(self.cnn_lr(question_embed), dim=-2)
        answer_cnn,_ = torch.max(self.cnn_lr(answer_embed), dim=-2)

        # cnn_count = len(self.cnn_list)
        # for depth, cnn in enumerate(self.cnn_list):
        #     # Convolution
        #     question_embed = cnn(question_embed)
        #     answer_embed = cnn(answer_embed)
        #     depth = depth + 1
        #     #k-max-pooling
        #     if depth < cnn_count:
        #         k_question = dynamic_k_cal(current_layer_depth=depth, cnn_layer_count=cnn_count, sentence_length=question_length,k_top=self.args.k_max_s)
        #         k_answer = dynamic_k_cal(current_layer_depth=depth, cnn_layer_count=cnn_count, sentence_length=answer_length,k_top=self.args.k_max_s)
        #     else:
        #         k_question = self.args.k_max_s
        #         k_answer = self.args.k_max_s
        #     #Non-linear Feature Function
        #     question_embed = torch.tanh(kmax_pooling(question_embed, -2, k_question))
        #     answer_embed = torch.tanh(kmax_pooling(answer_embed, -2, k_answer))
        #
        #
        #
        # # transpose question/answer embedding
        # # Final Layer
        # question_embed = matrix2vec_max_pooling(question_embed, dim=-1)
        # answer_embed = matrix2vec_max_pooling(answer_embed, dim =-1)
        question_cnn.squeeze_()
        answer_cnn.squeeze_()
        # question_embed.squeeze_()
        # answer_embed.squeeze_()
        # q_m_a = self.bilinear_M(question_embed, answer_embed)

        q_m_a = self.bilinear_M(question_cnn, answer_cnn)
        # q_m_a = q_m_a + self.linear_V(torch.cat((question_embed, answer_embed), dim=-1))
        q_m_a = q_m_a + self.linear_V(torch.cat((question_cnn, answer_cnn), dim=-1))
        q_m_a = torch.tanh(q_m_a)
        score = self.linear_U(q_m_a)
        score.squeeze_(-1)
        if self.args.is_classification:
            score_log_softmax = F.log_softmax(score, dim=-1)
            score_soft_max = F.softmax(score, dim=-1)
            predict = torch.argmax(score_soft_max, dim=-1)
            return_list = [score_log_softmax, score_soft_max, predict]
        else:
            return_list = [score]
        if need_feature:
            answer_vec = answer_cnn.detach()
            # answer_vec = answer_embed.detach()
            return_list.append(answer_vec)
        return tuple(return_list)





