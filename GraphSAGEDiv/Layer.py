import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Aggregate(torch.nn.Module):
    def __init__(self, input1_dim, input2_dim, bilinear_output_dim
                 ):
        super(Aggregate, self).__init__()
        self.bilinear = nn.Bilinear(input1_dim, input2_dim, bilinear_output_dim)

    def forward(self, neighbors, edges):
        middle = self.bilinear(neighbors, edges)
        middle_act = F.relu(middle)
        #max-pooling
        result,_ = torch.max(middle_act, dim=-2)
        return result

class OtherNodeGenerate(torch.nn.Module):
    def __init__(self, lstm_dim):
        super(OtherNodeGenerate, self).__init__()
        self.edge_node_weight = nn.Linear(2 * lstm_dim, lstm_dim)

    def forward(self, other_node, edge):
        middle = torch.cat((other_node, edge), dim=-1)
        #TODO: different activate function
        middle = self.edge_node_weight(middle)
        middle = F.tanh(middle)
        return middle


class AttentionAggregate__different_nodeedge(torch.nn.Module):
    def __init__(self, lstm_dim):
        super(AttentionAggregate__different_nodeedge, self).__init__()
        self.attention_weight_target_node = nn.Linear(lstm_dim, lstm_dim)
        self.attention_weight_other_node = nn.Linear(lstm_dim, lstm_dim)
        self.a_weight = nn.Linear(lstm_dim, 1)

    def forward(self, target, middle):
        # neighbors: batch * neighbor_count * dim
        # target: batch * 1 * dim => batch * dim
        target.unsqueeze_(-2)
        attention_coef = F.leaky_relu(self.a_weight(self.attention_weight_target_node(target) + self.attention_weight_other_node(middle)))
        attention_coef = F.softmax(attention_coef, dim=-2)

        neighbor_feature = torch.sum(attention_coef * middle, dim=-2)
        return neighbor_feature



class AttentionAggregate_weight_nodeedge(torch.nn.Module):
    def __init__(self, lstm_dim):
        super(AttentionAggregate_weight_nodeedge, self).__init__()
        self.attention_weight = nn.Linear(lstm_dim, lstm_dim)
        self.a_weight = nn.Linear(lstm_dim, 1)

    def forward(self, target, middle):
        # neighbors: batch * neighbor_count * dim
        # target: batch * 1 * dim
        target.unsqueeze_(-2)
        attention_coef = F.leaky_relu(self.a_weight(self.attention_weight(target) + self.attention_weight(middle)))
        attention_coef = F.softmax(attention_coef, dim=-2)

        neighbor_feature = torch.sum(attention_coef * middle, dim=-2)

        return neighbor_feature






class AttentionAggregate(torch.nn.Module):
    def __init__(self, input1_dim, input2_dim, bilinear_output_dim):
        super(AttentionAggregate, self).__init__()
        self.bilinear = nn.Bilinear(input1_dim, input2_dim, bilinear_output_dim)
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, neighbors, edges, node):
        middle = self.bilinear(neighbors, edges)
        middle_act = F.relu(middle)
        node_ = node.unsqueeze(-2)
        similarity = self.cos_sim(middle_act, node_)
        # # every element
        weight = F.softmax(similarity, dim=-1)
        weight.unsqueeze_(-1)
        result = torch.sum(middle_act, dim=-2)
        return result


class NodeGenerate_forget(torch.nn.Module):
    def __init__(self, input_dim):
        super(NodeGenerate_forget, self).__init__()
        self.forget_weight = nn.Linear(input_dim, input_dim)
        self.forget_gate = F.sigmoid(self.forget_weight)



    def forward(self, item, neighbor_agg):
        '''

        :param item: batch * feature
        :param neighbor_feature: batch * feature
        :return:
        '''
        result = (1 - self.forget_gate) * item + self.forget_gate * neighbor_agg
        result = F.normalize(result)
        return result

class NodeGenerate(torch.nn.Module):
    def __init__(self, input_dim):
        super(NodeGenerate, self).__init__()
        self.linear = nn.Linear(2*input_dim, input_dim)


    def forward(self, item, neighbor_agg):
        '''

        :param item: batch * feature
        :param neighbor_feature: batch * feature
        :return:
        '''
        concat = torch.cat((item, neighbor_agg), dim = -1)
        result = self.linear(concat)
        F.relu(result, inplace=True)
        result = F.normalize(result)
        return result

class EdgeGenerate(torch.nn.Module):
    def __init__(self):
        super(EdgeGenerate, self).__init__()

    def forward(self, edges, questions, users):
        return edges



class ARMNLScore(torch.nn.Module):
    def __int__(self, inputdim):
        super(ARMNLScore, self).__int__()
        self.bilinear = nn.Linear(inputdim, inputdim, 1)


    def forward(self, question, answer, user):
        relevance_score = self.bilinear(question, answer)
        q_u_score = torch.sum(question * user, dim=-1)
        score = relevance_score * q_u_score
        return score

class TensorScore(torch.nn.Module):
    def __init__(self, inputdim, cntn_feature_r, num_class):
        super(TensorScore, self).__init__()
        self.bilinear = nn.Bilinear(inputdim, inputdim, cntn_feature_r)
        self.linear = nn.Linear(2 * inputdim, cntn_feature_r, bias=False)
        self.last_weight = nn.Linear(cntn_feature_r, num_class)

    def forward(self, question, answer_user):
        score = self.last_weight(torch.tanh(self.bilinear(question, answer_user) + \
        self.linear(torch.cat(question, answer_user), dim=-1)))
        return score

class CosineSimilar(torch.nn.Module):
    def __init__(self):
        super(CosineSimilar, self).__init__()
        self.cos_sim = torch.nn.CosineSimilarity(dim=-1)
    def forward(self, question, answer_user):
        return self.cos_sim(question, answer_user)

class MultiplyDirect(torch.nn.Module):
    def __init__(self):
        super(MultiplyDirect, self).__init__()

    def forward(self, question, answer_user):
        return torch.sum(question * answer_user, dim=-1)

class WeightScore(torch.nn.Module):
    def __init__(self, input_dim, num_class):
        super(WeightScore, self).__init__()
        self.w_q = nn.Linear(input_dim, input_dim)
        self.w_a = nn.Linear(input_dim, input_dim)
        self.w_u = nn.Linear(input_dim, input_dim)
        self.last_weight = nn.Linear(input_dim, num_class)

    def forward(self, question, answer, user):
        return self.last_weight(
            torch.tanh(
                self.w_q(question) + self.w_a(answer) + self.w_u(user)
            )
        )


# for flexible length of content
# use cnn to speed the test
class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()
        self.args = args
        self.layer_cnn = args.layer_cnn
        self.cnn = nn.Conv2d(self.args.cnn_inchanel, self.args.cnn_outchannel, self.args.cnn_kernel_size)
        self.bn = nn.BatchNorm1d(self.args.cnn_feature)

    def forward(self, content):
        content_cnn = self.cnn(content)
        if self.args.cnn_pool == "max_pool":
            content_pool = torch.max(content_cnn, dim=-1)
        else:
            #batch * feature
            content_pool = spatial_pyramid_pool(content_cnn, content_cnn.shape[0], content_cnn.shape[-2:], self.args.cnn_out_pool_size)
        content_normal = self.bn(content_pool)
        return content_normal

#https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py
def spatial_pyramid_pool( previous_conv, num_sample, previous_conv_size, out_pool_size):
    '''
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    '''
    # print(previous_conv.size())
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2
        w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if (i == 0):
            spp = x.view(num_sample, -1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp






