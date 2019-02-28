import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Aggregate(torch.nn.Module):
    def __init__(self, input1_dim
                 ):
        super(Aggregate, self).__init__()
        self.bilinear = nn.Bilinear(input1_dim, input1_dim, input1_dim)

    def forward(self, neighbors_value, edges_value):
        middle = self.bilinear(neighbors_value, edges_value)
        middle_act = F.relu(middle)
        #max-pooling
        result, _ = torch.max(middle_act, dim=-2)
        return result





class AttentionAggregate_different_nodeedge(torch.nn.Module):
    def __init__(self, lstm_dim):
        super(AttentionAggregate_different_nodeedge, self).__init__()
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



class MiddleGeneration(torch.nn.Module):
    def __init__(self, dim):
        self.bilinear = nn.Bilinear(dim, dim, dim)

    def forward(self, neighbors, edges):
        middle = self.bilinear(neighbors, edges)
        middle_act = torch.tanh(middle)
        return middle_act


class AttentionAggregate_Weight(torch.nn.Module):
    def __init__(self, key_dim):
        super(AttentionAggregate_Weight, self).__init__()
        self.attention_weight = nn.Linear(key_dim, key_dim)
        self.a_weight = nn.Linear(key_dim, 1)



    def forward(self, middle_key, nodes_key, middle_value):
        # neighbors: batch * neighbor_count * dim
        # target: batch * 1 * dim

        nodes1 = nodes_key.unsqueeze(-2)
        attention_coef = F.relu(self.a_weight(self.attention_weight(nodes1) + self.attention_weight(middle_key)))
        attention_coef = F.softmax(attention_coef, dim=-2)
        neighbor_feature = torch.sum(attention_coef * middle_value, dim=-2)
        return neighbor_feature






class AttentionAggregate_Cos(torch.nn.Module):
    def __init__(self):
        super(AttentionAggregate_Cos, self).__init__()
        self.cos_sim = nn.CosineSimilarity(dim=-1)


    def forward(self, middle_key, nodes_key, middle_value):
        node_ = nodes_key.unsqueeze(-2)
        similarity = self.cos_sim(nodes_key, middle_key)
        # # every element
        weight = F.softmax(similarity, dim=-1)
        weight.unsqueeze_(-1)

        result = torch.sum(weight * middle_value, dim=-2)
        return result


class NodeEdgeCombinGenerate(torch.nn.Module):
    def __init__(self, value_dim):
        super(NodeEdgeCombinGenerate, self).__init__()
        self.edge_node_weight = nn.Linear(2 * value_dim, value_dim)

    def forward(self, other_node_value, edge_value):
        middle = torch.cat((other_node_value, edge_value), dim=-1)
        #TODO: different activate function
        middle = self.edge_node_weight(middle)
        middle = torch.tanh(middle)
        return middle

class NodeGenerate_Forgete_Gate(torch.nn.Module):
    def __init__(self, value_dim):
        super(NodeGenerate_Forgete_Gate, self).__init__()
        self.forget_weight = torch.rand(value_dim, value_dim)
        self.forget_weight = nn.Parameter(self.forget_weight)


    def forward(self, item_value, neighbor_agg_value):
        '''

        :param item_value: batch * feature
        :param neighbor_feature: batch * feature
        :return:
        '''

        result = F.linear(item_value, (1 - F.sigmoid(self.forget_weight))) + F.linear(item_value, F.sigmoid(self.forget_weight))
        result = F.leaky_relu(result)
        result = F.normalize(result,dim=-1)
        return result

class NodeGenerate_FeedForward(torch.nn.Module):
    def __init__(self, value_dim):
        super(NodeGenerate_FeedForward, self).__init__()
        self.linear = nn.Linear(2 * value_dim, value_dim)


    def forward(self, item_value, neighbor_agg_value):
        '''

        :param item_value: batch * feature
        :param neighbor_feature: batch * feature
        :return:
        '''
        concat = torch.cat((item_value, neighbor_agg_value), dim=-1)
        result = self.linear(concat)
        result = F.leaky_relu(result)
        result = F.normalize(result, dim=-1)
        return result

class EdgeGenerate(torch.nn.Module):
    def __init__(self):
        super(EdgeGenerate, self).__init__()
        self.cos = torch.nn.CosineSimilarity(dim=-1)

    #TODO: edge should get information from question and user node.
    #ATTENTION: Try using two items attention
    def forward(self, user_key, answer_key, user_value, answer_value):
        sim = self.cos(user_key, answer_key)
        edges = ( 1 - sim ) * answer_key + sim * user_key
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






