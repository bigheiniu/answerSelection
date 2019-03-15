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

class AttentionAggregate(torch.nn.Module):
    def __init__(self, input1_dim, input2_dim, bilinear_output_dim):
        super(AttentionAggregate, self).__init__()
        self.bilinear = nn.Bilinear(input1_dim, input2_dim, bilinear_output_dim)
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, neighbors, edges, node):
        middle = self.bilinear(neighbors, edges)
        middle_act = F.relu(middle)
        node_ = node.unsqueeze(-2)
        # print("\n[DEBUG]: size of node is {}".format(node.shape))
        # print("\n[DEBUG]: size of middleact is {}".format(middle_act.shape))
        similarity = self.cos_sim(middle_act, node_)
        # # every element
        weight = F.softmax(similarity, dim=-1)
        weight.unsqueeze_(-1)
        result = torch.sum(middle_act, dim=-2)
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


class PairWiseHingeLoss(nn.Module):
    def __init__(self, margin):
        super(PairWiseHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, postive_score, negative_score):
        loss_hinge = torch.mean(F.relu(self.margin - postive_score + negative_score))
        return loss_hinge