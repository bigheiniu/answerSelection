
import torch.nn as nn
import torch
import numpy as np
import gensim
import os
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.model_selection import train_test_split
import collections
import matplotlib.pyplot as plt
import logging
import torch.nn.functional as F
from sklearn.model_selection import ParameterGrid
from Constants import Constants
import random


class PairWiseHingeLoss(nn.Module):
    def __init__(self, margin):
        super(PairWiseHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, postive_score, negative_score):
        loss_hinge = torch.mean(F.relu(self.margin - postive_score + negative_score))
        return loss_hinge


def grid_search(params_dic):
    '''
    :param params_dic: similar to {"conv_size":[0,1,2], "lstm_hiden_size":[1,2,3]}
    :return: iter {"conv_size":1, "lstm_hidden_size":1}
    '''
    grid_parameter = ParameterGrid(params_dic)
    parameter_list = []
    for params in grid_parameter:
        params_dic_result = {}
        for key in params_dic.keys():
            params_dic_result[key] = params[key]
        parameter_list.append(params_dic_result)
    return parameter_list


def diversity_evaluation(diversity_answer_recommendation, topK, tfidf, lda):
    #init evaluate class
    tf_idf_score = 0
    lda_score = 0
    question_count = len(diversity_answer_recommendation)
    for candidate_answer_list in diversity_answer_recommendation:
        candidate_word_space = []
        top_word_space = []
        for answer_content in candidate_answer_list:
            candidate_word_space += answer_content.tolist()
        for top_answer_content in candidate_answer_list[:topK]:
            top_word_space += top_answer_content.tolist()

        tf_idf_score += tfidf.simiarity(candidate_word_space, top_word_space)
        lda_score += lda.similarity(candidate_word_space, top_word_space)
    return (tf_idf_score * 1.0) / question_count, (lda_score * 1.0) / question_count

class LSTM_MeanPool(nn.Module):
    def __init__(self, args):
        super(LSTM_MeanPool, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(args.embed_size, args.lstm_hidden_size, batch_first=True,
                            dropout=args.drop_out_lstm, num_layers=args.lstm_num_layers,bidirectional = args.bidirectional)

    def lstm_init(self, batch_size):
        h_0_size_1 = 1
        if self.args.bidirectional:
            h_0_size_1 *= 2
        h_0_size_1 *= self.args.lstm_num_layers
        hiddena = torch.zeros((h_0_size_1, batch_size, self.args.lstm_hidden_size),
                              dtype=torch.float, device=self.args.device)
        hiddenb = torch.zeros((h_0_size_1, batch_size, self.args.lstm_hidden_size),
                              dtype=torch.float, device=self.args.device)
        return hiddena, hiddenb

    def forward(self, input):
        shape = [*input.shape]
        input = input.view(-1, shape[-2], shape[-1])
        shape[-1] = 2 if self.args.bidirectional else 1
        shape[-1] = shape[-1] * self.args.lstm_hidden_size
        del shape[-2]
        hiddena, hiddenb = self.lstm_init(input.shape[0])
        output, _ = self.lstm(input, (hiddena, hiddenb))
        output = torch.mean(output, dim = -2)
        output = output.view(tuple(shape))
        return output

class ContentEmbed:
    def __init__(self, content):
        self.content = content

    @property
    def content_list(self):
        return self.content
    def content_embed(self, batch_id):
        if type(batch_id) is np.int or np.int64:
            content = self.content[batch_id]
        else:
            shape = [*batch_id.shape]
            shape.append(len(self.content[0]))
            shape = tuple(shape)
            batch_id = batch_id.view(-1, )
            content = self.content[batch_id]
            content = content.view(shape)
        return content

class ShripContext:
    def __init__(self, content_embed, user_count, args):
        self.content_embed = content_embed
        self.user_count = user_count
        self.max_len = args.max_u_len
        self.device = args.device

    def getUserContext(self, context_id_list):
        document = []
        for answer_id in context_id_list:
            document += self.content_embed.content_embed(answer_id - self.user_count)
            if len(document) > self.args.max_u_len:
                document = document[:self.args.max_u_len]
                break
        if len(document) < self.args.max_u_len:
            pad_word = [Constants.PAD] * (self.args.max_u_len - len(document))
            document += pad_word
        document = torch.LongTensor(document).to(self.device)
        return document

    def shripUser(self, batch_id):
        shape = batch_id.shape
        batch_id_new = batch_id.view(-1, shape[-1])
        batch_id_content = []
        for user_context in batch_id_new:
            temp = self.getUserContext(user_context)
            batch_id_content.append(temp)
        result = torch.cat(batch_id_content)
        shape[-1] = self.max_len
        result = result.view(shape)
        return result

class UserContextEmbed:
    def __init__(self, content_embed, user_context_embed, args, user_count):
        self.content_embed = content_embed
        self.user_context_embed = user_context_embed
        self.user_count = user_count
        self.args = args

    def getUserContext(self, context_id_list):
        document = []
        for answer_id in context_id_list:
            document += self.content_embed.content_embed(answer_id - self.user_count)
            if len(document) > self.args.max_u_len:
                document = document[:self.args.max_u_len]
                break
        if len(document) < self.args.max_u_len:
            pad_word = [Constants.PAD] * (self.args.max_u_len - len(document))
            document += pad_word
        return document

    def buildUserContextEmbed(self, userContext):
        userEmbed = []
        for context_list in userContext:
            userEmbed.append(self.getUserContext(context_list))
        return userEmbed





def Adjance(G, max_degree):
    nodes_count = len(G.nodes())
    adj = np.zeros((nodes_count + 1, max_degree), dtype=np.int64)
    adj_edge = np.zeros_like(adj, dtype=np.int64)
    adj_score = np.zeros_like(adj, dtype=np.float)
    adj_degree = np.zeros(nodes_count + 1,)
    for node in G.nodes():
        assert not isinstance(node, str), "[ERROR] Argument of node should be int"
        neighbors = np.array([neighbor for neighbor in G.neighbors(node)])
        adj_degree[node] = len(neighbors)

        if len(neighbors) == 0:
            continue
        if len(neighbors) < max_degree:
            neighbors = np.random.choice(neighbors, max_degree, replace=True)
        else:
            neighbors = np.random.choice(neighbors, max_degree, replace=False)

        for index, neigh_node in enumerate(neighbors):
            adj_edge[node, index] = G[node][neigh_node]['a_id']
            adj_score[node, index] = G[node][neigh_node]['score']

        adj[node, :] = neighbors
    adj = torch.LongTensor(adj)
    adj_edge = torch.LongTensor(adj_edge)
    adj_score = torch.IntTensor(adj_score)
    #all the return are tensor
    return adj, adj_edge, adj_score



def loadEmbed(file, embed_size, vocab_size, word2idx=None, Debug=True):
    # read pretrained word2vec, convert to floattensor
    if(Debug):
        print("[WARN] load randn embedding for DEBUG")
        embed = np.random.rand(vocab_size, embed_size)
        return torch.FloatTensor(embed)

    #load pretrained model
    else:
        embed_matrix = np.zeros([len(word2idx), embed_size])
        print("[Info] load pre-trained word2vec embedding")
        sub_dir = "/".join(file.split("/")[:-1])
        if "glove" in file:
            word2vec_file = ".".join(file.split("/")[-1].split(".")[:-1])+"word2vec"+".txt"
            if word2vec_file not in os.listdir(sub_dir):
                glove2word2vec(file, os.path.join(sub_dir, word2vec_file))
            file = os.path.join(sub_dir, word2vec_file)

        model = gensim.models.KeyedVectors.load_word2vec_format(file,
                                                                binary=False)
        print("[Info] Load glove finish")

        for word, i in word2idx.items():
            if word in model.vocab:
                embed_matrix[i] = model.word_vec(word)

        weights = torch.FloatTensor(embed_matrix)

        return weights



def content_len_statics(content_list):
    content_length = [len(i) for i in content_list]
    info = {}
    info['min'] = np.min(content_length)
    info['max']= np.max(content_length)
    info['median'] = np.median(content_length)
    info['mean'] = np.mean(content_length)
    info['var'] = np.var(content_length)
    info['twofive'] = np.percentile(content_length,25)
    info['fivezero'] = np.percentile(content_length,50)
    info['sevenfive'] = np.percentile(content_length,75)
    info['histogram'] = np.histogram(content_length,bins=10)
    return info


def tensorTonumpy(data, is_gpu):
    if is_gpu:
        return data.cpu().numpy()
    else:
        return data.numpy()


def train_test_split_len(question_count):
    question_list = list(range(question_count))
    question_train_list, question_test_list = train_test_split(question_list, random_state=91)
    return np.array(question_train_list), np.array(question_test_list)

def graph_eval(G, user_count):
    user_degree_list = sorted([d for n, d in G.degree() if n < user_count], reverse=True)
    question_degree_list = sorted([d for n, d in G.degree() if n > user_count], reverse=True)
    user_degree_count = collections.Counter(user_degree_list)
    question_degree_count = collections.Counter(question_degree_list)
    user_deg, user_cnt = zip(*user_degree_count.items())
    question_deg, question_cnt = zip(*question_degree_count.items())

    return (user_deg, user_cnt),(question_deg, question_cnt)

def plot_bar(deg, cnt, figure_path):
    fig, ax = plt.subplots()
    plt.bar(deg, cnt, width=0.80, color='b')

    plt.title("Degree Histogram")
    plt.ylabel("Count")
    plt.xlabel("Degree")
    ax.set_xticks([d + 0.4 for d in deg])
    ax.set_xticklabels(deg)
    plt.savefig(figure_path, dpi=150)

def createLogHandler(job_name,log_file):
    logger = logging.getLogger(job_name)
    ## create a file handler ##
    handler = logging.FileHandler(log_file)
    ## create a logging format ##
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


