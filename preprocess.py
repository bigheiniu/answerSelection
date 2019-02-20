''' Handling the data io '''
import argparse
import torch
# from XMLHandler.XMLHandler_SemEval import xmlhandler
from Util import content_len_statics, plot_bar, graph_eval
from TextClean import TextClean
import numpy as np
import networkx as nx
from networkx.algorithms.components.connected import  number_connected_components
from Constants import Constants
from Config import config_data_preprocess
from Util import createLogHandler
from XMLHandler_StackOverflow import xmlhandler as stack_xmlhandler
from XMLHandler_SemEval import xmlhandler as sem_xmlhandler


def shrink_clean_text(content, max_sent_len):
    ''' Convert file into word seq lists and vocab
        Pad all element in sequence in the static length
     '''

    word_insts = []
    trimmed_sent_count = 0
    i = 0
    j = 0
    for sent in content:
        i += 1
        words = TextClean.cleanText(sent)
        if len(words) > max_sent_len:
            trimmed_sent_count += 1
        elif len(words) < max_sent_len:
            pad_sequence = [Constants.PAD_WORD] * (max_sent_len - len(words))
            words = words + pad_sequence
        word_inst = words[:max_sent_len]
        # word_inst = words
        if word_inst:
            word_insts += [word_inst]
        else:
            j += 1
            word_insts += [Constants.PAD_WORD] * max_sent_len

    print('[Info] Get {} instances'.format(len(word_insts)))
    print('[Warning] {} instances is empty'.format(j))

    if trimmed_sent_count > 0:
        print('[Warning] {} instances are trimmed to the max sentence length {}.'
              .format(trimmed_sent_count, max_sent_len))

    return word_insts

def build_vocab_idx(word_insts, min_word_count):
    ''' Trim vocab by number of occurence '''

    full_vocab = set()
    for sent in word_insts:
        for w in sent:
            full_vocab.add(w)
    # full_vocab = set(w for sent in word_insts for w in sent if sent is not None)
    print('[Info] Original Vocabulary size =', len(full_vocab))

    word2idx = {
        # Constants.BOS_WORD: Constants.BOS,
        # Constants.EOS_WORD: Constants.EOS,
        Constants.PAD_WORD: Constants.PAD,
        Constants.UNK_WORD: Constants.UNK}

    word_count = {w: 0 for w in full_vocab}

    for sent in word_insts:
        for word in sent:
            word_count[word] += 1

    ignored_word_count = 0
    for word, count in word_count.items():
        if word not in word2idx:
            if count > min_word_count:
                word2idx[word] = len(word2idx)
            else:
                ignored_word_count += 1
    print('[Info] Trimmed vocabulary size = {},'.format(len(word2idx)),
          'each with minimum occurrence = {}'.format(min_word_count))
    print("[Info] Ignored word count = {}".format(ignored_word_count))
    return word2idx

def convert_instance_to_idx_seq(word_insts, word2idx):
    ''' Mapping words to idx sequence. '''
    return [[word2idx.get(w, Constants.UNK) for w in s] for s in word_insts]



def GenerateGraph(question_answer_user_label, train_index, val_index):
    train_data = question_answer_user_label[train_index]
    val_data = question_answer_user_label[val_index]
    G = nx.Graph()
    t = [1 for i in question_answer_user_label if len(i) == 3]
    ap = np.sum(t)
    th = np.unique(np.array([np.array([t[0], t[1]]) for t in question_answer_user_label ]), axis=0)
    th1 = np.unique(np.array([np.array([t[0], t[1]]) for t in question_answer_user_label]), axis=0)
    th2 = np.unique(np.array([ np.array(line) for line in question_answer_user_label]), axis=0)
    print("[INFO] unique question answer pair: {}, unique question answer user paris:{}, unique pairs {}".format(len(th), len(th1), len(th2)))
    print("[INFO] No Vote Answer: {}, All Answer: {}".format(ap, len(question_answer_user_label)))
    i = 0
    for line in train_data:
        question = line[0]
        answer = line[1]
        user = line[2]
        try:
            label = line[3]
        except:
            # no vote
            label = 0
        G.add_node(question)
        G.add_node(user)
        G.add_edge(question, user, a_id=answer, score=label, train_removed=False)
        i += 1
    for line in val_data:
        question = line[0]
        answer = line[1]
        user = line[2]
        try:
            label = line[3]
        except:
            # no vote
            label = 0
        G.add_node(question, type=0)
        G.add_node(user, type=1)
        G.add_edge(question, user, a_id=answer, score=label, train_removed=True)
        i += 1
    print("[INFO] Graph contains {} edge. QA pairs are {}, count is {}".format(len(G.edges()), len(question_answer_user_label), i))
    print("[INFO] Connected components {}".format(number_connected_components(G)))
    # print("[INFO] There are {} users, bigggest id is {}".format(len(set(user_list)), max(user_list)))
    return G

def plot_G(G, user_count):
    user, question = graph_eval(G, user_count)
    plot_bar(user[0], user[1], "result/user_degree.pdf")
    plot_bar(question[0], question[1],"result/question_degree.pdf")

def main():
    ''' Main function '''
    config = config_data_preprocess
    logger = createLogHandler(config.logger_name, config.log_file)
    title_world_list = []
    if config.is_classification is False:
        question_answer_user_vote, body, user_context, accept_answer_dic, title, user_count, question_count, love_list_count=stack_xmlhandler.main(config.raw_data)
        title_world_list = shrink_clean_text(title, config.max_len)
    else:
        question_answer_user_vote, body, user_context, user_count, question_count = sem_xmlhandler.main(config.raw_data)
    content_word_list = shrink_clean_text(body, config.max_len)
    question_answer_user_vote = np.array(question_answer_user_vote)

    # Build vocabulary
    word2idx = build_vocab_idx(content_word_list + title_world_list, config.min_word_count)
    # word to index
    print('[DEBUG] Convert  word instances into sequences of word index.')
    if config.is_classification is False:
        title_id = convert_instance_to_idx_seq(title_world_list, word2idx)
        info_title = convert_instance_to_idx_seq(title_id, word2idx)
        logger.info('Title length information: {}'.format(info_title))

    word_id = convert_instance_to_idx_seq(content_word_list, word2idx)
    info_content = content_len_statics(word_id)
    logger.info('Content length infomation: {}'.format(info_content))
    #split train-valid-test dataset

    index = np.arange(len(question_answer_user_vote))
    np.random.shuffle(index)
    length = len(question_answer_user_vote)
    train_end = int(config.train_size * length)
    train_index = index[:train_end]
    test_index = index[train_end:]

    G = GenerateGraph(question_answer_user_vote, train_index, test_index)

    # plot degree distribution of the graph
    plot_G(G, user_count)

    data = {
        'settings': config,
        'dict': word2idx,
        'content': word_id,
        'question_answer_user_train': question_answer_user_vote[train_index],
        'question_answer_user_test': question_answer_user_vote[test_index],
        'G': G,
        'user_count': user_count,
        'question_count': question_count,
        'user_context':user_context
    }
    if config.is_classification is not True:
        data["title"] = title_id
        data["love_list_count"] = love_list_count
        config.save_data="data/store_stackoverflow.torchpickle"
    else:
        config.save_data="data/store_SemEval.torchpickle"
    logger.info("Dumping the processed data to pickle file: {}".format(config.save_data))
    torch.save(data, config.save_data)
    logger.info("Finish text extraction, storing")


if __name__ == '__main__':
    # store logger information
    main()