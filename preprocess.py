''' Handling the data io '''
import argparse
import torch
# from XMLHandler.XMLHandler_SemEval import xmlhandler
from Util import content_len_statics
from XMLHandler.XMLHandler_StackOverflow import xmlhandler
from TextClean import textClean
import numpy as np
import networkx as nx
from Constants import Constants

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
        words = textClean.cleanText(sent)
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
    print("[INFO] No Vote Answer: {}, All Answer: {}".format(ap, len(question_answer_user_label)))
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
    print("[INFO] Graph contains {} edge.".format(len(G.edges())))
    # print("[INFO] There are {} users, bigggest id is {}".format(len(set(user_list)), max(user_list)))
    return G

def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()
    # add by yichuan li
    parser.add_argument('-raw_data',default="/home/yichuan/course/data")



    parser.add_argument('-max_len', '--max_word_seq_len', type=int, default=60)
    parser.add_argument('-min_word_count', type=int, default=5)
    parser.add_argument('-keep_case', action='store_true')

    parser.add_argument('-share_vocab', action='store_true')
    parser.add_argument('-vocab', default=None)
    parser.add_argument('-train_size', default=0.6)
    parser.add_argument('-val_size', default=0.4)
    parser.add_argument('-test_size', default=0.0)

    opt = parser.parse_args()
    question_answer_user_vote, body, user_context, accept_answer_dic, title, user_count, question_count = xmlhandler.main(opt.raw_data)

    content_word_list = shrink_clean_text(body, opt.max_word_seq_len)
    title_world_list = shrink_clean_text(title, opt.max_word_seq_len)
    question_answer_user_vote = np.array(question_answer_user_vote)

    # Build vocabulary
    word2idx = build_vocab_idx(content_word_list + title_world_list, opt.min_word_count)
    # word to index
    print('[DEBUG] Convert  word instances into sequences of word index.')
    title_id = convert_instance_to_idx_seq(title_world_list, word2idx)
    word_id = convert_instance_to_idx_seq(content_word_list, word2idx)

    info_content = content_len_statics(word_id)
    info_title = content_len_statics(title_id)
    print('[INFO] content length infomation: {}'.format(info_content))
    print('[INFO] title length information: {}'.format(info_title))
    #split train-valid-test dataset

    index = np.arange(len(question_answer_user_vote))
    np.random.shuffle(index)
    length = len(question_answer_user_vote)
    train_end = int(opt.train_size * length)
    val_end = int(opt.val_size * length) + train_end
    train_index = index[:train_end]
    val_index = index[train_end: val_end]
    test_index = index[val_end:]

    G = GenerateGraph(question_answer_user_vote, train_index, val_index)



    data = {
        'settings': opt,
        'dict': word2idx,
        'content': word_id,
        'title': title_id,
        'question_answer_user_train': question_answer_user_vote[train_index],
        'question_answer_user_val': question_answer_user_vote[val_index],
        'question_answer_user_test': question_answer_user_vote[test_index],
        'G': G,
        'user_count': user_count,
        'question_count': question_count,
        'user':user_context
    }

    opt.save_data="data/store_stackoverflow.torchpickle"
    print('[Info] Dumping the processed data to pickle file', opt.save_data)
    torch.save(data, opt.save_data)
    print('[Info] Finish.')





if __name__ == '__main__':
    main()