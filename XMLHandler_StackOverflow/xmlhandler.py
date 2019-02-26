"""This module produces SemEval Task 3, Subtask B datasets in JSON."""



from XMLHandler_StackOverflow.XMLpreprocessing import parse_post, parse_vote
import os
import numpy as np
import collections
from Config import config_data_preprocess
import operator

def debugTest(list_line):
    data = np.array(list_line)
    max_question = np.max(data[:,0])
    max_answer = np.max(data[:,1])
    max_user = np.max(data[:,2])
    length = len(list_line)
    print("[INFO] max question id: {}\n max answer id: {} \n max user id: {}\n all length is {}".format(max_question, max_answer, max_user, length))


def reorde_lover_dic(love_dic, user_dic, max_love_count, user_count):
    love_list = []
    user_dic_sorted = collections.OrderedDict(sorted(user_dic.items(), key=lambda x: x[1]))
    #TODO: Using sparse matrix for user followship
    for user_id, user_index in user_dic_sorted.items():
        # assert user_index == len(love_list), "Love dic is wrong"
        th = []
        if user_id not in love_dic:
            love_list.append(th)
            continue
        for love_user_id in love_dic[user_id]:
            try:
                love_user_index = user_dic[love_user_id]
                th.append(love_user_index)
            except:
                pass
        love_list.append(th)
    love_count = [len(love_users) if len(love_users) < max_love_count else max_love_count for love_users in love_list ]
    love_count = [max_love_count if i == 0 else i for i in love_count]
    shrink_love_list = []
    for love_users in love_list:
        if len(love_users) > max_love_count:
            th = love_users[:max_love_count]
        else:
            pad = [user_count] * (max_love_count - len(love_users))
            th = love_users + pad

        shrink_love_list.append(th)
    return shrink_love_list, love_count

# def love_list2sparse_matrx(love_list, dimention):
#     value = []
#     row = []
#     col = []
#     for index, love_users in enumerate(love_list):
#         value_each = 1.0 / len(love_users)
#         for love_user_index in love_users:
#             row.append(index)
#             col.append(love_user_index)
#             value.append(value_each)
#     return value,[row, col], dimention


def content_evaluation(content_kind, content_id_list, content):
    th = 0
    for content_id in content_id_list:
        th += len(content[content_id])
    avg_length = 1.0 * th / len(content_id_list)
    print("[INFO] {} content avg length is {}".format(content_kind, avg_length))

#quesition_answer_user, content_dic, title_dic, accept_answer_dic, user_context
def idReorder(question_answer_user_vote, body_dic, title_dic, accept_answer_dic, user_context, love_dic, max_love_count=config_data_preprocess.max_love_count):
    user_context_reorder = {}

    question = [line[0] for line in question_answer_user_vote]
    question_id_freq = list(zip(*np.unique(question, return_counts=True)))
    # remove question only have one answer

    print("[INFO] Question Answer pairs {}".format(len(question_answer_user_vote)))
    _index = 0
    t1 = 0
    question_dic = {}
    for id, freq in question_id_freq:
        #min question
        if freq >= 5:
            assert id not in question_dic, "question unique function is not right"
            question_dic[id] = _index
            _index += 1
        else:
            t1 += 1
    question_count = _index
    print("[INFO]question more than 5 answers {}, and {} has less 5 answers".format(question_count,t1 ))

    user = np.array([line[2] for line in question_answer_user_vote if line[0] in question_dic])
    user_id_unique = np.unique(user)

    user_length = len(user_id_unique)
    print("[INFO] user count {}".format(user_length))

    user_dic = {id: index for index, id in enumerate(user_id_unique)}

    answer = np.array([line[1] for line in question_answer_user_vote if line[0] in question_dic])
    answer_id = np.unique(answer)
    print("[INFO] answer count {}".format(len(answer_id)))

    answer_dic = {id: index + question_count for index, id in enumerate(answer_id)}

    remove_question_answer_user_vote = []
    for line_index in range(len(question_answer_user_vote)):
        question = question_answer_user_vote[line_index][0]
        if question in question_dic:
            question = question_dic[question] + user_length
            answer = answer_dic[question_answer_user_vote[line_index][1]] + user_length
            user = user_dic[question_answer_user_vote[line_index][2]]
            try:
                score = question_answer_user_vote[line_index][3]
            except:
                score = 0
            temp = [question, answer, user, score]
            remove_question_answer_user_vote.append(temp)


    one_answer_question_user = 0
    ti = 0
    content_pro = 0
    for user_id, context in user_context.items():
        try:
            user_context_reorder[user_dic[user_id]] = [answer_dic[i] + user_length for i in context if i in answer_dic]
        except:
            one_answer_question_user += 1

    assert  len(user_context_reorder) == len(user_dic), "length not equal {} != {}".format(len(user_context_reorder), len(user_dic))



    print("[INFO] {} user only answer question with one answer".format(one_answer_question_user))
    post_dic =  {**question_dic, **answer_dic}
    post_dic_sort = collections.OrderedDict(sorted(post_dic.items(), key=lambda x: x[1]))
    body_reorder = []

    question_dic_sort = collections.OrderedDict(sorted(question_dic.items(), key=lambda x:x[1]))
    title_reorder = []

    accept_answer_dic_reorder = {}
    for flag, (id, index) in enumerate(question_dic_sort.items()):
        assert flag == index, "[ERROR] Title reorder problem"
        title_reorder.append(title_dic[id])

    for flag, (id, index) in enumerate(post_dic_sort.items()):
        assert flag == index,"[ERROR] Content reorder problem"
        body_reorder.append(body_dic[id])

    content_evaluation("answer", list(range(question_count, len(body_reorder) - question_count)), body_reorder)
    content_evaluation("question", list(range(question_count)), body_reorder)
    content_evaluation("question title",list(range(question_count)), title_reorder)

    assert len(body_reorder) == len(post_dic), "[ERROR] Content length is not equal to answer + question"
    i = 0
    for id, index in question_dic.items():
        if id in accept_answer_dic:
            t = accept_answer_dic[id]

            try:
                accept_answer_dic_reorder[index] = answer_dic[t]
            except:
                i += 1
                continue
    print("[INFO] No accepted answer, question count {}".format(i))



    # love_list reorder
    love_list_count = reorde_lover_dic(love_dic=love_dic, user_dic=user_dic, user_count=user_length, max_love_count=max_love_count)
    # sparse_matrix_format = love_list2sparse_matrx(love_list, (user_length, user_length))



    return remove_question_answer_user_vote, body_reorder, user_context_reorder, accept_answer_dic_reorder, title_reorder, user_length, question_count, love_list_count






def read_xml_data(path):
    # hanle all the data under v3.2
    # for easy handle, we will read all the data and then random split data into "train, val, test"

    #TODO: handle multiple problem
    post_file = os.path.join(path, "Posts.xml")
    vote_file = os.path.join(path, "Votes.xml")
    quesition_answer_user_dic, body_dic, title_dic, accept_answer_dic, user_context = parse_post(post_file)

    vote_dic, love_dic = parse_vote(vote_file)


    for post_id, vote_count in vote_dic.items():
        try:
            quesition_answer_user_dic[post_id].append(vote_count)
        except:
            continue

    quesition_answer_user_vote = list(quesition_answer_user_dic.values())

    return quesition_answer_user_vote, body_dic, \
           title_dic, accept_answer_dic, user_context, love_dic




def main(path):
    print("STACK OVERFLOW")
    question_answer_user_vote, body, user_context, accept_answer_dic, title, user_count, question_count, love_list_count = idReorder(*(read_xml_data(path)))
    return question_answer_user_vote, body, user_context, accept_answer_dic, title, user_count, question_count,love_list_count