"""This module produces SemEval Task 3, Subtask B datasets in JSON."""



from XMLHandler.XMLHandler_StackOverflow.XMLpreprocessing import parse_post, parse_vote
import os
import  numpy as np
import collections
import gc


#quesition_answer_user_vote, content_dic, \
           #title_dic, accept_answer_dic, user_context

#quesition_answer_user, content_dic, title_dic, accept_answer_dic, user_context
def idReorder(question_answer_user_vote, body_dic, title_dic, accept_answer_dic, user_context):
    user_context_reorder = {}
    user= np.array([line[2] for line in question_answer_user_vote])
    user_id_ = np.unique(user)
    user_count = len(user_id_)
    user_dic = {id:index for index, id in enumerate(user_id_)}


    question = [line[0] for line in question_answer_user_vote]
    question_id = np.unique(question)
    question_dic = {id: index for index, id in enumerate(question_id)}
    question_count = len(question_id)

    answer = np.array([line[1] for line in question_answer_user_vote])


    answer_id = np.unique(answer)
    answer_dic = {id: index + question_count for index, id in enumerate(answer_id)}

    for line_index in range(len(question_answer_user_vote)):
        question_answer_user_vote[line_index][0] = question_dic[question[line_index]] + user_count
        question_answer_user_vote[line_index][1] = answer_dic[answer[line_index]] + user_count
        question_answer_user_vote[line_index][2] = user_dic[user[line_index]]
    for user_id, context in user_context.items():
        user_context_reorder[user_dic[user_id]] = context

    gc.collect()
    post_dic =  {**question_dic, **answer_dic}
    post_dic = collections.OrderedDict(sorted(post_dic.items(), key=lambda x: x[1]))
    body_reorder = []

    question_dic_sort = collections.OrderedDict(sorted(question_dic.items(), key=lambda x:x[1]))
    title_reorder = []

    accept_answer_dic_reorder = {}
    for flag, (id, index) in enumerate(question_dic_sort.items()):
        assert flag == index, "[ERROR] Title reorder problem"
        title_reorder.append(title_dic[id])

    for flag, (id, index) in enumerate(post_dic.items()):
        assert flag == index,"[ERROR] Content reorder problem"
        body_reorder.append(body_dic[id])

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

    return question_answer_user_vote, body_reorder, user_context_reorder, accept_answer_dic_reorder, title_reorder, user_count, question_count





def read_xml_data(path):
    # hanle all the data under v3.2
    # for easy handle, we will read all the data and then random split data into "train, val, test"

    #TODO: handle multiple problem
    post_file = os.path.join(path, "Posts.xml")
    vote_file = os.path.join(path, "Votes.xml")
    quesition_answer_user_dic, body_dic, title_dic, accept_answer_dic, user_context = parse_post(post_file)

    vote_dic = parse_vote(vote_file)


    for post_id, vote_count in vote_dic.items():
        try:
            quesition_answer_user_dic[post_id].append(vote_count)
        except:
            continue

    quesition_answer_user_vote = list(quesition_answer_user_dic.values())

    return quesition_answer_user_vote, body_dic, \
           title_dic, accept_answer_dic, user_context


def main(path):
    question_answer_user_vote, body, user_context, accept_answer_dic, title, user_count, question_count = idReorder(*(read_xml_data(path)))
    return question_answer_user_vote, body, user_context, accept_answer_dic, title, user_count, question_count