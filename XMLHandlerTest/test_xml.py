from XMLHandler_SemEval import xmlhandler
from preprocess import
import os
import numpy as np

def stats(question_answer_user_label):
    question = [line[0] for line in question_answer_user_label]
    question = np.unique(question)
    print("[INFO] Question: {}, QA pairs: {}".format(len(question), len(question_answer_user_label)))
if __name__ == '__main__':
    path = "/home/yichuan/course/induceiveAnswer/data/v3.2/train-more-for-subtaskA-from-2015"
    path1 = "/home/yichuan/course/induceiveAnswer/data/v3.2/train"
    path2 = "/home/yichuan/course/induceiveAnswer/data/v3.2/dev"
    train_file = [os.path.join(path, file) for file in os.listdir(path) if "train" in file and "multiline" not in file]
    train_file += [os.path.join(path1, file) for file in os.listdir(path1) if "taskA" in file and "multiline" not in file]

    test_file = [os.path.join(path, file) for file in os.listdir(path) if "test" in file and"multiline" not in file]
    dev_file = [os.path.join(path, file) for file in os.listdir(path) if "dev" in file and "multiline" not in file]
    dev_file += [os.path.join(path2, file) for file in os.listdir(path2) if "taskA" in file and "multiline" not in file]
    train_content, train_question_answer_user_label, train_user_post = xmlhandler.read_xml_data(train_file)
    test_content, test_question_answer_user_label, test_user_post = xmlhandler.read_xml_data(test_file)
    dev_content, dev_question_answer_user_label, dev_user_post = xmlhandler.read_xml_data(dev_file)
    stats(dev_question_answer_user_label)
    stats(train_question_answer_user_label)
    stats(test_question_answer_user_label)
