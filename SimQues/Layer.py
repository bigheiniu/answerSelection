from sklearn.feature_extraction.text import TfidfTransformer
from annoy import AnnoyIndex
import os
import itertools
import numpy as np
from gensim.models import LdaModel
from gensim.test.utils import datapath
from joblib import dump, load

class NeighborLocation:
    def __init__(self, feature_length, file_path, key_vector_list=None, build_tree=100, load_file=False):
        if load_file:
            self.annoy = AnnoyIndex(feature_length)
            self.annoy.load(file_path)
        else:
            self.annoy = AnnoyIndex(feature_length)
            for i in range(len(key_vector_list)):
                # try:
                self.annoy.add_item(i, key_vector_list[i])
                # except:
                #     th = key_vector_list[i]
                #     tj = th
            self.annoy.build(build_tree)
            # store annoy index
            self.annoy.save(file_path)

    def find_by_index(self, index, k):
        item_list = self.annoy.get_nns_by_item(index, k)
        return item_list

    def find_by_vector(self, feature_vec, k):
        item_list = self.annoy.get_nns_by_vector(feature_vec, k)
        return item_list

    def get_item(self, index):
        return self.annoy.get_item_vector(index)


class LDAKey:
    def __init__(self, background_data, topic_count, model_path, load_pretrain=False):
        file_name = "lda.model"
        self.corpus = [self.list2tuple(line) for line in background_data]
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(os.path.abspath(model_path), file_name)
        if load_pretrain:
            self.lda = self.loadModel(model_path)
        else:
            # corpus => (word_id, word_frequency)

            self.lda = LdaModel(self.corpus, num_topics=topic_count)
            self.saveModel(model_path)

    def list2tuple(self, data_list):
        data_list = np.array(data_list)
        y = np.bincount(data_list)

        ii = (np.nonzero(y))[0]
        return list(zip(ii, y[ii]))

    def saveModel(self, path):
        temp_file = datapath(path)
        self.lda.save(temp_file)

    def loadModel(self, path):
        temp_file = datapath(path)
        lda = LdaModel.load(temp_file)
        return lda

    def lda_by_index(self, index):
        corpus_vect = self.corpus[index]
        lda_vec = self.lda[corpus_vect]
        return lda_vec

    def lda_by_vec(self, document):
        corpus_vect = self.list2tuple(document)
        lda_vec = self.lda[corpus_vect]
        return lda_vec

class TFIDFKey:
    def __init__(self, background_data, model_path, load_pretrain=False):
        file_name = "tfidf.model"

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, file_name)
        if load_pretrain:
            self.tfModel = self.loadModel(model_path)
            self.n_features = (1, self.tfModel.idf_.shape[0])
        else:
            self.bc_vec = self.get_idf(background_data)
            self.tfModel = TfidfTransformer()
            self.bc_vec = self.tfModel.fit_transform(X=self.bc_vec)
            self.n_features = (1, self.tfModel.idf_.shape[0])
            self.saveModel(model_path)


    def saveModel(self, path):
        dump(self.tfModel, path)

    def loadModel(self, path):
        return load(path)

    def get_idf(self, background_data):
        content_count = len(background_data)
        max_item = np.max(background_data)
        base = np.zeros((content_count, max_item + 1))
        for index in range(content_count):
            item, count = np.unique(background_data[index], return_counts=True)
            for i, c in zip(item, count):
                base[index, i] = c
        return base

    def tfidf_by_index(self, index):
        doc_vec = self.bc_vec[index]
        return doc_vec

    def tfidf_by_vec(self, document):
        tf_idf_vec = np.zeros(self.n_features)
        for index in document:
            try:
                tf_idf_vec[0][index] += 1
            except:
                continue
        return tf_idf_vec









if __name__ == '__main__':
    content = np.random.randint(0, 100, (10, 1000))
    tfidf = TFIDFKey(background_data=content, model_path="./data/tf.m")
    th = tfidf.tfidf_by_index(1)
    t1 = th
    print(t1)