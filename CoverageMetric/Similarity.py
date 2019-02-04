from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy import spatial
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

class TFIDFSimilar:
    def __init__(self, background_data):
        self.bc_data = back_ground_data
        self.count = self.get_idf()

        self.tfModel = TfidfTransformer()
        self.tfModel.fit(X=self.count)

        #TODO: Sparse matrix, Due to large number of words
    def get_idf(self):
        item, count = np.unique(self.bc_data, return_counts=True)
        count = count.reshape(len(count), 1)
        item = item.reshape(len(item), 1)
        base = np.zeros((1, np.max(item) + 1))
        for i, c  in zip(item, count):
            base[0,i] = c
        return base

    def simiarity(self, content, highRank):
        i = self.tfModel.transform(content)
        j = self.tfModel.transform(highRank)
        return 1 - spatial.distance.cosine(i,j)


if __name__ == '__main__':
    back_ground_data = np.random.randint(0, 1000, (1, 5000))
    tfidf = TFIDFSimilar(back_ground_data)
    content = np.random.randint(0, 1000, (1, 30))
    highRank = np.random.randint(0, 1000, (1, 20))
    print("score is {}".format(tfidf.simiarity(content, highRank)))

class LDAsimilarity:
    def __init__(self, background_data, model_path, topic_count):
        count = []
        for i in background_data:
            unique, counts = np.unique(i, return_counts=True)
            count.append(np.asarray((unique, counts)))
        self.ldaModel = gensim.models.wrappers.LdaMallet(model_path, corpus=count, num_topics=topic_count)
        self.topic_dis = self.ldaModel[background_data]

    def similarity(self, content_id, target_id):
        content_topic = self.topic_dis[content_id]
        target_topic = self.topic_dis[target_id]
        return 1 - spatial.distance.cosine(content_topic, target_topic)









