from sklearn.feature_extraction.text import TfidfTransformer
from scipy import spatial
import numpy as np
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

class TFIDFSimilar:
    def __init__(self, background_data):
        self.bc_data = background_data
        self.transfomer = self.get_idf()
        #TODO: Sparse matrix, Due to large number of words
    def get_idf(self):
        return TfidfTransformer.fit(self.bc_data)

    def simiarity(self, content, highRank):
        i = self.transfomer(content)
        j = self.transfomer(highRank)
        return 1 - spatial.distance.cosine(i,j)


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









