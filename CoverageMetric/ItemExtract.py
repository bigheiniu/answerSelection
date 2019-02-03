from sklearn.feature_extraction.text import TfidfTransformer
from scipy import spatial

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
    def __init__(self):
        pass

    def similarity(self):
        pass








