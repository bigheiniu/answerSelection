# S: empty set contians nothing

# id of each answer
import copy
import numpy as np
from scipy import spatial


def diversity(featureMatrix, relevanceScore, rankList, early_stop):
    oldS = []
    S = [rankList[0]]
    L = Lmatrix(featureMatrix, relevanceScore)
    while True:
        score = 0
        candidate = -1
        for i in rankList:
            temp_set = copy.deepcopy(S)
            temp_set.append(i)
            temp_score = det(temp_set, L)
            if (score < temp_score):
                candidate = i
                score = temp_score
            if candidate != -1:
                rankList.remove(candidate)
            else:
                break
            oldS = copy.deepcopy(S)
            S.append(candidate)
        if(det(S,L) - det(oldS,L) < early_stop):
            break

def Lmatrix(featureMatrix, relevanceScore):
    size_n = len(featureMatrix)
    L = np.zeros(size_n, size_n)
    for i in range(size_n):
        for j in range(size_n):
            L[i][j] = relevanceScore[i]*relevanceScore[j]*\
                      (1 - spatial.distance.cosine(featureMatrix[i], featureMatrix[j]))
            return L




def det(ids, L):
    K = L[np.ix_([ids],[ids])]
    score = np.linalg.det(K)
    return score