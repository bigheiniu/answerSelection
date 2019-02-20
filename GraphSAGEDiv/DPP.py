# S: empty set contians nothing

# id of each answer
import copy
import numpy as np
from scipy import spatial


def diversity(featureMatrix, relevanceScore, rankList, early_stop):

    if type(rankList) is not list:
        rankList = rankList.tolist()

    S = [rankList[0]]
    rankList.remove(S[0])
    L = Lmatrix(featureMatrix, relevanceScore)
    while len(rankList) > 0:
        score = 0
        candidate = -1
        for i in rankList:
            temp_list = copy.deepcopy(S)
            temp_list.append(i)
            temp_score = det(temp_list, L)
            if (score < temp_score):
                candidate = i
                score = temp_score
        if candidate != -1:
            rankList.remove(candidate)
        else:
            break
        oldS = copy.deepcopy(S)
        S.append(candidate)
        if(det(S,L) - det(oldS,L) <= early_stop):
            break
    if len(rankList) != 0 :
        S += rankList
    return S

def Lmatrix(featureMatrix, relevanceScore):
    size_n = len(featureMatrix)
    L = np.zeros((size_n, size_n))
    for i in range(size_n):
        for j in range(size_n):
            L[i][j] = relevanceScore[i]*relevanceScore[j]*\
                    (1 - spatial.distance.cosine(featureMatrix[i,:], featureMatrix[j,:]))
    return L

def det(ids, L):
    K = L[np.ix_(ids,ids)]
    score = np.linalg.det(K)
    return score

if __name__ == '__main__':
    feature_matrix = np.random.rand(10,20)
    rele = np.random.rand(10)
    L = Lmatrix(feature_matrix, rele)
    rankList = list(range(len(L)))
    th = diversity(feature_matrix, rele, rankList, 0.000001)
    print(th)