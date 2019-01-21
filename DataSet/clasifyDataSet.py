
import  torch.utils.data as data
import numpy as np

class clasifyDataSet(data.Dataset):

    def __init__(self,
                 G, user_count,
                 args,
                 Istraining=True
                 ):
        super(clasifyDataSet, self).__init__()
        self.G = G
        self.args = args
        self.user_count = user_count
        self.edges = self.train_edge() if Istraining else self.val_edge()
        #TODO: wast many storation to zeros
        # self.item_count = item_count
        # self.content_count = content_count
        # self.adj_neighbor, self.adj_edge, self.adj_strength = self.adjancy()



    # def adjancy(self):
    #     adj_neighbor = np.zeros((self.item_count + 1, self.args.max_degree),dtype=np.int64)
    #     adj_strength = np.zeros((self.item_count + 1, self.args.max_degree))
    #     adj_degree = np.zeros_like(adj_neighbor)
    #     adj_edge = np.zeros_like(adj_neighbor,dtype=np.int64)
    #
    #     for node in self.G.nodes():
    #         neighbors = np.array([neighbor for neighbor in self.G.neighbors[node]])
    #         adj_degree[node] = len(neighbors)
    #
    #         if len(neighbors) == 0:
    #             continue
    #         if len(neighbors) < self.args.max_degree:
    #             neighbors = np.random.choice(neighbors, self.args.max_degree, replace=True)
    #         else:
    #             neighbors = np.random.choice(neighbors, self.args.max_degree, replace=False)
    #
    #         for index, neigh_node in enumerate(neighbors):
    #             adj_edge[node, index] = self.G[node][neigh_node]['a_id']
    #             adj_strength[node, index] = self.G[node][neigh_node]['score']
    #
    #         neighbors[neighbors > self.content_count] = neighbors[neighbors > self.content_count] - self.content_count
    #         adj_neighbor[node] = neighbors
    #     return adj_neighbor, adj_edge, adj_strength

    def train_edge(self):
        return [e for e in self.G.edges(data=True) if not self.G[e[0]][e[1]]['train_removed']]

    def val_edge(self):
        return [e for e in self.G.edges(data=True) if self.G[e[0]][e[1]]['train_removed']]

    def __len__(self):
        return len(self.edges)

    def __getitem__(self, idx):
        edge = self.edges[idx]
        question = max(edge[0:2])
        user = min(edge[0:2])
        answer = self.G[edge[0]][edge[1]]['a_id'] - self.user_count
        score = self.G[edge[0]][edge[1]]['score'] - self.user_count


        return question, answer, user, score

