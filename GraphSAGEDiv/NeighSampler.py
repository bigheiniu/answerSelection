class UniformNeighborSampler:
    '''
    Uniformly sample neighbors.
    TODO: Assume that adj lists are padded with random re-sampling
    '''
    def __init__(self, adj_neighbor, adj_edge):
        self.neighbor_embed = adj_neighbor
        self.edge_embed = adj_edge

    def sample(self, batch_ids, number):
        '''

        :param batch_ids: batch of node ids
        :param number: how many number should be count
        :return: return next layer neighbors and the edge connecting them
        '''
        batch_ids_shape = [*batch_ids.shape]
        batch_ids_shape.append(number)
        batch_ids_shape = tuple(batch_ids_shape)
        batch_ids_new = batch_ids.contiguous().view(-1)
        neighbor_next_layer = (self.neighbor_embed[batch_ids_new][:,:number]).view(batch_ids_shape)
        edge_next_layer =(self.edge_embed[batch_ids_new][:,:number]).view(batch_ids_shape)

        return neighbor_next_layer, edge_next_layer
