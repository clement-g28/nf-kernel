import numpy as np


def transform_graph_permutation(x, adj):
    node_list = [i for i in range(len(x))]
    np.random.shuffle(node_list)

    features = x[node_list]
    adj = adj[:, node_list][:, :, node_list]
    return features, adj
