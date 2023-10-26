import numpy as np


def transform_graph_permutation(x, adj):
    node_list = [i for i in range(len(x))]
    np.random.shuffle(node_list)

    features = x[node_list]
    adj = adj[:, node_list][:, :, node_list]
    return features, adj


def batch_graph_permutation(graph_li, return_sh_id=False):
    x = graph_li[0]
    adj = graph_li[1]
    node_list = [i for i in range(x.shape[1])]
    np.random.shuffle(node_list)

    features = x[:, node_list]
    adj = adj[:, :, node_list][:, :, :, node_list]

    if return_sh_id:
        return (features, adj), node_list
    else:
        return features, adj
