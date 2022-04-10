import networkx as nx
from graphsage.random_walks import ClusterTeleport_RandomWalk, FeatureTeleport_RandomWalk
from graphsage.walkshelper import generate_clusterteleportwalks
import numpy as np
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from graphsage.walkshelper import generate_featureteleportwalks, generate_clusterteleportwalks
import random
from graphsage.lsh import train_lsh,get_nearest_neighbors
from collections import defaultdict
from graphsage.aggregators import MeanAggregator
from graphsage.utils2 import load_wikics
import torch
import torch.nn as nn

def return_lsh_candidates(features, n_vectors=16, search_radius=3, num_lsh_neighbours=10, atleast=False,
                          includeNeighbourhood=False, adj_list=None):
    if includeNeighbourhood:
        neighbourhood_features = []
        features_embedding = nn.Embedding(features.shape[0], features.shape[1])
        features_embedding.weight = nn.Parameter(torch.FloatTensor(features), requires_grad=False)
        aggregator = MeanAggregator(features=features_embedding)
        nodes = list(range(features.shape[0]))
        ##get feature vector of neighbourhood
        neigh_feats = aggregator.forward(nodes, [adj_list[int(node)] for node in nodes],
                                         num_sample=None,
                                         lsh_neighbours={},
                                         n_lsh_neighbours=None, lsh_augment=False)
        neigh_feats = neigh_feats.detach().numpy()
        print('neigh_feats shape ', neigh_feats.shape)
        ##concatenate features with neigh_feats
        concat_features = np.concatenate((features, neigh_feats), axis=1)
        model = train_lsh(concat_features, n_vectors)

    else:
        model = train_lsh(features, n_vectors)
    ## copy feature vector for further use
    if includeNeighbourhood:
        query_vectors = np.concatenate((features, features), axis=1)
        features_copy = np.copy(concat_features)
    else:
        query_vectors = np.copy(features)
        features_copy = np.copy(features)
    print('features copy shape ', features_copy.shape)
    lsh_candidates_dic = {}
    for item_id in range(features_copy.shape[0]):
        lsh_candidates_dic[item_id] = {}
        query_vector = query_vectors[item_id]
        # print('shape of query vector',query_vector.shape)
        nearest_neighbors = get_nearest_neighbors(features_copy, query_vector.reshape(1, -1), model,
                                                  max_search_radius=search_radius)
        count = 0
        if atleast:
            if len(nearest_neighbors) < num_lsh_neighbours:
                radius = search_radius + 1
                while True:
                    nearest_neighbors = get_nearest_neighbors(features_copy, query_vector.reshape(1, -1), model,
                                                              max_search_radius=radius)
                    if (len(nearest_neighbors) > num_lsh_neighbours) or (radius >= n_vectors // 2):
                        break
                    radius = radius + 1

        if len(nearest_neighbors) == 0:
            lsh_candidates_dic[item_id] = {}
        else:
            for i, row in nearest_neighbors[:num_lsh_neighbours + 1].iterrows():
                if count == num_lsh_neighbours:
                    break
                if int(item_id) == int(row['id']):
                    continue
                count = count + 1
                lsh_candidates_dic[item_id][int(row['id'])] = row['similarity']
    return lsh_candidates_dic


lsh_helper = {'n_vectors':16, 'search_radius': 2, 'num_lsh_neighbours': 10,'atleast': True}
data_dic = load_wikics(lsh_helper)

return_lsh_candidates(np.array(data_dic['feat_data']),adj_list = data_dic['adj_lists'], includeNeighbourhood=True)
#
# karate_g = nx.read_edgelist('../datasets/karate/karate.edgelist')
# node_mapping = {}
# for i,node in enumerate(list(karate_g.nodes())):
#     node_mapping[node] = str(i)
# karate_g = nx.relabel_nodes(karate_g, node_mapping)
# features = np.random.rand(len(karate_g.nodes()),10)
# cluster_labels={}
# for node in karate_g.nodes():
#     cluster_labels[node]=random.choice([1,2,3])
# print(cluster_labels)
# random_walk = ClusterTeleport_RandomWalk(karate_g, walk_length=10, num_walks=20, p=1, q=0.01, workers=1, cluster_labels=list(cluster_labels.values()), clusterteleport_weight=.1)
#
# walklist = random_walk.walks
#
# for w in walklist:
#     if w[0] == 25:
#         print(w)