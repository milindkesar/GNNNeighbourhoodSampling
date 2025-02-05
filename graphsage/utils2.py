from graphsage.aggregators import MeanAggregator
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from _collections import defaultdict
import json
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from graphsage.walkshelper import generate_featureteleportwalks, generate_clusterteleportwalks
import random
from graphsage.lsh import train_lsh,get_nearest_neighbors
from graphsage.aggregators import MeanAggregator
from graphsage.planetoid import load_data

## Helper function for calculating F1 Score
def F1_score(y_true, y_pred):
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    # sum=0
    # for i in range(121):
    #     sum+= f1_score(y_true[:,i],y_pred[:,i],average="micro")
    return f1_score(y_true, y_pred, average="micro")


def get_average_lsh_added(adj_list, lsh_neighbours):
    average_for_lowdeg=[]
    average_added = []
    for node, neighbours in adj_list.items():
        if len(neighbours) < 5:
            average_for_lowdeg.append(len(set(lsh_neighbours[node])-set(adj_list[node])))
        average_added.append(len(set(lsh_neighbours[node])-set(adj_list[node])))
    return sum(average_added)/len(average_added), sum(average_for_lowdeg)/len(average_for_lowdeg)

## Helper function to get length of masks
def count_train_test_val(myarr):
    count=0
    for item in myarr:
        if item == True:
            count+=1
    return count

## Helper function to plot degree distribution from adjacency list represented as node-->neighbourlist
def adj_dis_plot(adj_list):
    sum=0
    max=0
    min=100
    myarr=[]
    for key,item in adj_list.items():
        myarr.append(len(list(item)))
        sum=sum+len(list(item))
        if len(list(item))>max:
            max=len(list(item))
        if len(list(item))<min:
            min=len(list(item))
    print(sum/len(adj_list))
    print('max',max)
    print('min',min)
    sns.distplot(myarr)
    plt.show()

## Helper function to calculate and write aggregated results if multiple iterations
def construct_agg(dir=None):
    try:
        os.makedirs(dir+'/agg')
    except:
        pass
    sub_dirs=[x[0] for x in os.walk(dir)]
    agg_training_info_l=[]
    agg_test_l=[]
    bigdata=[]
    best_test_f1={'epoch':-1,'test_f1_micro':-1, 'test_f1_macro':-1}
    for sub_dir in sub_dirs[1:]:
        if 'agg' in sub_dir:
            continue
        data=[]
        with open(sub_dir+'/test.txt') as f:
            for line in f:
                data.append(json.loads(line))
        bigdata.append(data)
    for i in range(len(bigdata[0])):
        agg_training_info = {'train_loss': [], 'val loss': []}
        agg_test = {'epoch':bigdata[0][i]['epoch'],'test_f1_micro': [], 'test_f1_macro':[]}
        for j in range(len(bigdata)):
            agg_test['test_f1_micro'].append(bigdata[j][i]['test_f1_micro'])
            agg_test['test_f1_macro'].append(bigdata[j][i]['test_f1_macro'])
        agg_test_l.append(agg_test)
    for item in agg_test_l:
        avg_test_f1=sum(item['test_f1_micro'])/len(item['test_f1_micro'])
        avg_test_accuracy = sum(item['test_f1_macro'])/len(item['test_f1_macro'])
        item['test_f1_micro'] = avg_test_f1
        item['test_f1_macro'] = avg_test_accuracy
        if avg_test_f1 > best_test_f1['test_f1_micro']:
            best_test_f1['test_f1_micro']=avg_test_f1
            best_test_f1['epoch']=item['epoch']
            best_test_f1['test_f1_macro']=avg_test_accuracy
    with open(dir+'/agg'+'/test.txt','a+') as out:
        for item in agg_test_l:
            out.write(json.dumps(item)+'\n')
    with open(dir+'/agg'+'/best.txt','a+') as out:
        out.write(json.dumps(best_test_f1))


def load_cora(lsh_helper, random_walk=False, root_folder='/home/thummala/graphsage-pytorch/datasets/cora',
              teleport_khop=False, augment_khop=False, dfactor=2, use_centroid=False, teleport=0.2, planetoid = True, load_embeds = False):
    def give_freq(t2):
        frequency = {}

        # iterating over the list
        for item in t2:
            # checking the element in dictionary
            if item in frequency:
                # incrementing the counr
                frequency[item] += 1
            else:
                # initializing the count
                frequency[item] = 1
        return frequency

    def get_adj_list(mydic):
        adj_lists = {}
        freq = {}
        for key, item in mydic.items():
            unique2 = []
            freq2 = {}
            for temp in item:
                for neigh in temp:
                    unique2.append(int(neigh))
            unique = set(unique2)
            freq2 = give_freq(unique2)
            adj_lists[int(key)] = unique
            freq[int(key)] = freq2
        return freq, adj_lists

    def generate_bias(path):
        kmeans = pickle.load(open(path, "rb"))
        centroids = kmeans.cluster_centers_
        distances = {}
        for i, c1 in enumerate(centroids):
            for j, c2 in enumerate(centroids):
                if not (i == j):
                    distances[(i, j)] = np.linalg.norm(c1 - c2)
                else:
                    distances[(i, j)] = 1
        feature_labels = kmeans.labels_
        return feature_labels, distances

    freq = {}
    dist_in_graph = {}
    ##Defining number of nodes etc
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    degrees = []
    train_mask = []
    test_mask = []
    val_mask = []
    lsh_neighbourlist_dic = {}
    if planetoid:
        print("Using planetoid split")
        adj_lists, feat_data, labels, train_mask, val_mask, test_mask = load_data('cora')
    else:
        with open(root_folder + "/cora.content") as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                feat_data[i, :] = list(map(float, info[1:-1]))
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels[i] = label_map[info[-1]]
    if use_centroid == True:
        cluster_labels, distances = generate_bias('/home/thummala/graphsage-pytorch/graphsage/corakmeans_7.pkl')
    else:
        cluster_labels = []
        distances = []
    if load_embeds:
        if not planetoid:
            feat_data = np.load('/home/thummala/graphsage-pytorch/res/cora/gswithembeds/allnodeembeddings.npy')
            print('loading gs embeddings as node features ', feat_data.shape)
        else:
            feat_data = np.load('/home/thummala/graphsage-pytorch/res/cora/gsplanetoid_2020/allnodeembeddings.npy')
            print('loading gs embeddings as node features ', feat_data.shape)
    if teleport_khop or augment_khop:
        print('creating lsh')
        lsh_cand_dic = return_lsh_candidates(np.array(feat_data), n_vectors=lsh_helper['n_vectors'],
                                             num_lsh_neighbours=lsh_helper['num_lsh_neighbours'],
                                             atleast=lsh_helper['atleast'], search_radius=lsh_helper['search_radius'])
        print('done', lsh_cand_dic[0])
    else:
        lsh_cand_dic = {}

    if random_walk:
        with open(root_folder + "/Corawalks_10_30_1_1.json", "r") as outfile:
            mydic2 = json.load(outfile)
        freq, adj_lists = get_adj_list(mydic2)
        with open(root_folder + "/CoraDist_10_30_1_1.json", "r") as outfile:
            dist_in_graph = json.load(outfile)
    else:
        if not planetoid:
            print("Not using planetoid split")
            adj_lists = defaultdict(set)
            with open(root_folder + "/cora.cites") as fp:
                for i, line in enumerate(fp):
                    info = line.strip().split()
                    paper1 = node_map[info[0]]
                    paper2 = node_map[info[1]]
                    adj_lists[paper1].add(paper2)
                    adj_lists[paper2].add(paper1)
    if augment_khop:
        for key, value in adj_lists.items():
            node = int(key)
            if np.all((feat_data[node] == 0)):
                lsh_neighbourlist_dic[node] = []
            else:
                lsh_neighbourlist_dic[node] = list(lsh_cand_dic[node].keys())
    if teleport_khop:
        adj_lists = Do_Teleport_Khop(adj_list=adj_lists, lsh_cand_dic=lsh_cand_dic, feats=feat_data, dfactor=dfactor)

    for i in range(num_nodes):
        degrees.append(len(list(adj_lists[i])))

    data_loader_dic = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'train_mask': train_mask,
                       'test_mask': test_mask, 'val_mask': val_mask, 'distances': distances,
                       'cluster_labels': cluster_labels, 'freq': freq, 'dist_in_graph': dist_in_graph,
                       'centralityev': [], 'centralitybtw': [], 'centralityh': [],
                       'centralityd': degrees, 'lsh_neighbour_list': lsh_neighbourlist_dic}

    return data_loader_dic



# function to return candidate nodes based on lsh
# def return_lsh_candidates(features, n_vectors=16, search_radius = 3, num_lsh_neighbours = 10, atleast = False, includeNeighbourhood = False, adj_list = None):
#
#     if includeNeighbourhood:
#         neighbourhood_features = []
#         aggregator = MeanAggregator(features=features)
#         nodes = list(range(features.shape[0]))
#         ##get feature vector of neighbourhood
#         neigh_feats = aggregator.forward(nodes, [adj_list[int(node)] for node in nodes],
#                                               num_sample=None,
#                                               lsh_neighbours={},
#                                               n_lsh_neighbours=None, lsh_augment=False)
#         ##concatenate features with neigh_feats
#         concat_features = np.concatenate((features,neigh_feats),axis = 1)
#         model = train_lsh(concat_features, n_vectors)
#
#     else:
#         model = train_lsh(features, n_vectors)
#     ## copy feature vector for further use
#     if includeNeighbourhood:
#         query_vectors = np.concatenate((features,features),dim=1)
#         features_copy= np.copy(concat_features)
#     else:
#         query_vectors = np.copy(features)
#         features_copy = np.copy(features)
#     lsh_candidates_dic = {}
#     for item_id in range(features_copy.shape[0]):
#         lsh_candidates_dic[item_id] = {}
#         query_vector = query_vectors[item_id]
#         nearest_neighbors = get_nearest_neighbors(features_copy, query_vector.reshape(1, -1), model, max_search_radius=search_radius)
#         count = 0
#         if atleast:
#             if len(nearest_neighbors) < num_lsh_neighbours:
#                 radius = search_radius + 1
#                 while True:
#                     nearest_neighbors = get_nearest_neighbors(features_copy, query_vector.reshape(1, -1), model, max_search_radius=radius)
#                     if (len(nearest_neighbors) > num_lsh_neighbours) or (radius >= n_vectors//2):
#                         break
#                     radius = radius + 1
#         for i, row in nearest_neighbors[:num_lsh_neighbours+1].iterrows():
#             if count == num_lsh_neighbours:
#                 break
#             if int(item_id) == int(row['id']):
#                 continue
#             count=count+1
#             lsh_candidates_dic[item_id][int(row['id'])] = row['similarity']
#     return lsh_candidates_dic

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


# {'num_vectors':16, 'search_radius': False, 'num_lsh_neighbours': 10,'atleast': False}
def Do_Teleport_Khop(adj_list, lsh_cand_dic, feats, dfactor):
    # helper function
    mod_adj_list = adj_list.copy()

    def flip(p):
        return True if random.random() < p else False

    teleport_count = 0
    for node, node_feature in enumerate(list(feats)):
        if len(set(adj_list[node])) < 10:
            if len(set(adj_list[node])) == 0:
                tel_prob = 1
            else:
                tel_prob = 1 / (dfactor * len(set(adj_list[node])))
            if flip(tel_prob):
                t = list(lsh_cand_dic[node].values())
                if sum(t) == 0:
                    print('problem encountered at node ', node,
                          ' lsh returned zero similarity with candidate nodes')
                    mod_adj_list[node] = set(adj_list[node])
                    print('the adjacency list of problem node is ', adj_list[node])
                else:
                    prob_weights = list(t / sum(t))
                    jump_node = np.random.choice(list(lsh_cand_dic[node].keys()), p=prob_weights)
                    if adj_list[node] != adj_list[jump_node]:
                        teleport_count += 1
                    mod_adj_list[node] = set(adj_list[jump_node])
            else:
                mod_adj_list[node] = set(adj_list[node])
        else:
            mod_adj_list[node] = set(adj_list[node])
    print("teleport count: ", teleport_count)
    return mod_adj_list


## Loader for wikics
def load_wikics(lsh_helper, random_walk=False, type_walk='default', p=1, q=1, num_walks=10, walk_length=10,
                teleport=0.2, workers=1,
                teleport_khop=False, augment_khop=False, dfactor=2, use_centroid=False, load_embeds = False):
    # function to format neighbours in random walk dic stored as string
    print("Loading WikiCS")
    wikics_loader_dic = {}

    def format_dic(mystring):
        mystring = mystring.replace('{', '')
        mystring = mystring.replace('}', '')
        myarr = mystring.split(',')
        if myarr == ['set()']:
            return set()
        myarr = [int(x) for x in myarr]
        return set(myarr)

    # function to format frequency and distance in graph etc
    def format_dic2(mystring):
        mystring = mystring.replace('{', '')
        mystring = mystring.replace('}', '')
        myarr = mystring.split(',')
        res = {}
        if myarr == ['']:
            return res
        for item in myarr:
            els = item.split(':')
            try:
                res[int(els[0].replace("'", ''))] = int(els[1])
            except:
                res[int(els[0])] = int(els[1])
        return res

    # function to generate distances in centroids (can be edited for different distances)
    def generate_bias(centroids):
        distances = {}
        for i, c1 in enumerate(centroids):
            for j, c2 in enumerate(centroids):
                if not (i == j):
                    distances[(i, j)] = np.linalg.norm(c1 - c2) * 100
                else:
                    distances[(i, j)] = 1
        return distances

    ##Root Folder
    root_folder = '/home/thummala/graphsage-pytorch/datasets/Dataset-WikiCS'

    ##Loading the Data
    with open(root_folder + "/data.json", "r") as file:
        wikics = json.load(file)
    print('loading dataframe of')
    if type_walk == 'default':
        print('Loading normal random walks')
        try:
            node_df = pd.read_csv(root_folder + '/wikics_nodeinfo.csv')
        except:
            node_df = generate_featureteleportwalks(data2=wikics, p=p, q=q, num_walks=num_walks, walklength=walk_length,
                                                    teleport_weight=teleport, root_folder=root_folder, workers=workers)
    elif type_walk == 'clusterteleport':
        print('Loading cluster teleport walks')
        node_df = pd.read_csv(root_folder + '/wikics_nodeinfo_clusterrw.csv')
    elif type_walk == 'featureteleport':
        print('Loading feature teleport walks')
        node_df = pd.read_csv(root_folder + '/wikics_nodeinfo_featureteleportrw.csv')
    elif type_walk == 'customfeatureteleport':
        print('generating feature teleport walks with parameters specified: p=' + str(p) + ' q=' + str(
            q) + ' walklength=' + str(
            walk_length) + ' num_walks=' + str(num_walks) + ' teleport=' + str(teleport))
        node_df = generate_featureteleportwalks(data2=wikics, p=p, q=q, num_walks=num_walks, walklength=walk_length,
                                                teleport_weight=teleport, root_folder=root_folder, workers=workers)
    elif type_walk == 'customclusterteleport':
        print('generating cluster teleport walks with parameters specified: p=' + str(p) + ' q=' + str(
            q) + ' walklength=' + str(
            walk_length) + ' num_walks=' + str(num_walks) + ' teleport=' + str(teleport))
        node_df = generate_clusterteleportwalks(data2=wikics, p=p, q=q, num_walks=num_walks, walklength=walk_length,
                                                teleport_weight=teleport, root_folder=root_folder, workers=workers,
                                                n_clusters=14)
    else:
        print('Specify correct type:- default, clusterteleport or featureteleport-->loading default now')
        node_df = pd.read_csv(root_folder + '/wikics_nodeinfo.csv')
    feat_data = wikics['features']
    if load_embeds:
        feat_data = np.load('/home/thummala/graphsage-pytorch/res/wikics/gs_2020_final/allnodeembeddings.npy')
        print('loading gs embeddings as node features ',feat_data.shape)
    labels = wikics['labels']
    train_mask, val_mask, test_mask = wikics['train_masks'][10], wikics['val_masks'][10], wikics['test_mask']
    centralityev = [0] * len(train_mask)
    centralityd = [0] * len(train_mask)
    centralitybtw = [0] * len(train_mask)
    centralityh = [0] * len(train_mask)
    lsh_neighbourlist_dic = {}
    adj_list = {}
    freq = {}
    dist_in_graph = {}
    if use_centroid:
        centroids = np.load(root_folder + '/centroids_14.npy')
        distances = generate_bias(centroids)
        cluster_labels = node_df['cluster_labels']
    else:
        centroids = []
        cluster_labels = []
        distances = []
    if random_walk:
        print('loading random walk samples')
        for i, row in node_df.iterrows():
            node = int(row['nodes'])
            adj_list[node] = format_dic(row['randomwalkneigh'])
            freq[node] = format_dic2(row['freq_in_randomwalk'])
            dist_in_graph[node] = format_dic2(row['dist_in_graph'])
            centralityev[node] = row['eigenvectorc']
            centralitybtw[node] = row['betweennessc']
            centralityd[node] = row['degreec']
            centralityh[node] = row['harmonicc']
    if not random_walk:
        print('loading khop neighbours')
        teleport_count = 0
        for i, row in node_df.iterrows():
            node = int(row['nodes'])
            adj_list[node] = set(wikics['links'][node])
            centralityev[node] = row['eigenvectorc']
            centralitybtw[node] = row['betweennessc']
            centralityd[node] = row['degreec']
            centralityh[node] = row['harmonicc']
        if teleport_khop or augment_khop:
            print('creating lsh')
            lsh_cand_dic = return_lsh_candidates(np.array(feat_data), n_vectors=lsh_helper['n_vectors'],
                                                 num_lsh_neighbours=lsh_helper['num_lsh_neighbours'],
                                                 atleast=lsh_helper['atleast'],
                                                 search_radius=lsh_helper['search_radius'], includeNeighbourhood=lsh_helper['includeNeighbourhood'], adj_list=adj_list)
            print('done')
        else:
            lsh_cand_dic = {}
        if augment_khop:
            for i, row in node_df.iterrows():
                node = int(row['nodes'])
                lsh_neighbourlist_dic[node] = list(lsh_cand_dic[node].keys())
        if teleport_khop:
            adj_list = Do_Teleport_Khop(adj_list=adj_list, lsh_cand_dic=lsh_cand_dic, feats=feat_data, dfactor=dfactor)
    wikics_loader_dic = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_list, 'train_mask': train_mask,
                         'test_mask': test_mask, 'val_mask': val_mask, 'distances': distances,
                         'cluster_labels': cluster_labels, 'freq': freq, 'dist_in_graph': dist_in_graph,
                         'centralityev': centralityev, 'centralitybtw': centralitybtw, 'centralityh': centralityh,
                         'centralityd': centralityd, 'lsh_neighbour_list': lsh_neighbourlist_dic}
    return wikics_loader_dic


def custom_load_pubmed(lsh_helper, random_walk=False, root_folder='/home/thummala/graphsage-pytorch/datasets/pubmed-data',
                teleport_khop=False, augment_khop=False, dfactor=2, use_centroid=False, teleport=0.2, planetoid = True, load_embeds = True):
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    degrees = []
    train_mask = []
    test_mask = []
    val_mask = []
    lsh_neighbourlist_dic = {}
    if planetoid:
        adj_lists, feat_data, labels, train_mask, val_mask, test_mask = load_data('pubmed')
    else:
        with open(root_folder + "/Pubmed-Diabetes.NODE.paper.tab") as fp:
            fp.readline()
            feat_map = {entry.split(":")[1]: i - 1 for i, entry in enumerate(fp.readline().split("\t"))}
            for i, line in enumerate(fp):
                info = line.split("\t")
                node_map[info[0]] = i
                labels[i] = int(info[1].split("=")[1]) - 1
                for word_info in info[2:-1]:
                    word_info = word_info.split("=")
                    feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
        adj_lists = defaultdict(set)
        with open(root_folder + "/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
            fp.readline()
            fp.readline()
            for line in fp:
                info = line.strip().split("\t")
                paper1 = node_map[info[1].split(":")[1]]
                paper2 = node_map[info[-1].split(":")[1]]
                adj_lists[paper1].add(paper2)
                adj_lists[paper2].add(paper1)
        print('using random splits...')
        np.random.seed(1)
        rand_indices = np.random.permutation(num_nodes)
        test = rand_indices[:int(0.2 * num_nodes)]
        val = rand_indices[int(0.2 * num_nodes):int(0.3 * num_nodes)]
        train = list(rand_indices[int(0.3 * num_nodes):])
        for node in range(num_nodes):
            if node in train:
                train_mask.append(True)
                val_mask.append(False)
                test_mask.append(False)
            elif node in val:
                train_mask.append(False)
                val_mask.append(True)
                test_mask.append(False)
            elif node in test:
                train_mask.append(False)
                val_mask.append(False)
                test_mask.append(True)
    if load_embeds:
        try:
            allnodesembeddingpath = '/home/thummala/graphsage-pytorch/res/pubmed/gcn2020planetoid/allnodeembeddings.npy'
            feat_data = np.load(allnodesembeddingpath)
            print('loading gs embeddings as node features ', feat_data.shape)
            print('loading from ',allnodesembeddingpath)
        except:
            print("Unable to load emmbeddings...working with input feature data")
    if teleport_khop or augment_khop:
        print('creating lsh')
        lsh_cand_dic = return_lsh_candidates(np.array(feat_data), n_vectors=lsh_helper['n_vectors'],
                                             num_lsh_neighbours=lsh_helper['num_lsh_neighbours'],
                                             atleast=lsh_helper['atleast'], search_radius=lsh_helper['search_radius'])
        print('done')
    else:
        lsh_cand_dic = {}

    for key, value in adj_lists.items():
        node = int(key)
        if augment_khop:
            if np.all((feat_data[node] == 0)):
                lsh_neighbourlist_dic[node] = []
            else:
                lsh_neighbourlist_dic[node] = list(lsh_cand_dic[node].keys())

    if teleport_khop:
        adj_lists = Do_Teleport_Khop(adj_list=adj_lists, lsh_cand_dic=lsh_cand_dic, feats=feat_data, dfactor=dfactor)

    for i in range(num_nodes):
        degrees.append(len(list(adj_lists[i])))

    data_loader_dic = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_lists, 'train_mask': train_mask,
                       'test_mask': test_mask, 'val_mask': val_mask, 'distances': [],
                       'cluster_labels': [], 'freq': {}, 'dist_in_graph': {},
                       'centralityev': [], 'centralitybtw': [], 'centralityh': [],
                       'centralityd': degrees, 'lsh_neighbour_list': lsh_neighbourlist_dic}

    return data_loader_dic


## Loader for ppi
def load_ppi(lsh_helper, random_walk=False, root_folder='/home/thummala/graph-datasets/Dataset-PPI/ppi',
             teleport_khop=False, augment_khop=False, dfactor=2, use_centroid=False, teleport=0.2, load_embeds = False):
    # function to format neighbours in random walk dic stored as string
    print("Loading PPI Dataset", "random_walk", random_walk)

    def format_dic(mystring):
        mystring = mystring.replace('{', '')
        mystring = mystring.replace('}', '')
        myarr = mystring.split(',')
        if myarr == ['set()']:
            return set()
        myarr = [int(x) for x in myarr]
        return set(myarr)

    # function to format frequency and distance in graph etc
    def format_dic2(mystring):
        mystring = mystring.replace('{', '')
        mystring = mystring.replace('}', '')
        myarr = mystring.split(',')
        res = {}
        if myarr == ['']:
            return res
        for item in myarr:
            els = item.split(':')
            try:
                res[int(els[0].replace("'", ''))] = int(els[1])
            except:
                res[int(els[0])] = int(els[1])
        return res

    def generate_bias(centroids):
        distances = {}
        for i, c1 in enumerate(centroids):
            for j, c2 in enumerate(centroids):
                if not (i == j):
                    distances[(i, j)] = np.linalg.norm(c1 - c2) * 100
                else:
                    distances[(i, j)] = 1
        return distances

    def remove_self(adj_list):
        for key in list(adj_list.keys()):
            adj_list[key] = set([x for x in list(adj_list[key]) if x != key])
        return adj_list

    ppi_data_dic = {}
    num_nodes = 56944
    adj_list = {}
    freq = {}
    dist_in_graph = {}
    lsh_neighbourlist_dic = {}
    node_df = pd.read_csv(root_folder + '/node_info.csv')
    with open(root_folder + '/ppi-G.json', ) as F:
        data = json.load(F)
    # data.keys()
    with open(root_folder + '/ppi-id_map.json', ) as F:
        id_maps = json.load(F)
    with open(root_folder + '/ppi-class_map.json', ) as F:
        label_map = json.load(F)
    temp = [0] * len(label_map)
    for key in label_map.keys():
        temp[int(key)] = label_map[key]
    labels = np.array(temp)
    feat_data = np.load(root_folder + '/ppi-feats.npy')
    if load_embeds:
        feat_data = np.load('/home/thummala/graphsage-pytorch/res/ppi/gswithembeds/allnodeembeddings.npy')
        print('loading gs embeddings as node features ', feat_data.shape)
    nodedf = pd.read_csv(root_folder + '/node_info.csv')

    if use_centroid:
        centroids = np.load(root_folder + '/centroids_kmeans_39.npy')
        distances = generate_bias(centroids)
        cluster_labels = node_df['cluster_labels']
    else:
        centroids = []
        distances = []
        cluster_labels = []

    type_ = node_df['type']
    centralityev = [0] * num_nodes
    centralityd = list(node_df['degreec'])
    centralitybtw = [0] * num_nodes
    centralityh = list(node_df['harmonicc'])
    if random_walk:
        print("Random Walk Samples")
        for i, row in node_df.iterrows():
            node = int(row['nodes'])
            adj_list[node] = format_dic(row['randomwalkneigh'])
            freq[node] = format_dic2(row['freq_in_randomwalk'])
            dist_in_graph[node] = format_dic2(row['dist_in_graph'])
        adj_list = remove_self(adj_list)
    if not random_walk:
        print("K-Hop Neighbours")
        if teleport_khop or augment_khop:
            print('creating lsh')
            lsh_cand_dic = return_lsh_candidates(np.array(feat_data), n_vectors=lsh_helper['n_vectors'],
                                                 num_lsh_neighbours=lsh_helper['num_lsh_neighbours'],
                                                 atleast=lsh_helper['atleast'],
                                                 search_radius=lsh_helper['search_radius'])
            print('done')
        else:
            lsh_cand_dic = {}
        for i, row in node_df.iterrows():
            node = int(row['nodes'])
            adj_list[node] = set()
            if augment_khop:
                if np.all((feat_data[node] == 0)):
                    lsh_neighbourlist_dic[node] = []
                else:
                    lsh_neighbourlist_dic[node] = list(lsh_cand_dic[node].keys())

        for edge in data['links']:
            adj_list[int(edge['source'])].add(int(edge['target']))
            adj_list[int(edge['target'])].add(int(edge['source']))

        if teleport_khop:
            adj_list = Do_Teleport_Khop(adj_list=adj_list, lsh_cand_dic=lsh_cand_dic, feats=feat_data, dfactor=dfactor)

    train_mask, val_mask, test_mask = [], [], []
    for t in type_:
        if t == 'train':
            train_mask.append(True)
            test_mask.append(False)
            val_mask.append(False)
        elif t == 'val':
            train_mask.append(False)
            test_mask.append(False)
            val_mask.append(True)
        elif t == 'test':
            train_mask.append(False)
            test_mask.append(True)
            val_mask.append(False)
        else:
            print('problem in type_')
    data_loader_dic = {'feat_data': feat_data, 'labels': labels, 'adj_lists': adj_list, 'train_mask': train_mask,
                       'test_mask': test_mask, 'val_mask': val_mask, 'distances': distances,
                       'cluster_labels': cluster_labels, 'freq': freq, 'dist_in_graph': dist_in_graph,
                       'centralityev': centralityev, 'centralitybtw': centralitybtw, 'centralityh': centralityh,
                       'centralityd': centralityd, 'lsh_neighbour_list': lsh_neighbourlist_dic}
    return data_loader_dic