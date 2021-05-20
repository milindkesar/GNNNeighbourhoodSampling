from graphsage.aggregators import MeanAggregator
import numpy as np
import pandas as pd
import torch.nn as nn
import torch
from _collections import defaultdict
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
#
# features=nn.Embedding(8,5)
# features.weight=nn.Parameter(torch.rand(8,5),requires_grad=False)
# adj_list=defaultdict(set)
# adj_list[1].add(5)
# adj_list[1].add(2)
# adj_list[5].add(8)
# adj_list[5].add(1)
# adj_list[2].add(1)
# adj_list[2].add(8)
# adj_list[2].add(6)
# adj_list[2].add(3)
# adj_list[8].add(5)
# adj_list[8].add(2)
# adj_list[6].add(2)
# adj_list[6].add(7)
# adj_list[7].add(6)
# adj_list[3].add(2)
# adj_list[3].add(4)
# adj_list[4].add(3)
# print(features.weight)
# print(features(torch.LongTensor([1,2,3,4,5,6,7])))
# nodes=[1,2,3,4,5,6,7,8]
# agg1=MeanAggregator(features,gcn=False)
# neig_feat=agg1.forward(nodes,[adj_list[node] for node in nodes])
# print(neig_feat)
#
def custom_load_cora():
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
            # print(freq2)
            # print(unique)
            adj_lists[int(key)] = unique
            #print(adj_lists[key])
            freq[int(key)] = freq2
        return freq, adj_lists

    def generate_bias(path):
        kmeans = pickle.load(open(path, "rb"))
        centroids = kmeans.cluster_centers_
        distances = {}
        # print(set(kmeans.labels_))
        # print(centroids)
        for i, c1 in enumerate(centroids):
            for j, c2 in enumerate(centroids):
                if not (i == j):
                    distances[(i, j)] = np.linalg.norm(c1 - c2)
                else:
                    distances[(i, j)] = 1
        feature_labels = kmeans.labels_
        return feature_labels, distances

    feature_labels, distances = generate_bias('/home/thummala/graphsage-pytorch/graphsage/corakmeans_7.pkl')

    with open("/home/thummala/graphsage-pytorch/cora/Corawalks_10_30_1_1.json", "r") as outfile:
        mydic2 = json.load(outfile)
    freq, adj_lists2 = get_adj_list(mydic2)
    with open("/home/thummala/graphsage-pytorch/cora/CoraDist_10_30_1_1.json","r") as outfile:
        dist_in_graph = json.load(outfile)
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    degrees = []
    with open("/home/thummala/graphsage-pytorch/cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]
    for i in range(num_nodes):
        degrees.append(len(list(adj_lists2[i])))



    return feat_data, labels, adj_lists2,dist_in_graph,freq,feature_labels, distances,degrees

## Function to load cora in the hop fashion
def custom_load_cora2():
    def load_cora():
        num_nodes = 2708
        num_feats = 1433
        feat_data = np.zeros((num_nodes, num_feats))
        labels = np.empty((num_nodes, 1), dtype=np.int64)
        node_map = {}
        label_map = {}
        with open("/home/thummala/graphsage-pytorch/cora/cora.content") as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                feat_data[i, :] = list(map(float, info[1:-1]))
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels[i] = label_map[info[-1]]

        adj_lists = defaultdict(set)
        with open("/home/thummala/graphsage-pytorch/cora/cora.cites") as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                adj_lists[paper1].add(paper2)
                adj_lists[paper2].add(paper1)
        return feat_data, labels, adj_lists

    feat_data, labels, adj_lists2 = load_cora()
    freq={}
    dist_in_graph={}
    for key in adj_lists2.keys():
        unique={}
        unique2={}
        for item in adj_lists2[key]:
            unique[int(item)]=1
            unique2[str(item)]=1
        freq[int(key)]=unique
        dist_in_graph[str(key)]=unique2
    def generate_bias(path):
        kmeans = pickle.load(open(path, "rb"))
        centroids = kmeans.cluster_centers_
        distances = {}
        # print(set(kmeans.labels_))
        # print(centroids)
        for i, c1 in enumerate(centroids):
            for j, c2 in enumerate(centroids):
                if not (i == j):
                    distances[(i, j)] = np.linalg.norm(c1 - c2)
                else:
                    distances[(i, j)] = 1
        feature_labels = kmeans.labels_
        return feature_labels, distances

    feature_labels, distances = generate_bias('/home/thummala/graphsage-pytorch/graphsage/corakmeans_7.pkl')

    with open("/home/thummala/graphsage-pytorch/cora/Corawalks_10_30_1_1.json", "r") as outfile:
        mydic2 = json.load(outfile)
    num_nodes = 2708
    num_feats = 1433
    degrees = [0]*2708
    for i in range(num_nodes):
        degrees[i]=len(list(adj_lists2[i]))



    return feat_data, labels, adj_lists2,dist_in_graph,freq,feature_labels, distances,degrees
def load_wikics(random_walk=False):
    #function to format neighbours in random walk dic stored as string
    def format_dic(mystring):
        mystring=mystring.replace('{','')
        mystring=mystring.replace('}','')
        myarr=mystring.split(',')
        if myarr==['set()']:
            return set()
        myarr=[int(x) for x in myarr]
        return set(myarr)
    #function to format frequency and distance in graph etc
    def format_dic2(mystring):
        mystring=mystring.replace('{','')
        mystring=mystring.replace('}','')
        myarr=mystring.split(',')
        res={}
        if myarr==['']:
            return res
        for item in myarr:
            els=item.split(':')
            try:
                res[int(els[0].replace("'",''))]=int(els[1])
            except:
                res[int(els[0])] = int(els[1])
        return res
    def generate_bias(centroids):
        distances = {}
        for i, c1 in enumerate(centroids):
            for j, c2 in enumerate(centroids):
                if not (i == j):
                    distances[(i, j)] = np.linalg.norm(c1 - c2)*100
                else:
                    distances[(i, j)] = 1
        return distances
    root_folder='/home/thummala/graph-datasets/Dataset-WikiCS'
    node_df=pd.read_csv(root_folder+'/wikics_nodeinfo.csv')
    with open(root_folder+"/data.json", "r") as file:
        wikics = json.load(file)
    feat_data=wikics['features']
    labels=wikics['labels']
    train_mask,val_mask,test_mask=wikics['train_masks'][10],wikics['val_masks'][10],wikics['test_mask']
    centroids=np.load(root_folder+'/centroids_14.npy')
    distances = generate_bias(centroids)
    feature_labels = node_df['cluster_labels']
    if random_walk:
        adj_list={}
        freq={}
        dist_in_graph={}
        centralityev=[0]*len(train_mask)
        centralityd = [0] * len(train_mask)
        centralitybtw = [0] * len(train_mask)
        centralityh = [0] * len(train_mask)
        for i,row in node_df.iterrows():
            node=int(row['nodes'])
            adj_list[node]=format_dic(row['randomwalkneigh'])
            freq[node]=format_dic2(row['freq_in_randomwalk'])
            dist_in_graph[node]=format_dic2(row['dist_in_graph'])
            centralityev[node]=row['eigenvectorc']
            centralitybtw[node]=row['betweennessc']
            centralityd[node]=row['degreec']
            centralityh[node]=row['harmonicc']
    if not random_walk:
        adj_list={}
        freq={}
        dist_in_graph={}
        centralityev=[0]*len(train_mask)
        centralityh=[0]*len(train_mask)
        centralitybtw=[0]*len(train_mask)
        centralityd=[0]*len(train_mask)
        for i,row in node_df.iterrows():
            node=int(row['nodes'])
            adj_list[node]=set(wikics['links'][node])
            centralityev[node]=row['eigenvectorc']
            centralitybtw[node]=row['betweennessc']
            centralityd[node]=row['degreec']
            centralityh[node]=row['harmonicc']
        for key in adj_list.keys():
            unique = {}
            unique2 = {}
            for item in adj_list[key]:
                unique[int(item)] = 1
                unique2[int(item)] = 1
            freq[int(key)] = unique
            dist_in_graph[int(key)] = unique2
    # sum=0
    # max=0
    # min=100
    # myarr=[]
    # for key,item in adj_list.items():
    #     myarr.append(len(list(item)))
    #     sum=sum+len(list(item))
    #     if len(list(item))>max:
    #         max=len(list(item))
    #     if len(list(item))<min:
    #         min=len(list(item))
    # print(sum/11701)
    # print('max',max)
    # print('min',min)
    # sns.distplot(myarr)
    # plt.show()
    return feat_data,labels,adj_list,train_mask,test_mask,val_mask,distances,feature_labels,freq,dist_in_graph,centralityev,centralitybtw,centralityh,centralityd
feat_data,labels,adj_list,train_mask,test_mask,val_mask,distances,feature_labels,freq,dist_in_graph,centralityev,centralitybtw,centralityh,centralityd=load_wikics(True)
