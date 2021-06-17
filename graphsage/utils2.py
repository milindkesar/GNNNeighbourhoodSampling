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

def F1_score(y_true, y_pred):
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    # sum=0
    # for i in range(121):
    #     sum+= f1_score(y_true[:,i],y_pred[:,i],average="micro")
    return f1_score(y_true, y_pred, average="micro")

def count_train_test_val(myarr):
    count=0
    for item in myarr:
        if item == True:
            count+=1
    return count
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
def construct_agg(dir=None):
    try:
        os.makedirs(dir+'/agg')
    except:
        pass
    sub_dirs=[x[0] for x in os.walk(dir)]
    agg_training_info_l=[]
    agg_test_l=[]
    bigdata=[]
    best_test_f1={'epoch':-1,'test_f1':-1}
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
        agg_test = {'epoch':bigdata[0][i]['epoch'],'test_f1': []}
        for j in range(len(bigdata)):
            agg_test['test_f1'].append(bigdata[j][i]['test_f1'])
        agg_test_l.append(agg_test)
    for item in agg_test_l:
        avg_test_f1=sum(item['test_f1'])/len(item['test_f1'])
        item['test_f1'] = avg_test_f1
        if avg_test_f1 > best_test_f1['test_f1']:
            best_test_f1['test_f1']=avg_test_f1
            best_test_f1['epoch']=item['epoch']
    with open(dir+'/agg'+'/test.txt','a+') as out:
        for item in agg_test_l:
            out.write(json.dumps(item)+'\n')
    with open(dir+'/agg'+'/best.txt','a+') as out:
        out.write(json.dumps(best_test_f1))
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
    print("Loading WikiCS")
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
        print('loading random walk samples')
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
    return feat_data,labels,adj_list,train_mask,test_mask,val_mask,distances,feature_labels,freq,dist_in_graph,centralityev,centralitybtw,centralityh,centralityd
def load_ppi(random_walk=False,root_folder='/home/thummala/graph-datasets/Dataset-PPI/ppi'):
    #function to format neighbours in random walk dic stored as string
    print("Loading PPI Dataset","random_walk",random_walk)
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
    def remove_self(adj_list):
      for key in list(adj_list.keys()):
        adj_list[key] = set([x for x in list(adj_list[key]) if x != key])
      return adj_list
    root_folder=root_folder
    node_df=pd.read_csv(root_folder+'/node_info.csv')
    with open(root_folder+'/ppi-G.json',) as F:
      data=json.load(F)
    #data.keys()
    with open(root_folder+'/ppi-id_map.json',) as F:
      id_maps=json.load(F)
    with open(root_folder+'/ppi-class_map.json',) as F:
      label_map=json.load(F)
    temp=[0]*len(label_map)
    for key in label_map.keys():
      temp[int(key)] = label_map[key]
    labels=np.array(temp)
    feat_data = np.load(root_folder+'/ppi-feats.npy')
    nodedf=pd.read_csv(root_folder+'/node_info.csv')
    centroids=np.load(root_folder+'/centroids_kmeans_39.npy')
    distances = generate_bias(centroids)
    feature_labels = node_df['cluster_labels']
    type_ = node_df['type']
    num_nodes=56944
    adj_list={}
    freq={}
    dist_in_graph={}
    centralityev=[0]*num_nodes
    centralityd=[0]*num_nodes
    centralitybtw=[0]*num_nodes
    centralityh=[0]*num_nodes
    if random_walk:
        print("Random Walk Samples")
        for i,row in node_df.iterrows():
            node=int(row['nodes'])
            adj_list[node]=format_dic(row['randomwalkneigh'])
            freq[node]=format_dic2(row['freq_in_randomwalk'])
            dist_in_graph[node]=format_dic2(row['dist_in_graph'])
            centralityd[node]=row['degreec']
            centralityh[node]=row['harmonicc']
        adj_list=remove_self(adj_list)
    if not random_walk:
        print("K-Hop Neighbours")
        for i,row in node_df.iterrows():
            node=int(row['nodes'])
            centralityd[node]=row['degreec']
            centralityh[node]=row['harmonicc']
            adj_list[node]=set()
            freq[node]=format_dic2(row['freq_in_randomwalk'])
            dist_in_graph[node]=format_dic2(row['dist_in_graph'])
        for edge in data['links']:
            adj_list[int(edge['source'])].add(int(edge['target']))
            adj_list[int(edge['target'])].add(int(edge['source']))
        for key in adj_list.keys():
            for item in adj_list[key]:
                if item not in freq[key].keys():
                  freq[key][int(item)]=0
                if item not in dist_in_graph[key].keys():
                  dist_in_graph[key][int(item)]=1
    train_mask,val_mask,test_mask = [],[],[]
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
    return feat_data,labels,adj_list,train_mask,test_mask,val_mask,distances,feature_labels,freq,dist_in_graph,centralityev,centralitybtw,centralityh,centralityd
