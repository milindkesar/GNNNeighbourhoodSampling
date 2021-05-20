import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import matplotlib.pyplot as plt

import numpy as np
import time
import random
from sklearn.metrics import f1_score
from collections import defaultdict

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator, Aggregator1
from graphsage.utils import myfunc,load_data
from graphsage.utils2 import custom_load_cora,custom_load_cora2,load_wikics
import pickle
"""
Simple supervised GraphSAGE model as well as examples running the model
on the Cora and Pubmed datasets.
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        #self.xent = nn.BCEWithLogitsLoss()
        self.sig = nn.Sigmoid()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())
def generate_bias(path):
    kmeans = pickle.load(open(path, "rb"))
    centroids=kmeans.cluster_centers_
    distances={}
    #print(set(kmeans.labels_))
    #print(centroids)
    for i,c1 in enumerate(centroids):
        for j,c2 in enumerate(centroids):
            if not (i == j):
                distances[(i,j)]=np.linalg.norm(c1-c2)*100
            else:
                distances[(i,j)]=1
    feature_labels=kmeans.labels_
    return feature_labels,distances

def generate_bias_ppi(path1,path2):
    CModel = pickle.load(open(path1, "rb"))
    centroids = np.load(path2)
    distances={}
    for i,c1 in enumerate(centroids):
        for j,c2 in enumerate(centroids):
            if not (i == j):
                distances[(i,j)]=np.linalg.norm(c1-c2)*1000
                #distances[(i, j)] = 1000
            else:
                distances[(i,j)]=1
    feature_labels=CModel.labels_
    feature_labels[feature_labels==-1] = 38
    return feature_labels,distances

def load_cora():
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open("cora/cora.content") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open("cora/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_cora():
    torch.autograd.set_detect_anomaly(True)
    np.random.seed(147)
    random.seed(147)
    num_nodes = 2708
    #feat_data, labels, adj_lists = load_cora()
    feat_data, labels, adj_lists, dist_in_graph, freq, feature_labels, distance, degrees = custom_load_cora2()
    #print(adj_lists[2627])
    #feat_data, labels, adj_lists = load_cora()
    features = nn.Embedding(2708, 1433)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    #print("labels",labels)
   # features.cuda()
    ## My Changes
    feature_labels,distance=generate_bias('/home/thummala/graphsage-pytorch/graphsage/corakmeans_7.pkl')
    n1=20
    n2=10
    agg1 = MeanAggregator(features, cuda=False,feature_labels=feature_labels, distance=distance)
    #agg1 = Aggregator1(features, cuda=False, feature_labels=feature_labels, distance=distance,freq= freq,spectral=degrees,dist_btwn=dist_in_graph, adj_list=adj_lists)
    enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=False, cuda=False, feature_labels=feature_labels, distance=distance,num_sample=n1)
    #print(enc1(nodes).numpy().shape)
    #agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False,feature_labels=feature_labels, distance=distance)
    #enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
    #        base_model=enc1, gcn=False, cuda=False,feature_labels=feature_labels, distance=distance,num_sample=n2)
    #enc1.num_samples = 0
    #enc2.num_samples = 0

    #graphsage = SupervisedGraphSage(7, enc2)
    graphsage = SupervisedGraphSage(7, enc1)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1500]
    val = rand_indices[1500:2000]
    train = list(rand_indices[2000:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(150):
        batch_nodes = train[:256]
        #print(batch_nodes)
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.data)
    # #
    #print("node is",val[:1])
    #print("adjacency list is",adj_lists[int(val[:1])])
    val_output = graphsage.forward(val)
    test_output = graphsage.forward(test)
    print("Validation F1:",f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:"+ str(np.mean(times)))
    print("test F1: ", f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro"))

def load_pubmed():
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open("pubmed-data/Pubmed-Diabetes.NODE.paper.tab") as fp:
        fp.readline()
        feat_map = {entry.split(":")[1]:i-1 for i,entry in enumerate(fp.readline().split("\t"))}
        for i, line in enumerate(fp):
            info = line.split("\t")
            node_map[info[0]] = i
            labels[i] = int(info[1].split("=")[1])-1
            for word_info in info[2:-1]:
                word_info = word_info.split("=")
                feat_data[i][feat_map[word_info[0]]] = float(word_info[1])
    adj_lists = defaultdict(set)
    with open("pubmed-data/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def run_pubmed():
    np.random.seed(1)
    random.seed(1)
    num_nodes = 19717
    feat_data, labels, adj_lists = load_pubmed()
    features = nn.Embedding(19717, 500)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, 500, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 25

    graphsage = SupervisedGraphSage(3, enc2)
#    graphsage.cuda()
    rand_indices = np.random.permutation(num_nodes)
    test = rand_indices[:1000]
    val = rand_indices[1000:1500]
    train = list(rand_indices[1500:])

    optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
    times = []
    for batch in range(200):
        batch_nodes = train[:1024]
        random.shuffle(train)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, 
                Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time-start_time)
        print(batch, loss.data)

    val_output = graphsage.forward(val)
    test_output=graphsage.forward(test)
    print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:", np.mean(times))
    print("test F1: ",f1_score(labels[test],test_output.data.numpy().argmax(axis=1),average="micro"))
def ppi():
    np.random.seed(1)
    random.seed(1)
    feat_data, labels, adj_lists,train,test,val = myfunc()
    num_nodes = labels.shape[0]
    features = nn.Embedding(num_nodes, 50)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
   # features.cuda()
    ## My Changes
    feature_labels,distance=generate_bias_ppi('/home/thummala/graphsage-pytorch/graphsage/ppi_given.pkl','/home/thummala/graphsage-pytorch/graphsage/ppi-centroids.npy')
    n1=25
    n2=5
    agg1 = MeanAggregator(features, cuda=True,feature_labels=feature_labels, distance=distance)
    enc1 = Encoder(features, 50, 128, adj_lists, agg1, gcn=False, cuda=False, feature_labels=feature_labels, distance=distance,num_sample=n1)
    #print(enc1(nodes).numpy().shape)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False,feature_labels=feature_labels, distance=distance)
    enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
            base_model=enc1, gcn=False, cuda=False,feature_labels=feature_labels, distance=distance,num_sample=n2)
    #enc1.num_samples = 0
    #enc2.num_samples = 0

    graphsage = SupervisedGraphSage(121, enc2)
    #graphsage = SupervisedGraphSage(7, enc1)
#    graphsage.cuda()
#     rand_indices = np.random.permutation(num_nodes)
#     test = rand_indices[:1000]
#     val = rand_indices[1000:1500]
#     train = list(rand_indices[1500:])

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.01)
    times = []
    batch_size=256
    for epoch in range(5):
        random.shuffle(train)
        for batch in range(1,len(train)//batch_size):
            batch_nodes = train[(batch-1)*batch_size:batch*batch_size]
            #random.shuffle(train)
            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes,
                    Variable(torch.from_numpy(labels[np.array(batch_nodes)]).to(torch.float32)))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time-start_time)
            print(batch, loss.data)
    # #
    #print("node is",val[:1])
    #print("adjacency list is",adj_lists[int(val[:1])])
    #test1=graphsage.forward(test)
    #print("test",test1)
    val_output = graphsage.forward(val)
    test_output = graphsage.forward(test)
    print("Validation F1:",F1_score(labels[val], torch.sigmoid(val_output).data.numpy()))
    print("Average batch time:"+ str(np.mean(times)))
    print("test F1: ", F1_score(labels[test], torch.sigmoid(test_output).data.numpy()))
def F1_score(y_true, y_pred):
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    return f1_score(y_true, y_pred, average="micro")

def run_wiki_cs():
    num_nodes = 11701
    feat_data, labels, adj_lists,train_mask,test_mask,val_mask,distance,feature_labels,freq,dist_in_graph,centralityev,centralitybtw,centralityh,centralityd= load_wikics(random_walk=True)
    features = nn.Embedding(11701, 300)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    n1 = 65
    n2 = 20
    epochs=50
    #agg1 = MeanAggregator(features, cuda=False, feature_labels=feature_labels, distance=distance)
    agg1 = Aggregator1(features, cuda=False, feature_labels=feature_labels, distance=distance,freq= freq,spectral=[centralityev,centralitybtw,centralityh,centralityd],dist_btwn=dist_in_graph, adj_list=adj_lists)
    enc1 = Encoder(features, 300, 128, adj_lists, agg1, gcn=False, cuda=False, feature_labels=feature_labels,
                   distance=distance, num_sample=n1)
    #agg2 = Aggregator1(lambda nodes : enc1(nodes).t(), cuda=False, feature_labels=feature_labels, distance=distance, freq=freq, spectral=[centralityev,centralitybtw,centralityh,centralityd], dist_btwn=dist_in_graph, adj_list=adj_lists)
    # print(enc1(nodes).numpy().shape)
    #agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False,feature_labels=feature_labels, distance=distance)
    #enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,base_model=enc1, gcn=False, cuda=False,feature_labels=feature_labels, distance=distance,num_sample=n2)
    graphsage = SupervisedGraphSage(10, enc1)
    #graphsage = SupervisedGraphSage(10, enc1)
    test = np.array([i for i, x in enumerate(test_mask) if x])
    val = np.array([i for i, x in enumerate(val_mask) if x])
    train = np.array([i for i, x in enumerate(train_mask) if x])
    print("Train Size ",len(train))
    print("Test Size ",len(test))
    print("Val Size ",len(val))
    #optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.05)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=0.02)
    times = []
    labels=np.array(labels)
    train_loss=[]
    val_losses=[]
    for epoch in range(epochs):
        batch_nodes = train[:256]
        val_nodes = val[:256]
        random.shuffle(train)
        random.shuffle(val)
        start_time = time.time()
        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes,Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
        loss.backward()
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        train_loss.append(loss.data)
        with torch.no_grad():
                graphsage.eval()
                val_loss = graphsage.loss(val_nodes, Variable(torch.LongTensor(labels[np.array(val_nodes)])))
                val_losses.append(val_loss.data)
        if epoch%5 == 0:
            print(epoch, " training loss: " +str(loss.data) + "  Validation Loss: "+str(val_loss.data))



    # #
    #val_output = graphsage.forward(val)
    test_output = graphsage.forward(test)
    #print("Validation F1:", f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
    print("Average batch time:" + str(np.mean(times)))
    print("test F1: ", f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro"))
    plt.plot(range(epochs),train_loss)
    plt.plot(range(epochs),val_losses)
    plt.legend(['train_loss','val_loss'])
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.show()

if __name__ == "__main__":
    run_wiki_cs()

