import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
import matplotlib.pyplot as plt

import numpy as np
import time
import random
from sklearn.metrics import f1_score,accuracy_score
from collections import defaultdict
import os
from graphsage.encoders import Encoder, LSHNeighboursEncoder
from graphsage.aggregators import MeanAggregator, Aggregator1, PureMeanAggregator
from graphsage.utils import myfunc,load_data
from graphsage.utils2 import load_cora,load_wikics,construct_agg, load_ppi, F1_score, custom_load_pubmed, get_average_lsh_added
import pickle
import json

from torch.utils.tensorboard import SummaryWriter

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()
        self.xent2 = nn.BCEWithLogitsLoss()
        self.sig = nn.Sigmoid()
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels, classification='normal'):
        if classification == 'multi_label':
            scores = self.forward(nodes)
            return self.xent2(scores.float(),labels.float())
        else:
            scores = self.forward(nodes)
            return self.xent(scores, labels.squeeze())

    def returnembedding(self,nodes):
        embeds = self.enc(nodes)
        return embeds
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
#
# def run_cora():
#     torch.autograd.set_detect_anomaly(True)
#     np.random.seed(147)
#     random.seed(147)
#     num_nodes = 2708
#     #feat_data, labels, adj_lists = load_cora()
#     feat_data, labels, adj_lists, dist_in_graph, freq, feature_labels, distance, degrees = custom_load_cora2()
#     #print(adj_lists[2627])
#     #feat_data, labels, adj_lists = load_cora()
#     features = nn.Embedding(2708, 1433)
#     features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
#     #print("labels",labels)
#    # features.cuda()
#     ## My Changes
#     feature_labels,distance=generate_bias('/home/thummala/graphsage-pytorch/graphsage/corakmeans_7.pkl')
#     n1=20
#     n2=10
#     agg1 = MeanAggregator(features, cuda=False,feature_labels=feature_labels, distance=distance)
#     #agg1 = Aggregator1(features, cuda=False, feature_labels=feature_labels, distance=distance,freq= freq,spectral=degrees,dist_btwn=dist_in_graph, adj_list=adj_lists)
#     enc1 = Encoder(features, 1433, 128, adj_lists, agg1, gcn=False, cuda=False, feature_labels=feature_labels, distance=distance,num_sample=n1)
#     #print(enc1(nodes).numpy().shape)
#     #agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False,feature_labels=feature_labels, distance=distance)
#     #enc2 = Encoder(lambda nodes : enc1(nodes).t(), enc1.embed_dim, 128, adj_lists, agg2,
#     #        base_model=enc1, gcn=False, cuda=False,feature_labels=feature_labels, distance=distance,num_sample=n2)
#     #enc1.num_samples = 0
#     #enc2.num_samples = 0
#
#     #graphsage = SupervisedGraphSage(7, enc2)
#     graphsage = SupervisedGraphSage(7, enc1)
# #    graphsage.cuda()
#     rand_indices = np.random.permutation(num_nodes)
#     test = rand_indices[:1500]
#     val = rand_indices[1500:2000]
#     train = list(rand_indices[2000:])
#
#     optimizer = torch.optim.SGD(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.7)
#     times = []
#     for batch in range(150):
#         batch_nodes = train[:256]
#         #print(batch_nodes)
#         random.shuffle(train)
#         start_time = time.time()
#         optimizer.zero_grad()
#         loss = graphsage.loss(batch_nodes,
#                 Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
#         loss.backward()
#         optimizer.step()
#         end_time = time.time()
#         times.append(end_time-start_time)
#         print(batch, loss.data)
#     # #
#     #print("node is",val[:1])
#     #print("adjacency list is",adj_lists[int(val[:1])])
#     val_output = graphsage.forward(val)
#     test_output = graphsage.forward(test)
#     print("Validation F1:",f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro"))
#     print("Average batch time:"+ str(np.mean(times)))
#     print("test F1: ", f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro"))

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


### Helper function to evaluate and write test results
def evaluate(model,test,out_dir,labels,epoch,classification):
    results={}
    test_output = model.forward(test)
    if classification == 'normal':
        test_f1 = f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
        test_f1_macro = f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="macro")
    elif classification == 'multi_label':
        test_f1 = F1_score(labels[test], torch.sigmoid(test_output).data.numpy())
        test_f1_macro = 0.70
    results['epoch']=epoch
    results['test_f1_micro']=test_f1
    results['test_f1_macro']=test_f1_macro
    with open(out_dir+'/test.txt','a+') as out:
        out.write(json.dumps(results)+'\n')

### Helper function to run the model for a given dataset
def run_general(name,outdir,rw=False,neighbours1=20,neighbours2=20,epochs=100,attention='normal',aggregator='mean',n_layers=1,random_iter=1,lr=0.01,includenodefeats="no", type_of_walk = 'default', p=1,q=1,num_walks=10,walk_length=10,teleport=0.2, teleport_khop=False, dfactor=2, save_predictions = False, n_vectors = 16, search_radius = 2, atleast = False, num_lsh_neigbours=10, n_lsh_neighbours_sample = None, augment_khop=False, includeNeighbourhood = False, gcn = False, load_embeds=False):
    for k in range(random_iter):
        if gcn:
            print("Using gcn formulation")
        classification='normal'
        print("random walk",rw)
        lsh_helper = {'n_vectors': n_vectors, 'search_radius': search_radius, 'num_lsh_neighbours': num_lsh_neigbours,'atleast': atleast, 'includeNeighbourhood':includeNeighbourhood}
        if augment_khop or teleport_khop:
            print('lsh to be constructed with ', lsh_helper)
            print('lsh neighbours to sample per node',n_lsh_neighbours_sample)
            print('include neighbourhood (in LSH) ',includeNeighbourhood)
        ## Loading dataset

        if name == 'wikics':
            data_dic = load_wikics(lsh_helper, random_walk=rw,type_walk=type_of_walk, p=p ,q=q,num_walks=num_walks,walk_length=walk_length,teleport=teleport, teleport_khop=teleport_khop, dfactor=dfactor, augment_khop=augment_khop, load_embeds = load_embeds)
            #feat_data, labels, adj_lists, train_mask, test_mask, val_mask, distance, feature_labels, freq, dist_in_graph, centralityev, centralitybtw, centralityh, centralityd = load_wikics(random_walk=rw,type_walk=type_of_walk, p=p ,q=q,num_walks=num_walks,walk_length=walk_length,teleport=teleport, teleport_khop=teleport_khop, dfactor=dfactor)
            data_dic['feat_data']=np.array(data_dic['feat_data'])
            data_dic['labels'] = np.array(data_dic['labels'])
        if name == 'ppi':
            data_dic = load_ppi(lsh_helper, random_walk=rw,teleport=teleport, teleport_khop=teleport_khop, dfactor=dfactor,augment_khop = augment_khop, load_embeds = load_embeds)
            #feat_data,labels,adj_lists,train_mask,test_mask,val_mask,distance,feature_labels,freq,dist_in_graph,centralityev,centralitybtw,centralityh,centralityd = load_ppi(random_walk=rw)
            classification='multi_label'
        if name == 'cora':
            data_dic = load_cora(lsh_helper, random_walk=rw, teleport=teleport, teleport_khop=teleport_khop, dfactor=dfactor, augment_khop=augment_khop, load_embeds = load_embeds)
        if name == 'pubmed':
            data_dic = custom_load_pubmed(lsh_helper, random_walk=rw, teleport=teleport, teleport_khop=teleport_khop, dfactor=dfactor, augment_khop=augment_khop)

        if augment_khop:
            average_lsh_add, average_lsh_add_low = get_average_lsh_added(data_dic['adj_lists'],data_dic['lsh_neighbour_list'])
            print('average lsh added for all nodes, nodes with degree less than 5',average_lsh_add, average_lsh_add_low)
        ## Defining some useful terms (such as features, number of classes, etc.)
        num_nodes = data_dic['feat_data'].shape[0]
        features = nn.Embedding(data_dic['feat_data'].shape[0],data_dic['feat_data'].shape[1])
        features.weight = nn.Parameter(torch.FloatTensor(data_dic['feat_data']), requires_grad=False)
        n1 = neighbours1
        n2 = neighbours2
        n_lsh_neighbours = n_lsh_neighbours_sample
        num_feats=data_dic['feat_data'].shape[1]
        n_layers = n_layers
        rand_split = 1
        if name=='ppi':
            n_classes=121
        else:
            unique_labels=np.unique(data_dic['labels'])
            n_classes = len(unique_labels)
        epochs=epochs

        print("aggregator is :",aggregator)
        print("using softmax attention :",attention)

    ## Training Loop


        ## Using Mean Aggregator
        if aggregator == 'mean':
            agg1 = MeanAggregator(features, cuda=False, feature_labels=data_dic['cluster_labels'], distance=data_dic['distances'], gcn=gcn)
            enc1 = Encoder(features, num_feats, 128, data_dic['adj_lists'], agg1, gcn=gcn, cuda=False,
                           feature_labels=data_dic['cluster_labels'], distance=data_dic['distances'], num_sample=n1,
                           lsh_neighbours=data_dic['lsh_neighbour_list'], n_lsh_neighbours=n_lsh_neighbours,
                           lsh_augment=augment_khop)
            if n_layers > 1:
                agg2 = MeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False,
                                      feature_labels=data_dic['cluster_labels'], distance=data_dic['distances'], gcn=gcn)
                enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, data_dic['adj_lists'], agg2,
                               base_model=enc1, gcn=gcn, cuda=False, feature_labels=data_dic['cluster_labels'],
                               distance=data_dic['distances'], num_sample=n2,
                               lsh_neighbours=data_dic['lsh_neighbour_list'], n_lsh_neighbours=n_lsh_neighbours,
                               lsh_augment=augment_khop)
                graphsage = SupervisedGraphSage(n_classes, enc2)
            else:
                graphsage = SupervisedGraphSage(n_classes, enc1)

        ## New LSH Formulation
        elif aggregator == 'lsh_mean':
            agg1 = PureMeanAggregator(features, cuda=False, gcn = gcn)
            lsh_agg1 = PureMeanAggregator(features, cuda=False, gcn=gcn)
            enc1= LSHNeighboursEncoder(features=features, feature_dim=num_feats, embed_dim=128, adj_lists=data_dic['adj_lists'], aggregator=agg1,lsh_aggregator=lsh_agg1, num_sample=n1, gcn=gcn, cuda=False, lsh_neighbours = data_dic['lsh_neighbour_list'], n_lsh_neighbours = n_lsh_neighbours, lsh_augment=augment_khop)
            if n_layers > 1:
                agg2 = PureMeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False, gcn=gcn)
                lsh_agg2 = PureMeanAggregator(lambda nodes: enc1(nodes).t(), cuda=False, gcn=gcn)
                enc2 = LSHNeighboursEncoder(lambda nodes: enc1(nodes).t(), feature_dim=enc1.embed_dim, embed_dim=128,
                                            adj_lists=data_dic['adj_lists'], aggregator=agg2, lsh_aggregator=lsh_agg2,
                                            num_sample=n2, gcn=gcn, cuda=False,
                                            lsh_neighbours=data_dic['lsh_neighbour_list'],
                                            n_lsh_neighbours=n_lsh_neighbours, lsh_augment=augment_khop)
                graphsage = SupervisedGraphSage(n_classes, enc2)
            else:
                graphsage = SupervisedGraphSage(n_classes, enc1)

        ## Using Weighted Mean Aggregator (Changes to attention should be done HERE)
        elif aggregator == 'weighted_mean':
            print('using weighted mean aggregator...')
            agg1 = Aggregator1(features, cuda=False, feature_labels=data_dic['cluster_labels'],
                               distance=data_dic['distances'], freq=data_dic['freq'],
                               spectral=[data_dic['centralityev'], data_dic['centralitybtw'], data_dic['centralityh'],
                                         data_dic['centralityd']], dist_btwn=data_dic['dist_in_graph'],
                               adj_list=data_dic['adj_lists'], attention=attention, addnodefeats=includenodefeats,
                               layerno=1, gcn=gcn)
            enc1 = Encoder(features, num_feats, 128, data_dic['adj_lists'], agg1, gcn=gcn, cuda=False,
                           feature_labels=data_dic['cluster_labels'], distance=data_dic['distances'], num_sample=n1)
            if n_layers > 1:
                agg2 = Aggregator1(lambda nodes: enc1(nodes).t(), cuda=False, feature_labels=data_dic['cluster_labels'],
                                   distance=data_dic['distances'], freq=data_dic['freq'],
                                   spectral=[data_dic['centralityev'], data_dic['centralitybtw'],
                                             data_dic['centralityh'], data_dic['centralityd']],
                                   dist_btwn=data_dic['dist_in_graph'], adj_list=data_dic['adj_lists'],
                                   attention=attention, addnodefeats=includenodefeats, layerno=2, gcn = gcn)
                enc2 = Encoder(lambda nodes: enc1(nodes).t(), enc1.embed_dim, 128, data_dic['adj_lists'], agg2,
                               base_model=enc1, gcn=gcn, cuda=False, feature_labels=data_dic['cluster_labels'],
                               distance=data_dic['distances'], num_sample=n2)
                graphsage = SupervisedGraphSage(n_classes, enc2)
            else:
                graphsage = SupervisedGraphSage(n_classes, enc1)
        else:
            print('aggregator not supported')

        if outdir != None:
            out_dir = outdir
        else:
            out_dir = 'res/' + name
        out_dir=out_dir+'/'+str(k)
        os.makedirs(out_dir, exist_ok=True)
        ## To see if predefined splits exist (Otherwise use random splits (As of Now: 0.7 train, 0.2 test, 0.1 val))
        if data_dic['train_mask'] != []:
            rand_split=0
            test = np.array([i for i, x in enumerate(data_dic['test_mask']) if x])
            val = np.array([i for i, x in enumerate(data_dic['val_mask']) if x])
            train = np.array([i for i, x in enumerate(data_dic['train_mask']) if x])
            print('using given splits...')
            print('test',test.shape)
            print('val',val.shape)
            print('train',train.shape)
        elif name == 'cora':
            if save_predictions:
                np.random.seed(1)
            rand_indices = np.random.permutation(num_nodes)
            test = rand_indices[:1500]
            val = rand_indices[1500:2000]
            train = list(rand_indices[2000:])
        # elif name == 'pubmed':
        #     rand_indices = np.random.permutation(num_nodes)
        #     test = rand_indices[:1000]
        #     val = rand_indices[1000:1500]
        #     train = list(rand_indices[1500:])
        else:
            print('using random splits...')
            rand_indices = np.random.permutation(num_nodes)
            test = rand_indices[:int(0.2*num_nodes)]
            val = rand_indices[int(0.2*num_nodes):int(0.3*num_nodes)]
            train = list(rand_indices[int(0.3*num_nodes):])

        ## Defining optimizer and batch_size
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=lr)
        times = []
        labels=np.array(data_dic['labels'])
        batch_size=256

        for epoch in range(epochs):
            val_nodes = val[:256]
            random.shuffle(train)
            random.shuffle(val)
            start_time = time.time()
            training_loss_epoch = []
            for batch in range(0,len(train),batch_size):
                batch_nodes = train[batch:batch+batch_size]
                optimizer.zero_grad()
                loss = graphsage.loss(batch_nodes,Variable(torch.LongTensor(labels[np.array(batch_nodes)])),classification)
                loss.backward()
                optimizer.step()
                training_loss_epoch.append(loss.data.numpy())
                #train_loss.append(loss.data)
                        #val_losses.append(val_loss.data)
            end_time = time.time()
            times.append(end_time - start_time)
            # if epoch == 0:
            #     print("layer 1 lsh neighbours added on average ", agg1.average_lsh_added / agg1.count, agg1.count)
            #     print("layer 2 lsh neighbours added on average", agg2.average_lsh_added/agg2.count, agg2.count)
            with torch.no_grad():
                graphsage.eval()
                val_loss = graphsage.loss(val_nodes, Variable(torch.LongTensor(labels[np.array(val_nodes)])),classification)
            ## Every 5 epochs write training loss (mean of all batches) and validation loss
            if epoch%5 == 0:
                temp={}
                temp['epoch']=epoch
                temp['training loss']=str(sum(training_loss_epoch)/len(training_loss_epoch))
                temp['validation loss']=str(val_loss.data)
                with open(out_dir+'/training_info.txt', 'a+') as convert_file:
                    convert_file.write(json.dumps(temp)+'\n')
                print(epoch, " training loss: " +str(sum(training_loss_epoch)/len(training_loss_epoch)) + "  Validation Loss: "+str(val_loss.data))
            ## Every 50 epochs or the final epoch, write the test loss to file.
            if epoch%50 == 0 or epoch == epochs-1:
                evaluate(graphsage,test, out_dir,labels,epoch,classification)

    ## Helper code to write save the predictions and embeddings for qualitative analysis
    if save_predictions:
        test_output = graphsage.forward(test)
        predicted_scores=test_output.data.numpy().argmax(axis=1)
        np.save(outdir+'/predictions.npy', predicted_scores)
        embeds = graphsage.returnembedding(test).detach().t().numpy()
        np.save(outdir+'/embeddings.npy', embeds)
        ## save embeddings for all nodes using GS
        nodes = np.arange(num_nodes)
        batch_size = 512
        ind = 0
        for batch in range(0, len(nodes), batch_size):
            batch_nodes = nodes[batch:batch + batch_size]
            batch_embeds = graphsage.returnembedding(np.array(batch_nodes)).detach().t().numpy()
            # batch_embeds = np.random.rand(len(batch_nodes),128)
            if ind == 0:
                allnodesembeds = batch_embeds
            else:
                allnodesembeds = np.append(allnodesembeds, batch_embeds, axis=0)
            ind += 1
        # allnodesembeds = graphsage.returnembedding(np.arange(num_nodes)).detach().t().numpy()
        np.save(outdir+'/allnodeembeddings.npy',allnodesembeds)
        if rand_split:
            np.save(outdir+'/test.npy',np.array(test))
            np.save(outdir+'/train.npy',np.array(train))
            np.save(outdir+'/val.npy',np.array(val))

    ## Helper code to write the best results and mean results of multiple iterations
    if random_iter == 1:
        pass
    else:
        construct_agg(outdir)

    if augment_khop:
        file = open(outdir+"/otherstats.txt", "w")
        file.write("Average LSH added for all nodes = " + str(average_lsh_add) + "\n" + "Average LSH added for nodes with degree less than 5 = " + str(average_lsh_add_low))
        file.close()
        # val_output = graphsage.forward(val)
        # test_output = graphsage.forward(test)
        # valid_f1 = f1_score(labels[val], val_output.data.numpy().argmax(axis=1), average="micro")
        # test_f1 = f1_score(labels[test], test_output.data.numpy().argmax(axis=1), average="micro")
        # results={}
        # results['valid_f1']=valid_f1
        # results['test_f1']=test_f1
        # with open(out_dir+'/results.txt','a') as out:
        #     out.write(json.dumps(results))
        # print("Validation F1:", valid_f1)
        # print("Average batch time:" + str(np.mean(times)))
        # print("test F1: ", test_f1)

