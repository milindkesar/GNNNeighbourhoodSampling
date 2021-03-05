import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False,feature_labels=None,distance=None):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

        self.feature_labels = feature_labels
        self.distance=distance
    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        #print("gcn ", self.gcn)
        def custom_sampler():
            bias=[]
            samp_neighs=[]
            neigh_list=[list(to_neigh) for to_neigh in to_neighs]
            for j,node in enumerate(nodes):
                #print(node)
                #print()
                class_node=self.feature_labels[node]
                # for label in neigh_list[j]:
                #     print(label)
                #print(len(self.feature_labels))
                class_neighs=[self.feature_labels[label] for label in neigh_list[j]]
                temp=[]
                for neigh_class in class_neighs:
                    temp.append(self.distance[(class_node,neigh_class)])
                prob = np.array(temp) / np.sum(temp)
                #print(prob)
                if num_sample < len(class_neighs):
                    choice_indices = np.random.choice(len(prob), size=num_sample, replace=False, p=prob)
                else:
                    choice_indices = np.random.choice(len(prob), size=len(prob), replace=False, p=prob)
                #print(choice_indices)
                #print(neigh_list[j])
                temp2=[]
                for index in choice_indices:
                    temp2.append(neigh_list[j][index])
                samp_neighs.append(set(temp2))
            return samp_neighs
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
            #print("sample :" +str(len(samp_neighs)))
            #print("Size is"+str(np.array(samp_neighs).shape))
            #samp_neighs = custom_sampler()
            #print("to_neighs",to_neighs)
        else:
            samp_neighs = to_neighs

        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]
        unique_nodes_list = list(set.union(*samp_neighs))
        #print(unique_nodes_list)
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        #print(unique_nodes)
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        #print("column",np.array(column_indices).shape)
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        #mask[row_indices, column_indices] = 1
        for i,samp_neigh in enumerate(samp_neighs):
            for node in samp_neigh:
                mask[i,unique_nodes[node]]=1

        if self.cuda:
            mask = mask.cuda()
        num_neigh = mask.sum(1, keepdim=True)
        #print("num neigh",num_neigh.shape)
        #print("mask before 1.5", torch.isnan(mask).any())
        mask = mask.div(num_neigh)
        #print(num_neigh)
        # if torch.isnan(mask).any():
        #     mask[mask != mask] = 0
        #print("mask before 2", torch.isnan(mask).any())
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))
            #embed_matrix = self.features(torch.LongTensor(list(unique_nodes.values())))
        #print("embed matrix",embed_matrix)
        #print("nan check", torch.isnan(embed_matrix).any())
        to_feats = mask.mm(embed_matrix)
        return to_feats
