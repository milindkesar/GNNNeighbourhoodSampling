import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

class Encoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach
    """
    def __init__(self, features, feature_dim, 
            embed_dim, adj_lists, aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False,feature_labels=None,distance=None,
            feature_transform=False, lsh_neighbours = None, n_lsh_neighbours = None, lsh_augment=False):
        super(Encoder, self).__init__()
        self.feature_labels=feature_labels
        self.distance=distance
        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.lsh_neighbours = lsh_neighbours
        self.n_lsh_neighbours = n_lsh_neighbours
        self.lsh_augment=lsh_augment
        if base_model != None:
            self.base_model = base_model

        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        self.weight = nn.Parameter(
                torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        #print("In encoders" +str(self.adj_lists[4]))
        if self.lsh_augment:
            neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],
                    self.num_sample, lsh_neighbours = [self.lsh_neighbours[int(node)] for node in nodes], n_lsh_neighbours= self.n_lsh_neighbours, lsh_augment = self.lsh_augment)
        else:
            neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes],self.num_sample)
        #print("neigh_feats",torch.isnan(neigh_feats).any())

        if not self.gcn:
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            combined = torch.cat([self_feats, neigh_feats], dim=1)
        else:
            combined = neigh_feats
        #print("shape combinded before:", combined.t().shape)
        combined = F.relu(self.weight.mm(combined.t()))

        return combined


## Custom encoder class for incorporating LSH neighbours as well

class LSHNeighboursEncoder(nn.Module):
    """
    Encodes a node's using 'convolutional' GraphSage approach and also allows to incorporate LSH neighbours
    """
    def __init__(self, features, feature_dim,
            embed_dim, adj_lists, aggregator,lsh_aggregator,
            num_sample=10,
            base_model=None, gcn=False, cuda=False, lsh_neighbours = None, n_lsh_neighbours = 5, lsh_augment=False):
        super(LSHNeighboursEncoder, self).__init__()
        self.features = features
        self.feat_dim = feature_dim
        self.adj_lists = adj_lists
        self.aggregator = aggregator
        self.num_sample = num_sample
        self.lsh_neighbours = lsh_neighbours
        self.n_lsh_neighbours = n_lsh_neighbours
        self.lsh_augment=lsh_augment
        self.lsh_aggregator = lsh_aggregator
        if base_model != None:
            self.base_model = base_model
        self.gcn = gcn
        self.embed_dim = embed_dim
        self.cuda = cuda
        self.aggregator.cuda = cuda
        if self.lsh_augment:
            self.weight = nn.Parameter(torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 3 * self.feat_dim))
            if self.gcn:
                self.weight_lsh_gcn = nn.Parameter(torch.FloatTensor(embed_dim, self.feat_dim))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(embed_dim, self.feat_dim if self.gcn else 2 * self.feat_dim))
        init.xavier_uniform(self.weight)

    def forward(self, nodes):
        """
        Generates embeddings for a batch of nodes.

        nodes     -- list of nodes
        """
        #print("In encoders" +str(self.adj_lists[4]))
        neigh_feats = self.aggregator.forward(nodes, [self.adj_lists[int(node)] for node in nodes], self.num_sample)
        if self.lsh_augment:
            # print('aggregating LSH Neighbour features')
            # print('lsh neighbours', self.lsh_neighbours[1])
            # print('graphsage neighbours', self.adj_lists[1])
            lsh_neigh_feats = self.lsh_aggregator.forward(nodes, [set(self.lsh_neighbours[int(node)]) for node in nodes], self.n_lsh_neighbours)
        #print("neigh_feats",torch.isnan(neigh_feats).any())
        if not self.gcn:
            ## GraphSAGE formulation (take concat of features not sum) (if LSH augment concat all three to differentialte LSH neighbours)
            if self.cuda:
                self_feats = self.features(torch.LongTensor(nodes).cuda())
            else:
                self_feats = self.features(torch.LongTensor(nodes))
            if self.lsh_augment:
                # print("Using GraphSAGE formulation with triple concat")
                # print('lsh neighbour feature', lsh_neigh_feats)
                # print('graphsage neighbour feature', neigh_feats)
                combined = torch.cat([self_feats, neigh_feats, lsh_neigh_feats], dim=1)
            else:
                combined = torch.cat([self_feats, neigh_feats], dim=1)
            combined = F.relu(self.weight.mm(combined.t()))
        else:
            ## GCN formulation (if LSH augment take two matrices as discussed)
            if self.lsh_augment:
                combined = F.relu(self.weight.mm(neigh_feats.t())+self.weight_lsh_gcn.mm(lsh_neigh_feats.t()))
            else:
                combined = F.relu(self.weight.mm(neigh_feats.t()))
        return combined
