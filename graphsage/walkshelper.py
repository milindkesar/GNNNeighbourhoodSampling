### One function
#function to give frequency of elements in a list
import pandas as pd
import numpy as np
import json
import collections
import networkx as nx
from graphsage.random_walks import FeatureTeleport_RandomWalk, ClusterTeleport_RandomWalk
from sklearn.cluster import KMeans
def give_freq(t2):
  frequency = {}
  for item in t2:
    if item in frequency:
        frequency[item] += 1
    else:
        frequency[item] = 1
  return frequency
#function to generate adjacency matrix and frequency of neighbours in random walks
def get_adj_list(mydic):
  adj_lists={}
  freq={}
  for key,item in mydic.items():
    unique2=[]
    freq2={}
    for temp in item:
      for neigh in temp:
        unique2.append(int(neigh))
    unique=set(unique2)
    freq2=give_freq(unique2)
    adj_lists[int(key)]=unique
    freq[int(key)]=freq2
  return freq,adj_lists
def getsp(mygraph,source,target):
  path=nx.shortest_path(mygraph,source,target)
  return len(path)-1
def format_walks2(walklist,mygraph):
  walks = [[str(n) for n in walk] for walk in walklist]
  mydic={}
  for node in mygraph.nodes():
    mydic[node]=[]
  for item in walklist:
    node=item[0]
    mydic[node].append(item[1:])
  return mydic

def generate_clusterteleportwalks(data2,p=1,q=1,teleport_weight=0.2, walklength=10, num_walks=10, workers=1,save=True,source_path='/home/thummala/graphsage-pytorch/datasets/Dataset-WikiCS/wikics_nodeinfo.csv',root_folder='/home/thummala/graphsage-pytorch/datasets/Dataset-WikiCS', n_clusters = 14):
  try:
    nodedf = pd.read_csv(root_folder+'/wikics_ctrw_'+str(p)+'_'+str(q)+'_'+str(num_walks)+'_'+str(walklength)+'_'+str(teleport_weight)+'.csv')
    print('file exists reading')
  except:
    print('performing random walks')
    mygraph = nx.Graph()
    for i in range(len(data2["features"])):
      mygraph.add_node(i,features=data2["features"][i])
    for i in range(len(data2["links"])):
      edge_list=data2["links"][i]
      for j in edge_list:
        mygraph.add_edge(i,j)
    test=pd.read_csv(source_path)
    degreecentrality=test['degreec']
    eigenvectorcentrality=test['eigenvectorc']
    harmoniccentrality=test['harmonicc']
    betweennesscentrality=test['betweennessc']

    ## Modify Clustering HERE
    kmeans = KMeans(n_clusters=n_clusters).fit(data2['features'])

    random_walk = ClusterTeleport_RandomWalk(mygraph, walk_length=walklength, num_walks=num_walks, p=p, q=q, workers=workers, clusterteleport_weight = teleport_weight, cluster_labels=kmeans.labels_)
    print('random walks done...formatting')
    walklists = random_walk.walks
    walks_dic=format_walks2(walklists,mygraph)
    wikics_freq,wikics_adj_lists = get_adj_list(walks_dic)
    dist_in_graph={}
    for root in list(wikics_adj_lists.keys()):
      neighs=wikics_adj_lists[root]
      temp={}
      for neigh in neighs:
        try:
          temp[str(neigh)]=getsp(mygraph,int(root),int(neigh))
        except:
          temp[str(neigh)]=10
      dist_in_graph[root] = temp
    nodedf = pd.DataFrame(columns=('nodes','randomwalkneigh','freq_in_randomwalk','dist_in_graph','eigenvectorc','degreec','betweennessc','harmonicc','cluster_labels'))
    for i,node in enumerate(mygraph.nodes()):
      if node != -1:
        nodedf.loc[i]=[node,wikics_adj_lists[node],wikics_freq[node],dist_in_graph[node],eigenvectorcentrality[node],degreecentrality[node],betweennesscentrality[node],harmoniccentrality[node], kmeans.labels_[node]]
    nodedf['cluster_labels']=test['cluster_labels']
    if save:
      ##naming convention wikics_ftrw_p_q_numwalks_walklength_featureteleport.csv
      print('saving')
      nodedf.to_csv(root_folder+'/wikics_ftrw_'+str(p)+'_'+str(q)+'_'+str(num_walks)+'_'+str(walklength)+'_'+str(teleport_weight)+'.csv')
    nodedf = pd.read_csv(root_folder+'/wikics_ftrw_'+str(p)+'_'+str(q)+'_'+str(num_walks)+'_'+str(walklength)+'_'+str(teleport_weight)+'.csv')
    print('done')
  return nodedf

def generate_featureteleportwalks(data2,p=1,q=1,teleport_weight=0.2, walklength=10, num_walks=10, workers=1,save=True,source_path='/home/thummala/graphsage-pytorch/datasets/Dataset-WikiCS/wikics_nodeinfo.csv',root_folder='/home/thummala/graphsage-pytorch/datasets/Dataset-WikiCS'):
  try:
    nodedf = pd.read_csv(root_folder+'/wikics_ftrw_'+str(p)+'_'+str(q)+'_'+str(num_walks)+'_'+str(walklength)+'_'+str(teleport_weight)+'.csv')
    print('file exists reading')
  except:
    print('performing random walks')
    mygraph = nx.Graph()
    for i in range(len(data2["features"])):
      mygraph.add_node(i,features=data2["features"][i])
    for i in range(len(data2["links"])):
      edge_list=data2["links"][i]
      for j in edge_list:
        mygraph.add_edge(i,j)
    test=pd.read_csv(source_path)
    degreecentrality=test['degreec']
    eigenvectorcentrality=test['eigenvectorc']
    harmoniccentrality=test['harmonicc']
    betweennesscentrality=test['betweennessc']
    random_walk = FeatureTeleport_RandomWalk(mygraph, walk_length=walklength, num_walks=num_walks, p=p, q=q, workers=workers,node_features=data2['features'], teleport_weight = teleport_weight)
    print('random walks done...formatting')
    walklists = random_walk.walks
    walks_dic=format_walks2(walklists,mygraph)
    wikics_freq,wikics_adj_lists = get_adj_list(walks_dic)
    dist_in_graph={}
    for root in list(wikics_adj_lists.keys()):
      neighs=wikics_adj_lists[root]
      temp={}
      for neigh in neighs:
        try:
          temp[str(neigh)]=getsp(mygraph,int(root),int(neigh))
        except:
          temp[str(neigh)]=10
      dist_in_graph[root] = temp
    nodedf = pd.DataFrame(columns=('nodes','randomwalkneigh','freq_in_randomwalk','dist_in_graph','eigenvectorc','degreec','betweennessc','harmonicc'))
    for i,node in enumerate(mygraph.nodes()):
      if node != -1:
        nodedf.loc[i]=[node,wikics_adj_lists[node],wikics_freq[node],dist_in_graph[node],eigenvectorcentrality[node],degreecentrality[node],betweennesscentrality[node],harmoniccentrality[node]]
    nodedf['cluster_labels']=test['cluster_labels']
    if save:
      ##naming convention wikics_ftrw_p_q_numwalks_walklength_featureteleport.csv
      print('saving')
      nodedf.to_csv(root_folder+'/wikics_ftrw_'+str(p)+'_'+str(q)+'_'+str(num_walks)+'_'+str(walklength)+'_'+str(teleport_weight)+'.csv')
    nodedf = pd.read_csv(root_folder+'/wikics_ftrw_'+str(p)+'_'+str(q)+'_'+str(num_walks)+'_'+str(walklength)+'_'+str(teleport_weight)+'.csv')
    print('done')
  return nodedf