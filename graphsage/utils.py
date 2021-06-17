import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np
import os
from _collections import defaultdict
def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    #print(G_data.keys())
    # print(G_data['nodes'][0])
    # print(G_data['links'][0])
    # for key in list(G_data.keys()):
    #     if key != 'links':
    #         print(key,G_data[key])
    G = json_graph.node_link_graph(G_data)
    #print(G.edges())
    if isinstance(G.nodes()[0], int):
        conversion = lambda n: int(n)
    else:
        conversion = lambda n: n

    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
    id_map = json.load(open(prefix + "-id_map.json"))
    id_map = {conversion(k): int(v) for k, v in id_map.items()}
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n: n
    else:
        lab_conversion = lambda n: int(n)

    class_map = {conversion(k): lab_conversion(v) for k, v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    ## (necessary because of networkx weirdness with the Reddit data)
    broken_count = 0
    for node in G.nodes():
        if not 'val' in G.nodes[node] or not 'test' in G.nodes[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    ## (some datasets might already have this..)
    print("Loaded data.. now preprocessing..")
    for edge in G.edges():
        if (G.nodes[edge[0]]['val'] or G.nodes[edge[1]]['val'] or
                G.nodes[edge[0]]['test'] or G.nodes[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[str(n)] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)

    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                walks.append(map(conversion, line.split()))

    return G, feats, id_map, walks, class_map
def myfunc(type='train'):
    G, feats, id_map, walks, class_map = load_data('/home/thummala/GraphSage/ppi/ppi')
    train=[]
    test=[]
    val=[]
    num_nodes=len(G.nodes())
    adj_lists = defaultdict(set)
    for edge in G.edges():
        adj_lists[edge[0]].add(edge[1])
        adj_lists[edge[1]].add(edge[0])
    labels = np.empty((num_nodes, len(class_map['0'])), dtype=np.int64)
    for i in G.nodes():
        labels[id_map[str(i)]] = class_map[str(id_map[str(i)])]
    for node in G.nodes(data=True):
        if node[1]['test']==False and node[1]['val']==False:
            train.append(node[0])
        elif node[1]['test']==True:
            test.append(node[0])
        else:
            val.append(node[0])
    print('train',len(train))
    print('test',len(test))
    print('val',len(val))
    # print(nx.isolates(G))
    # print(G.neighbors(20732))
    # if (20732,18454) in G.edges():
    #     print("yes")
    # elif (18454,20732) in G.edges():
    #     print("Oh no")
    # print(adj_lists[20732])
    # print(adj_lists[20803])
    for key in list(adj_lists.keys()):
        if adj_lists[key] == {}:
            print(key)
    return feats,labels,adj_lists,train,test,val

# myfunc()
# G, feats, id_map, walks, class_map = load_data('/home/thummala/GraphSage/ppi/ppi')
# for key in list(id_map.keys()):
#     if key != id_map[key]:
#         print(key,id_map[key])

# print(len(G.nodes()))
# print()
# print(feats.shape)
# print(len(id_map.keys()))
# print(id_map)
# print(len(class_map.keys()))
# print(len(class_map[0]))
# print(len(class_map[2708]))
# feats,labels,adj_lists=myfunc()
# print(feats.shape)
# print(adj_lists[0])
# print(labels[0])