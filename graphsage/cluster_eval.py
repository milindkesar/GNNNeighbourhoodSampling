import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from _collections import defaultdict
from sklearn.metrics import silhouette_samples, silhouette_score, normalized_mutual_info_score,adjusted_rand_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from graphsage.utils import myfunc,load_data

num_nodes=2708
num_feats=1433
kmeans=pickle.load(open('/home/thummala/graphsage-pytorch/graphsage/corakmeans_7.pkl', "rb"))
feature_labels = kmeans.labels_
feat_data = np.zeros((num_nodes, num_feats))
labels = np.empty((num_nodes, 1), dtype=np.int64)
node_map = {}
label_map = {}
with open('/home/thummala/graphsage-pytorch/cora/cora.content') as fp:
    for i, line in enumerate(fp):
        info = line.strip().split()
        feat_data[i, :] = list(map(float, info[1:-1]))
        node_map[info[0]] = i
        if not info[-1] in label_map:
            label_map[info[-1]] = len(label_map)
        labels[i] = label_map[info[-1]]
print("true labels",labels.reshape((-1)))
print("cluster labels",feature_labels)
print("NMI Score is",normalized_mutual_info_score(labels.reshape(-1),feature_labels))
print("Adjusted Rand Score",adjusted_rand_score(labels.reshape(-1),feature_labels))