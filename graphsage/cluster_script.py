import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from sklearn.metrics import silhouette_score
from _collections import defaultdict
from sklearn.metrics import silhouette_samples, silhouette_score

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from graphsage.utils import myfunc,load_data
def convert(path1,save_path):
    CModel = joblib.load(path1)
    centroids = np.zeros(shape=(len(set(list(CModel.labels_))),50),dtype='float')
    G, feats, id_map, walks, class_map = load_data('/home/thummala/GraphSage/ppi/ppi')
    for i,label in enumerate(set(list(CModel.labels_))):
        centroids[i]=np.mean(feats[CModel.labels_ == label])
    np.save('ppi-centroids.npy',centroids)
    print(centroids)
    print(set(list(CModel.labels_)))
    print([CModel.labels_ == 1])
    print(CModel)
    pickle.dump(CModel, open(save_path, "wb"))
def load_cora(path):
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes,1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(path+'/cora.content') as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            feat_data[i,:] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open(path+"/cora.cites") as fp:
        for i,line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists

def load_pubmed(path):
    #hardcoded for simplicity...
    num_nodes = 19717
    num_feats = 500
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    with open(path+"/Pubmed-Diabetes.NODE.paper.tab") as fp:
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
    with open(path+"/Pubmed-Diabetes.DIRECTED.cites.tab") as fp:
        fp.readline()
        fp.readline()
        for line in fp:
            info = line.strip().split("\t")
            paper1 = node_map[info[1].split(":")[1]]
            paper2 = node_map[info[-1].split(":")[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)
    return feat_data, labels, adj_lists
feat_data, labels, adj_lists = load_cora('/home/thummala/graphsage-pytorch/cora')
#feat_data, labels, adj_lists = load_pubmed('/home/thummala/graphsage-pytorch/pubmed-data')
scores=[]
costs=[]
def test_kmeans(scores=[],start=2,end=8):
    for i in range(start,end):
        kmeans=KMeans(n_clusters=i, init='k-means++',random_state=0)
        y_kmeans = kmeans.fit_predict(feat_data)
        scores.append(silhouette_score(feat_data,y_kmeans))
        costs.append(kmeans.inertia_)
    fig, axs = plt.subplots(2)
    axs[0].plot(range(start,end),scores)
    axs[0].set_xlabel('value of k')
    axs[0].set_ylabel('silhouette score')
    axs[1].plot(range(start,end),costs)
    axs[1].set_xlabel('value of k')
    axs[1].set_ylabel('Cost')
    plt.show()
def save_cora_kmeans(path,clusters=3):
    kmeans=KMeans(n_clusters=clusters,init='k-means++',random_state=0)
    y_kmeans = kmeans.fit_predict(feat_data)
    count_dic={}
    for label in set(y_kmeans):
        count_dic[label]=0
    for label in y_kmeans:
        count_dic[label]+=1
    print(count_dic)
    pickle.dump(kmeans, open(path, "wb"))
def read_pickle(path):
    kmeans=pickle.load(open(path, "rb"))
    # y_kmeans = kmeans.fit_predict(feat_data)
    # count_dic={}
    # for label in set(y_kmeans):
    #     count_dic[label]=0
    # for label in y_kmeans:
    #     count_dic[label]+=1
    #print(count_dic)
    print(kmeans.cluster_centers_)
    print(kmeans.labels_[0])
def sil_analysis(range_n_clusters,X):
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
        plt.show()

#save_cora_kmeans(3)
#read_pickle("corakmeans_3.pkl")
#test_kmeans(start=30,end=40)
#save_cora_kmeans(path='corakmeans_7.pkl',clusters=7)
sil_analysis([2,3,4,5,6,7],feat_data)
#convert('/home/thummala/GraphSage/ppi_extra/ppi.skl','ppi_given.pkl')

