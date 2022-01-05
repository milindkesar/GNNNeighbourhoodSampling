import random
import numpy as np
import gensim
import os
from collections import defaultdict
from joblib import Parallel, delayed, load, dump
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm
from graphsage.lsh import train_lsh,get_nearest_neighbors
import networkx as nx

class ClusterTeleport_RandomWalk:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph, walk_length=80, num_walks=10, p=1, q=1, weight_key='weight',
                 workers=1, sampling_strategy=None, quiet=False, temp_folder=None, cluster_labels=None, clusterteleport_weight=0.1):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.
        :param graph: Input graph
        :type graph: Networkx Graph
        :param walk_length: Number of nodes in each walk (default: 80)
        :type walk_length: int
        :param num_walks: Number of walks per node (default: 10)
        :type num_walks: int
        :param p: Return hyper parameter (default: 1)
        :type p: float
        :param q: Inout parameter (default: 1)
        :type q: float
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :type weight_key: str
        :param workers: Number of workers for parallel execution (default: 1)
        :type workers: int
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        :type temp_folder: str
        """

        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.d_graph = defaultdict(dict)

        self.cluster_labels = cluster_labels
        self.clusterteleport_weight = clusterteleport_weight
        degree_dict = dict(graph.degree(graph.nodes()))
        self.avg_degree = sum(degree_dict.values())/len(degree_dict.values())
        self.features_exist=False
        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError(
                    "temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"
        self._preprocess()
        self._precompute_probabilities()
        self.walks = self._generate_walks()

    def _preprocess(self):
        for edge in self.graph.edges():
            self.graph[edge[0]][edge[1]]['weight'] = 1
        spcl_node=-1
        edges = []
        for edge in self.graph.edges():
            self.graph[edge[0]][edge[1]]['weight'] = 1
        for node in self.graph.nodes():
            edges.append((node, spcl_node, self.clusterteleport_weight))
        self.graph.add_weighted_edges_from(edges)

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph
        first_travel_done = set()

        nodes_generator = self.graph.nodes() if self.quiet else tqdm(
            self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:
            if source == -1:
                d_graph[source] = {}
                continue
            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):
                if current_node == -1:
                     continue

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                first_travel_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if self.graph[current_node][destination].get(self.weight_key,-1) == self.clusterteleport_weight:##Cluster probability
                        ss_weight = self.clusterteleport_weight

                    elif destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(
                            self.weight_key, 1) * 1 / p
                    # If the neighbor is connected to the source
                    elif destination in self.graph[source]:
                        ss_weight = self.graph[current_node][destination].get(
                            self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(
                            self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(
                            self.graph[current_node][destination].get(self.weight_key, 1))
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

                if current_node not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    d_graph[current_node][self.FIRST_TRAVEL_KEY] = unnormalized_weights / \
                        unnormalized_weights.sum()
                    first_travel_done.add(current_node)

                # Save neighbors
                d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors
        for node in nodes_generator:
            if self.NEIGHBORS_KEY not in d_graph[node]:
                d_graph[node][self.NEIGHBORS_KEY] = [-1]
                d_graph[node][self.FIRST_TRAVEL_KEY] = [1]

    def _generate_walks(self):
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        def flatten(l): return [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(clusterteleport_parallel_generate_walks)(self.d_graph,
                                             self.walk_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet,
                                             self.cluster_labels) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks


def clusterteleport_parallel_generate_walks(d_graph, global_walk_length, num_walks, cpu_num, sampling_strategy=None,
                            num_walks_key=None, walk_length_key=None, neighbors_key=None, probabilities_key=None,
                            first_travel_key=None, quiet=False, cluster_labels = None):
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """
    def do_cluster_teleport(source):
        current_label = cluster_labels[int(source)]
        potential_nodes = []
        for i,newlabel in enumerate(cluster_labels):
            if (current_label == newlabel) and (int(i) != int(source)):
                potential_nodes.append(i)
        if potential_nodes != []:
            destination = np.random.choice(potential_nodes, size=1)[0]
        else:
            destination = source
            print('some error, most likely only node with that cluster label')
        #print('cluster teleport')
        return destination

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks,
                    desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:
            if source == -1:
                continue

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(
                    walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                # if not walk_options:
                #     break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = np.random.choice(
                        walk_options, size=1, p=probabilities)[0]
                else:
                    ##it may be possible that the node t might not be in neighbourhood of v due to teleportation(node2vec)
                    if walk[-2] in d_graph[walk[-1]][probabilities_key]:
                        probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    else:
                        probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = np.random.choice(
                        walk_options, size=1, p=probabilities)[0]
                if int(walk_to) == -1:
                    walk_to = do_cluster_teleport(walk[-1])

                walk.append(str(walk_to))

            walk = list(map(int, walk))  # Convert all to strings
            walks.append(walk)
    if not quiet:
        pbar.close()

    return walks

class Node2Vec:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph: nx.Graph, dimensions: int = 128, walk_length: int = 80, num_walks: int = 10, p: float = 1,
                 q: float = 1, weight_key: str = 'weight', workers: int = 1, sampling_strategy: dict = None,
                 quiet: bool = False, temp_folder: str = None, seed: int = None):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.
        :param graph: Input graph
        :param dimensions: Embedding dimensions (default: 128)
        :param walk_length: Number of nodes in each walk (default: 80)
        :param num_walks: Number of walks per node (default: 10)
        :param p: Return hyper parameter (default: 1)
        :param q: Inout parameter (default: 1)
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :param workers: Number of workers for parallel execution (default: 1)
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        :param seed: Seed for the random number generator.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        """

        self.graph = graph
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.d_graph = defaultdict(dict)

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError("temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self._precompute_probabilities()
        self.walks = self._generate_walks()

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph

        nodes_generator = self.graph.nodes() if self.quiet \
            else tqdm(self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:

            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / p
                    elif destination in self.graph[source]:  # If the neighbor is connected to the source
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

            # Calculate first_travel weights for source
            first_travel_weights = []

            for destination in self.graph.neighbors(source):
                first_travel_weights.append(self.graph[source][destination].get(self.weight_key, 1))

            first_travel_weights = np.array(first_travel_weights)
            d_graph[source][self.FIRST_TRAVEL_KEY] = first_travel_weights / first_travel_weights.sum()

            # Save neighbors
            d_graph[source][self.NEIGHBORS_KEY] = list(self.graph.neighbors(source))

    def _generate_walks(self) -> list:
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        flatten = lambda l: [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(parallel_generate_walks)(self.d_graph,
                                             self.walk_length,
                                             len(num_walks),
                                             idx,
                                             self.sampling_strategy,
                                             self.NUM_WALKS_KEY,
                                             self.WALK_LENGTH_KEY,
                                             self.NEIGHBORS_KEY,
                                             self.PROBABILITIES_KEY,
                                             self.FIRST_TRAVEL_KEY,
                                             self.quiet) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks

def parallel_generate_walks(d_graph: dict, global_walk_length: int, num_walks: int, cpu_num: int,
                            sampling_strategy: dict = None, num_walks_key: str = None, walk_length_key: str = None,
                            neighbors_key: str = None, probabilities_key: str = None, first_travel_key: str = None,
                            quiet: bool = False) -> list:
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks, desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # Skip dead end nodes
                if not walk_options:
                    break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]
                else:
                    probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    walk_to = random.choices(walk_options, weights=probabilities)[0]

                walk.append(walk_to)

            walk = list(map(str, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks

class FeatureTeleport_RandomWalk:
    FIRST_TRAVEL_KEY = 'first_travel_key'
    PROBABILITIES_KEY = 'probabilities'
    NEIGHBORS_KEY = 'neighbors'
    WEIGHT_KEY = 'weight'
    NUM_WALKS_KEY = 'num_walks'
    WALK_LENGTH_KEY = 'walk_length'
    P_KEY = 'p'
    Q_KEY = 'q'

    def __init__(self, graph, node_features, walk_length=80, num_walks=10, p=1, q=1, weight_key='weight',
                 workers=1, sampling_strategy=None, quiet=False, temp_folder=None,
                 teleport_weight=0.1):
        """
        Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.
        :param graph: Input graph
        :type graph: Networkx Graph
        :param walk_length: Number of nodes in each walk (default: 80)
        :type walk_length: int
        :param num_walks: Number of walks per node (default: 10)
        :type num_walks: int
        :param p: Return hyper parameter (default: 1)
        :type p: float
        :param q: Inout parameter (default: 1)
        :type q: float
        :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
        :type weight_key: str
        :param workers: Number of workers for parallel execution (default: 1)
        :type workers: int
        :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
        Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
        :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
        :type temp_folder: str
        """

        self.graph = graph
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.weight_key = weight_key
        self.workers = workers
        self.quiet = quiet
        self.d_graph = defaultdict(dict)
        self.teleport_weight = teleport_weight
        degree_dict = dict(graph.degree(graph.nodes()))
        self.avg_degree = sum(degree_dict.values()) / len(degree_dict.values())
        self.features_exist = True
        self.node_features = np.array(node_features)

        if sampling_strategy is None:
            self.sampling_strategy = {}
        else:
            self.sampling_strategy = sampling_strategy

        self.temp_folder, self.require = None, None
        if temp_folder:
            if not os.path.isdir(temp_folder):
                raise NotADirectoryError(
                    "temp_folder does not exist or is not a directory. ({})".format(temp_folder))

            self.temp_folder = temp_folder
            self.require = "sharedmem"
        self.pairwisesim = self._preprocess()
        self._precompute_probabilities()
        self.walks = self._generate_walks()

    def _preprocess(self):
        temp = {}
        if self.features_exist:
            spcl_node = -1
            if self.features_exist:
                n_vectors = 16
                model = train_lsh(self.node_features, n_vectors, seed=143)
                for item_id in range(self.node_features.shape[0]):
                    temp[item_id] = {}
                    query_vector = self.node_features[item_id]
                    nearest_neighbors = get_nearest_neighbors(self.node_features, query_vector.reshape(1, -1), model,
                                                              max_search_radius=2)
                    if len(nearest_neighbors) < 20:
                        radius = 3
                        while True:
                            nearest_neighbors = get_nearest_neighbors(self.node_features, query_vector.reshape(1, -1),
                                                                      model, max_search_radius=radius)
                            if len(nearest_neighbors) > 20:
                                break
                            radius = radius + 1
                    for i, row in nearest_neighbors[:100].iterrows():
                        temp[item_id][int(row['id'])] = row['similarity']
                    # print(item_id)
            edges = []
            for edge in self.graph.edges():
                self.graph[edge[0]][edge[1]]['weight'] = 1
            for node in self.graph.nodes():
                edges.append((node, spcl_node, self.teleport_weight))
            self.graph.add_weighted_edges_from(edges)
        return temp

    def _precompute_probabilities(self):
        """
        Precomputes transition probabilities for each node.
        """

        d_graph = self.d_graph
        first_travel_done = set()

        nodes_generator = self.graph.nodes() if self.quiet else tqdm(
            self.graph.nodes(), desc='Computing transition probabilities')

        for source in nodes_generator:
            if source == -1:
                d_graph[source] = {}
                continue
            # Init probabilities dict for first travel
            if self.PROBABILITIES_KEY not in d_graph[source]:
                d_graph[source][self.PROBABILITIES_KEY] = dict()

            for current_node in self.graph.neighbors(source):
                if current_node == -1:
                    continue

                # Init probabilities dict
                if self.PROBABILITIES_KEY not in d_graph[current_node]:
                    d_graph[current_node][self.PROBABILITIES_KEY] = dict()

                unnormalized_weights = list()
                first_travel_weights = list()
                d_neighbors = list()

                # Calculate unnormalized weights
                for destination in self.graph.neighbors(current_node):

                    p = self.sampling_strategy[current_node].get(self.P_KEY,
                                                                 self.p) if current_node in self.sampling_strategy else self.p
                    q = self.sampling_strategy[current_node].get(self.Q_KEY,
                                                                 self.q) if current_node in self.sampling_strategy else self.q

                    if self.graph[current_node][destination].get(self.weight_key,
                                                                 -1) == self.teleport_weight:  ##Cluster probability
                        ss_weight = self.teleport_weight

                    elif destination == source:  # Backwards probability
                        ss_weight = self.graph[current_node][destination].get(
                            self.weight_key, 1) * 1 / p
                    # If the neighbor is connected to the source
                    elif destination in self.graph[source]:
                        ss_weight = self.graph[current_node][destination].get(
                            self.weight_key, 1)
                    else:
                        ss_weight = self.graph[current_node][destination].get(
                            self.weight_key, 1) * 1 / q

                    # Assign the unnormalized sampling strategy weight, normalize during random walk
                    unnormalized_weights.append(ss_weight)
                    if current_node not in first_travel_done:
                        first_travel_weights.append(
                            self.graph[current_node][destination].get(self.weight_key, 1))
                    d_neighbors.append(destination)

                # Normalize
                unnormalized_weights = np.array(unnormalized_weights)
                d_graph[current_node][self.PROBABILITIES_KEY][
                    source] = unnormalized_weights / unnormalized_weights.sum()

                if current_node not in first_travel_done:
                    unnormalized_weights = np.array(first_travel_weights)
                    d_graph[current_node][self.FIRST_TRAVEL_KEY] = unnormalized_weights / \
                                                                   unnormalized_weights.sum()
                    first_travel_done.add(current_node)

                # Save neighbors
                d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors
        for node in nodes_generator:
            if self.NEIGHBORS_KEY not in d_graph[node]:
                d_graph[node][self.NEIGHBORS_KEY] = [-1]
                d_graph[node][self.FIRST_TRAVEL_KEY] = [1]

    def _generate_walks(self):
        """
        Generates the random walks which will be used as the skip-gram input.
        :return: List of walks. Each walk is a list of nodes.
        """

        def flatten(l): return [item for sublist in l for item in sublist]

        # Split num_walks for each worker
        num_walks_lists = np.array_split(range(self.num_walks), self.workers)

        walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
            delayed(featureteleport_parallel_generate_walks)(self.d_graph,
                                                    self.walk_length,
                                                    len(num_walks),
                                                    idx,
                                                    self.sampling_strategy,
                                                    self.NUM_WALKS_KEY,
                                                    self.WALK_LENGTH_KEY,
                                                    self.NEIGHBORS_KEY,
                                                    self.PROBABILITIES_KEY,
                                                    self.FIRST_TRAVEL_KEY,
                                                    self.quiet,
                                                    self.pairwisesim) for
            idx, num_walks
            in enumerate(num_walks_lists, 1))

        walks = flatten(walk_results)

        return walks


def featureteleport_parallel_generate_walks(d_graph, global_walk_length, num_walks, cpu_num, sampling_strategy=None,
                                   num_walks_key=None, walk_length_key=None, neighbors_key=None, probabilities_key=None,
                                   first_travel_key=None, quiet=False, pairwisesim=None):
    """
    Generates the random walks which will be used as the skip-gram input.
    :return: List of walks. Each walk is a list of nodes.
    """

    def do_feature_teleport(source):
        temp = pairwisesim[int(source)]
        probabilities = np.array(list(temp.values()))
        probabilities = probabilities / probabilities.sum()
            # print("prob",probabilities.shape)
        try:
            destination = np.random.choice(list(temp.keys()), size=1, p=probabilities.flatten())[0]
                # print('success')
        except:
            print('oh no the problamatic node is', source)
            destination = source
        return destination

    walks = list()

    if not quiet:
        pbar = tqdm(total=num_walks,
                    desc='Generating walks (CPU: {})'.format(cpu_num))

    for n_walk in range(num_walks):

        # Update progress bar
        if not quiet:
            pbar.update(1)

        # Shuffle the nodes
        shuffled_nodes = list(d_graph.keys())
        random.shuffle(shuffled_nodes)

        # Start a random walk from every node
        for source in shuffled_nodes:
            if source == -1:
                continue

            # Skip nodes with specific num_walks
            if source in sampling_strategy and \
                    num_walks_key in sampling_strategy[source] and \
                    sampling_strategy[source][num_walks_key] <= n_walk:
                continue

            # Start walk
            walk = [source]

            # Calculate walk length
            if source in sampling_strategy:
                walk_length = sampling_strategy[source].get(
                    walk_length_key, global_walk_length)
            else:
                walk_length = global_walk_length

            # Perform walk
            while len(walk) < walk_length:

                walk_options = d_graph[walk[-1]].get(neighbors_key, None)

                # # Skip dead end nodes
                # if not walk_options:
                #     break

                if len(walk) == 1:  # For the first step
                    probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = np.random.choice(
                        walk_options, size=1, p=probabilities)[0]
                else:
                    ##it may be possible that the node t might not be in neighbourhood of v due to teleportation(node2vec)
                    if walk[-2] in d_graph[walk[-1]][probabilities_key]:
                        probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
                    else:
                        probabilities = d_graph[walk[-1]][first_travel_key]
                    walk_to = np.random.choice(
                        walk_options, size=1, p=probabilities)[0]
                if int(walk_to) == -1:
                    walk_to = do_feature_teleport(walk[-1])

                walk.append(str(walk_to))

            walk = list(map(int, walk))  # Convert all to strings

            walks.append(walk)

    if not quiet:
        pbar.close()

    return walks


#
# class Cluster_RandomWalk:
#     FIRST_TRAVEL_KEY = 'first_travel_key'
#     PROBABILITIES_KEY = 'probabilities'
#     NEIGHBORS_KEY = 'neighbors'
#     WEIGHT_KEY = 'weight'
#     NUM_WALKS_KEY = 'num_walks'
#     WALK_LENGTH_KEY = 'walk_length'
#     P_KEY = 'p'
#     Q_KEY = 'q'
#
#     def __init__(self, graph, walk_length=80, num_walks=10, p=1, q=1, weight_key='weight',
#                  workers=1, sampling_strategy=None, quiet=False, temp_folder=None, cluster_labels=None,
#                  cluster_weight=0.1, node_features=None):
#         """
#         Initiates the Node2Vec object, precomputes walking probabilities and generates the walks.
#         :param graph: Input graph
#         :type graph: Networkx Graph
#         :param walk_length: Number of nodes in each walk (default: 80)
#         :type walk_length: int
#         :param num_walks: Number of walks per node (default: 10)
#         :type num_walks: int
#         :param p: Return hyper parameter (default: 1)
#         :type p: float
#         :param q: Inout parameter (default: 1)
#         :type q: float
#         :param weight_key: On weighted graphs, this is the key for the weight attribute (default: 'weight')
#         :type weight_key: str
#         :param workers: Number of workers for parallel execution (default: 1)
#         :type workers: int
#         :param sampling_strategy: Node specific sampling strategies, supports setting node specific 'q', 'p', 'num_walks' and 'walk_length'.
#         Use these keys exactly. If not set, will use the global ones which were passed on the object initialization
#         :param temp_folder: Path to folder with enough space to hold the memory map of self.d_graph (for big graphs); to be passed joblib.Parallel.temp_folder
#         :type temp_folder: str
#         """
#
#         self.graph = graph
#         self.walk_length = walk_length
#         self.num_walks = num_walks
#         self.p = p
#         self.q = q
#         self.weight_key = weight_key
#         self.workers = workers
#         self.quiet = quiet
#         self.d_graph = defaultdict(dict)
#
#         self.cluster_labels = cluster_labels
#         self.cluster_weight = cluster_weight
#         degree_dict = dict(graph.degree(graph.nodes()))
#         self.avg_degree = sum(degree_dict.values()) / len(degree_dict.values())
#         self.features_exist = False
#         if node_features != None:
#             self.features_exist = True
#             self.node_features = np.array(node_features)
#
#         if sampling_strategy is None:
#             self.sampling_strategy = {}
#         else:
#             self.sampling_strategy = sampling_strategy
#
#         self.temp_folder, self.require = None, None
#         if temp_folder:
#             if not os.path.isdir(temp_folder):
#                 raise NotADirectoryError(
#                     "temp_folder does not exist or is not a directory. ({})".format(temp_folder))
#
#             self.temp_folder = temp_folder
#             self.require = "sharedmem"
#         # print("Hi this is updated version")
#         self.pairwisesim = self._preprocess2()
#         # self._preprocess()
#         self._precompute_probabilities()
#         self.walks = self._generate_walks()
#
#     def _preprocess(self):
#         if self.cluster_labels != None:
#             for edge in self.graph.edges():
#                 self.graph[edge[0]][edge[1]]['weight'] = 1
#             for node in self.graph.nodes():
#                 temp = []
#                 count = 0
#                 items = list(self.cluster_labels.items())
#                 random.shuffle(items)
#                 for key, label in items:
#                     if count == int(self.avg_degree):
#                         break
#                     if (label == self.cluster_labels[node]) and (node != key) and (key not in self.graph[node]):
#                         temp.append((node, key, self.cluster_weight))
#                         count = count + 1
#                 self.graph.add_weighted_edges_from(temp)
#         else:
#             pass
#
#     def _preprocess2(self):
#         temp = {}
#         if self.features_exist:
#             spcl_node = -1
#             if self.features_exist:
#                 # candidate_pairs = lsh_euclidean(np.array(self.node_features), 1024,10,0.5)
#                 # for i, node1_f in enumerate(self.node_features):
#                 # 	temp[str(i+1)] = {}
#                 # 	for j, node2_f in enumerate(self.node_features):
#                 # 		temp[str(i+1)][str(j+1)] = dot(node1_f, node2_f)/(norm(node1_f)*norm(node2_f))
#                 n_vectors = 16
#                 model = train_lsh(self.node_features, n_vectors, seed=143)
#                 for item_id in range(self.node_features.shape[0]):
#                     temp[item_id] = {}
#                     query_vector = self.node_features[item_id]
#                     nearest_neighbors = get_nearest_neighbors(self.node_features, query_vector.reshape(1, -1), model,
#                                                               max_search_radius=2)
#                     if len(nearest_neighbors) < 20:
#                         radius = 3
#                         while True:
#                             nearest_neighbors = get_nearest_neighbors(self.node_features, query_vector.reshape(1, -1),
#                                                                       model, max_search_radius=radius)
#                             if len(nearest_neighbors) > 20:
#                                 break
#                             radius = radius + 1
#                     for i, row in nearest_neighbors[:100].iterrows():
#                         temp[item_id][int(row['id'])] = row['similarity']
#                     # print(item_id)
#             edges = []
#             for edge in self.graph.edges():
#                 self.graph[edge[0]][edge[1]]['weight'] = 1
#             for node in self.graph.nodes():
#                 edges.append((node, spcl_node, self.cluster_weight))
#             self.graph.add_weighted_edges_from(edges)
#         # print("temp",temp)
#         return temp
#
#     def _precompute_probabilities(self):
#         """
#         Precomputes transition probabilities for each node.
#         """
#
#         d_graph = self.d_graph
#         first_travel_done = set()
#
#         nodes_generator = self.graph.nodes() if self.quiet else tqdm(
#             self.graph.nodes(), desc='Computing transition probabilities')
#
#         for source in nodes_generator:
#             if source == -1:
#                 d_graph[source] = {}
#                 continue
#             # Init probabilities dict for first travel
#             if self.PROBABILITIES_KEY not in d_graph[source]:
#                 d_graph[source][self.PROBABILITIES_KEY] = dict()
#
#             for current_node in self.graph.neighbors(source):
#                 if current_node == -1:
#                     continue
#
#                 # Init probabilities dict
#                 if self.PROBABILITIES_KEY not in d_graph[current_node]:
#                     d_graph[current_node][self.PROBABILITIES_KEY] = dict()
#
#                 unnormalized_weights = list()
#                 first_travel_weights = list()
#                 d_neighbors = list()
#
#                 # Calculate unnormalized weights
#                 for destination in self.graph.neighbors(current_node):
#
#                     p = self.sampling_strategy[current_node].get(self.P_KEY,
#                                                                  self.p) if current_node in self.sampling_strategy else self.p
#                     q = self.sampling_strategy[current_node].get(self.Q_KEY,
#                                                                  self.q) if current_node in self.sampling_strategy else self.q
#
#                     if self.graph[current_node][destination].get(self.weight_key,
#                                                                  -1) == self.cluster_weight:  ##Cluster probability
#                         ss_weight = self.cluster_weight
#
#                     elif destination == source:  # Backwards probability
#                         ss_weight = self.graph[current_node][destination].get(
#                             self.weight_key, 1) * 1 / p
#                     # If the neighbor is connected to the source
#                     elif destination in self.graph[source]:
#                         ss_weight = self.graph[current_node][destination].get(
#                             self.weight_key, 1)
#                     else:
#                         ss_weight = self.graph[current_node][destination].get(
#                             self.weight_key, 1) * 1 / q
#
#                     # Assign the unnormalized sampling strategy weight, normalize during random walk
#                     unnormalized_weights.append(ss_weight)
#                     if current_node not in first_travel_done:
#                         first_travel_weights.append(
#                             self.graph[current_node][destination].get(self.weight_key, 1))
#                     d_neighbors.append(destination)
#
#                 # Normalize
#                 unnormalized_weights = np.array(unnormalized_weights)
#                 d_graph[current_node][self.PROBABILITIES_KEY][
#                     source] = unnormalized_weights / unnormalized_weights.sum()
#
#                 if current_node not in first_travel_done:
#                     unnormalized_weights = np.array(first_travel_weights)
#                     d_graph[current_node][self.FIRST_TRAVEL_KEY] = unnormalized_weights / \
#                                                                    unnormalized_weights.sum()
#                     first_travel_done.add(current_node)
#
#                 # Save neighbors
#                 d_graph[current_node][self.NEIGHBORS_KEY] = d_neighbors
#         for node in nodes_generator:
#             if self.NEIGHBORS_KEY not in d_graph[node]:
#                 d_graph[node][self.NEIGHBORS_KEY] = [-1]
#                 d_graph[node][self.FIRST_TRAVEL_KEY] = [1]
#
#     def _generate_walks(self):
#         """
#         Generates the random walks which will be used as the skip-gram input.
#         :return: List of walks. Each walk is a list of nodes.
#         """
#
#         def flatten(l): return [item for sublist in l for item in sublist]
#
#         # Split num_walks for each worker
#         num_walks_lists = np.array_split(range(self.num_walks), self.workers)
#
#         walk_results = Parallel(n_jobs=self.workers, temp_folder=self.temp_folder, require=self.require)(
#             delayed(custom_parallel_generate_walks)(self.d_graph,
#                                                     self.walk_length,
#                                                     len(num_walks),
#                                                     idx,
#                                                     self.sampling_strategy,
#                                                     self.NUM_WALKS_KEY,
#                                                     self.WALK_LENGTH_KEY,
#                                                     self.NEIGHBORS_KEY,
#                                                     self.PROBABILITIES_KEY,
#                                                     self.FIRST_TRAVEL_KEY,
#                                                     self.quiet,
#                                                     self.pairwisesim) for
#             idx, num_walks
#             in enumerate(num_walks_lists, 1))
#
#         walks = flatten(walk_results)
#
#         return walks
#
#
# def custom_parallel_generate_walks(d_graph, global_walk_length, num_walks, cpu_num, sampling_strategy=None,
#                                    num_walks_key=None, walk_length_key=None, neighbors_key=None, probabilities_key=None,
#                                    first_travel_key=None, quiet=False, pairwisesim=None):
#     """
#     Generates the random walks which will be used as the skip-gram input.
#     :return: List of walks. Each walk is a list of nodes.
#     """
#
#     def do_feature_teleport(source):
#         if pairwisesim != None:
#             temp = pairwisesim[source]
#             probabilities = np.array(list(temp.values()))
#             probabilities = probabilities / probabilities.sum()
#             # print("prob",probabilities.shape)
#             try:
#                 destination = np.random.choice(list(temp.keys()), size=1, p=probabilities.flatten())[0]
#                 # print('success')
#             except:
#                 print('oh no the problamatic node is', source)
#                 destination = source
#         return destination
#
#     walks = list()
#
#     if not quiet:
#         pbar = tqdm(total=num_walks,
#                     desc='Generating walks (CPU: {})'.format(cpu_num))
#
#     for n_walk in range(num_walks):
#
#         # Update progress bar
#         if not quiet:
#             pbar.update(1)
#
#         # Shuffle the nodes
#         shuffled_nodes = list(d_graph.keys())
#         random.shuffle(shuffled_nodes)
#
#         # Start a random walk from every node
#         for source in shuffled_nodes:
#             if source == -1:
#                 continue
#
#             # Skip nodes with specific num_walks
#             if source in sampling_strategy and \
#                     num_walks_key in sampling_strategy[source] and \
#                     sampling_strategy[source][num_walks_key] <= n_walk:
#                 continue
#
#             # Start walk
#             walk = [source]
#
#             # Calculate walk length
#             if source in sampling_strategy:
#                 walk_length = sampling_strategy[source].get(
#                     walk_length_key, global_walk_length)
#             else:
#                 walk_length = global_walk_length
#
#             # Perform walk
#             while len(walk) < walk_length:
#
#                 walk_options = d_graph[walk[-1]].get(neighbors_key, None)
#
#                 # Skip dead end nodes
#                 if not walk_options:
#                     break
#
#                 if len(walk) == 1:  # For the first step
#                     probabilities = d_graph[walk[-1]][first_travel_key]
#                     walk_to = np.random.choice(
#                         walk_options, size=1, p=probabilities)[0]
#                 else:
#                     try:
#                         probabilities = d_graph[walk[-1]][probabilities_key][walk[-2]]
#                     except:
#                         probabilities = d_graph[walk[-1]][first_travel_key]
#                     walk_to = np.random.choice(
#                         walk_options, size=1, p=probabilities)[0]
#                 if int(walk_to) == -1:
#                     walk_to = do_feature_teleport(walk[-1])
#
#                 walk.append(walk_to)
#
#             walk = list(map(int, walk))  # Convert all to strings
#
#             walks.append(walk)
#
#     if not quiet:
#         pbar.close()
#
#     return walks
