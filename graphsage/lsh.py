# import numpy as np
# import itertools
# import collections

# def cossim(u,v):
#     norm = np.linalg.norm(u)*np.linalg.norm(v)
#     cosine = u@v/norm
#     ang = np.arccos(cosine)
#     return 1-ang/np.pi

# def lsh_euclidean(A, b, r, thresh):
#   """ A must be an (N,D) matrix consisting of N real valued euclidean vectors.
#   Args:
#     A: the (N,D) matrix of vectors to with which to find similar pairs
#     b: the number of bands
#     r: the number of rows per band
#     thresh: a float value [-1,1] determining the required cosine similarity threshold
#   Returns:
#     a set of pairs with requisite similarity. Contains no false positives, but may omit False negatives
#   """
#   N, D = A.shape
#   n = b*r
  
#   # Compute signature matrix
#   R = A@np.random.randn(D, n)
#   S = np.where(R>0, 1, 0)

#   # Break into bands
#   S = np.split(S, b, axis=1)

#   # column vector to convert binary vector to integer e.g. (1,0,1)->5
#   binary_column = 2**np.arange(r).reshape(-1,  1)

#   # convert each band into a single integer, 
#   # i.e. convert band matrices to band columns
#   S = np.hstack([M@binary_column for M in S])

#   # Every value in the matrix represents a hash bucket assignment
#   # For every bucket in row i, add index i to that bucket
#   d = collections.defaultdict(set)
#   with np.nditer(S,flags=['multi_index']) as it:
#       for x in it:
#           d[int(x)].add(it.multi_index[0])

#   # For every bucket, find all pairs. These are the LSH pairs.
#   candidate_pairs = set()
#   for k,v in d.items():
#       if len(v) > 1:
#           for pair in itertools.combinations(v, 2):
#               candidate_pairs.add(tuple(sorted(pair)))

#   # Finally, perform the actually similarity computation
#   # to weed out false positives
#   lsh_pairs = set()
#   for (i, j) in candidate_pairs:
#       if cossim(A[i],A[j]) > thresh:
#           lsh_pairs.add((i, j))
          
#   return lsh_pairs
from collections import defaultdict
from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
import pandas as pd
from itertools import combinations


def generate_random_vectors(dim, n_vectors):
    """
    generate random projection vectors
    the dims comes first in the matrix's shape,
    so we can use it for matrix multiplication.
    """
    return np.random.randn(dim, n_vectors)
def train_lsh(X_tfidf, n_vectors, seed=None):    
    if seed is not None:
        np.random.seed(seed)

    dim = X_tfidf.shape[1]
    random_vectors = generate_random_vectors(dim, n_vectors)  

    # partition data points into bins,
    # and encode bin index bits into integers
    bin_indices_bits = X_tfidf.dot(random_vectors) >= 0
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)
    bin_indices = bin_indices_bits.dot(powers_of_two)

    # update `table` so that `table[i]` is the list of document ids with bin index equal to i
    table = defaultdict(list)
    for idx, bin_index in enumerate(bin_indices):
        table[bin_index].append(idx)
    
    # note that we're storing the bin_indices here
    # so we can do some ad-hoc checking with it,
    # this isn't actually required
    model = {'table': table,
             'random_vectors': random_vectors,
             'bin_indices': bin_indices,
             'bin_indices_bits': bin_indices_bits}
    return model

def search_nearby_bins(query_bin_bits, table, search_radius=3, candidate_set=None):
    """
    For a given query vector and trained LSH model's table
    return all candidate neighbors with the specified search radius.
    
    Example
    -------
    model = train_lsh(X_tfidf, n_vectors=16, seed=143)
    query = model['bin_index_bits'][0]  # vector for the first document
    candidates = search_nearby_bins(query, model['table'])
    """
    if candidate_set is None:
        candidate_set = set()

    n_vectors = query_bin_bits.shape[0]
    powers_of_two = 1 << np.arange(n_vectors - 1, -1, step=-1)

    for different_bits in combinations(range(n_vectors), search_radius):
        # flip the bits (n_1, n_2, ..., n_r) of the query bin to produce a new bit vector
        index = list(different_bits)
        alternate_bits = query_bin_bits.copy()
        alternate_bits[index] = np.logical_not(alternate_bits[index])

        # convert the new bit vector to an integer index
        nearby_bin = alternate_bits.dot(powers_of_two)

        # fetch the list of documents belonging to
        # the bin indexed by the new bit vector,
        # then add those documents to candidate_set;
        # make sure that the bin exists in the table
        if nearby_bin in table:
            candidate_set.update(table[nearby_bin])

    return candidate_set
def get_nearest_neighbors(X_tfidf, query_vector, model, max_search_radius=3):
    table = model['table']
    random_vectors = model['random_vectors']

    # compute bin index for the query vector, in bit representation.
    bin_index_bits = np.ravel(query_vector.dot(random_vectors) >= 0)

    # search nearby bins and collect candidates
    candidate_set = set()
    for search_radius in range(max_search_radius + 1):
        candidate_set = search_nearby_bins(bin_index_bits, table, search_radius, candidate_set)

    # sort candidates by their true distances from the query
    candidate_list = list(candidate_set)
    candidates = X_tfidf[candidate_list]
    #print(candidates.shape, query_vector.shape)
    distance = pairwise_distances(candidates, query_vector, metric='cosine').flatten()
    
    distance_col = 'similarity'
    nearest_neighbors = pd.DataFrame({
        'id': candidate_list, distance_col:1 - distance
    }).sort_values(distance_col, ascending=False).reset_index(drop=True)
    return nearest_neighbors[1:]
