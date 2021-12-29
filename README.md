
## GNN Neighbourhood Sampling Strategy - by Milind Thummala
#### forked from https://github.com/williamleif/GraphSAGE <br>


### Datasets
You can download the datasets from [download link](https://drive.google.com/drive/folders/1XpFrrHgVcGxJJts0zpka5f03Bhhl62jE?usp=sharing) <br>
The datasets used for the experiments so far are [WikiCS](https://arxiv.org/abs/2007.02901), [Cora](https://paperswithcode.com/dataset/cora) and [PPI](https://cs.stanford.edu/people/jure/pubs/graphsage-nips17.pdf)<br>

#### Requirements

pytorch >0.2 is required.

#### Running examples

Execute `python -m graphsage.main -h` to see how to use numerous neighbourhood sampling strategies (such as cluster teleport random walk, feature teleport random walk, khop with teleport, etc.) and to specify different parameters for these strategies. <br>

Example use: `python -m graphsage.main --dataset wikics --rand_walk True --aggregator mean --attention normal --iter 3 --outdir <the output directory to save the results> --n_layers 2 --includenodefeats no --typewalk customfeatureteleport --p 1 --q 1 --num_walks 10 --walk_length 10 --teleport 0.2` <br>

The functionality of the non-self explanatory flags is:<br>
p,q,num_walks, walk_length: Parameters to control the random walks generated as in [node2vec](https://arxiv.org/abs/1607.00653)
teleport: the teleportation probability if applicable <br>
aggregator: whether to use weighted_mean or mean aggregator while aggregating neighbourhood information. (See attention strategies) <br>
type_walk: the type of random walk to perform (ex: feature teleport random walk)(check neighbourhood sampling strategies for more details) <br>
iter: the number of iterations to perform the experiment (not to be confused with epochs) <br>
rand_walk: whether to use random walk based strategies or khop. <br>
attention: the type of attention to use (see attention strategies). <br>
n_layers: the number of GNN layers<br>

More examples are available in the experiments folder.



#### Neighbourhood Sampling Strategies
For now kindly refer to [link](https://drive.google.com/file/d/1Dc619c7ODd1nLHIQd7Pi0Xsb_BmDB8kU/view?usp=sharing)

#### Attention Strategies
For now kindly refer to [link](https://drive.google.com/file/d/1Dc619c7ODd1nLHIQd7Pi0Xsb_BmDB8kU/view?usp=sharing)

#### Results
The results so far can be found in [results](https://drive.google.com/drive/folders/1CYNFD9lgxdlIM_yiYmct1BBx2Cficyrq?usp=sharing)

#### Related Colab Notebooks
Experiments on Cora and how to use colab runtime for performing experiments --> [link](https://colab.research.google.com/drive/1I7zwTNGXFBmtZwlkR9FwU5cWM3-rz9i5?usp=sharing) <br>
For generating node properties, random walks, analyzing clusters --> [link](https://colab.research.google.com/drive/1Q8RZ1LL3H4neIbDOYCt9xl1O8trmHi1Z?usp=sharing) <br>
