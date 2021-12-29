#!/bin/bash

#Naming Convention --> Dataset_rw_n1_n2_aggregator_nlayers_softmax_nodefeats

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk False --neighbours1 25 --neighbours2 0 --aggregator mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_khop_25_0_mean_1_normal_no --n_layers 1

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk True --neighbours1 25 --neighbours2 0 --aggregator mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_rw_25_0_mean_1_normal_no --n_layers 1

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk True --neighbours1 60 --neighbours2 0 --aggregator mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_rw_60_0_mean_1_normal_no --n_layers 1

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk False --neighbours1 60 --neighbours2 0 --aggregator mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_khop_60_0_mean_1_normal_no --n_layers 1

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk False --neighbours1 25 --neighbours2 0 --aggregator weighted_mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_khop_25_0_weighted_mean_1_normal_no --n_layers 1

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk True --neighbours1 60 --neighbours2 0 --aggregator weighted_mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_rw_60_0_weightedmean_1_normal_no --n_layers 1

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk False --neighbours1 60 --neighbours2 0 --aggregator weighted_mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_khop_60_0_weightedmean_1_normal_no --n_layers 1

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk False --neighbours1 25 --neighbours2 25 --aggregator mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_khop_25_25_mean_2_normal_no --n_layers 2

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk True --neighbours1 25 --neighbours2 25 --aggregator mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_rw_25_25_mean_2_normal_no --n_layers 2

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk True --neighbours1 60 --neighbours2 60 --aggregator mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_rw_60_60_mean_2_normal_no --n_layers 2

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk False --neighbours1 60 --neighbours2 60 --aggregator mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_khop_60_60_mean_2_normal_no --n_layers 2

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk False --neighbours1 25 --neighbours2 25 --aggregator weighted_mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_khop_25_25_weighted_mean_2_normal_no --n_layers 2

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk True --neighbours1 60 --neighbours2 60 --aggregator weighted_mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_rw_60_60_weightedmean_2_normal_no --n_layers 2

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk False --neighbours1 60 --neighbours2 60 --aggregator weighted_mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_khop_60_60_weightedmean_2_normal_no --n_layers 2
##softmax
python -m graphsage.main --dataset ppi --epochs 200 --rand_walk False --neighbours1 25 --neighbours2 10 --aggregator weighted_mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_khop_25_10_weighted_mean_2_softmax_no --n_layers 2 --attention softmax

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk True --neighbours1 60 --neighbours2 60 --aggregator weighted_mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_rw_60_60_weightedmean_2_softmax_no --n_layers 2 --attention softmax

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk True --neighbours1 25 --neighbours2 10 --aggregator weighted_mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_rw_25_10_weightedmean_2_softmax_no --n_layers 2 --attention softmax
##nodefeats
python -m graphsage.main --dataset ppi --epochs 200 --rand_walk False --neighbours1 25 --neighbours2 10 --aggregator weighted_mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_khop_25_10_weightedmean_2_normal_yes --n_layers 2 --attention normal --includenodefeats yes

python -m graphsage.main --dataset ppi --epochs 200 --rand_walk True --neighbours1 60 --neighbours2 60 --aggregator weighted_mean --iter 1 --outdir /home/thummala/graphsage-pytorch/res/ppi_rw_60_60_weightedmean_2_normal_yes --n_layers 2 --attention normal --includenodefeats yes

