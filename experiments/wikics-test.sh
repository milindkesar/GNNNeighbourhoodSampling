#!/bin/bash
python -m graphsage.main --epochs 20 --rw True --neighbours1 5 --aggregator mean --attention softmax --iter 2 --outdir /home/thummala/graphsage-pytorch/res/wikics_rw_mean
python -m graphsage.main --epochs 20 --rw False --neighbours1 5 --aggregator mean --attention softmax --iter 2 --outdir /home/thummala/graphsage-pytorch/res/wikics_hop_mean
