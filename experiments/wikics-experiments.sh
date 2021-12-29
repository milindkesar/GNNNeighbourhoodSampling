python -m graphsage.main --dataset wikics --epochs 100 --rand_walk True --neighbours1 60 --neighbours2 60 --aggregator mean --attention normal --iter 3 --outdir /home/thummala/graphsage-pytorch/res/wikics_frw_6060_mean --n_layers 2 --includenodefeats no --typewalk featureteleport

python -m graphsage.main --dataset wikics --epochs 100 --rand_walk True --neighbours1 20 --neighbours2 20 --aggregator mean --attention normal --iter 3 --outdir /home/thummala/graphsage-pytorch/res/wikics_frw_2020_mean --n_layers 2 --includenodefeats no --typewalk featureteleport

python -m graphsage.main --dataset wikics --epochs 100 --rand_walk True --neighbours1 60 --neighbours2 60 --aggregator mean --attention normal --iter 3 --outdir /home/thummala/graphsage-pytorch/res/wikics_crw_6060_mean --n_layers 2 --includenodefeats no --typewalk clusterteleport

python -m graphsage.main --dataset wikics --epochs 100 --rand_walk True --neighbours1 20 --neighbours2 20 --aggregator mean --attention normal --iter 3 --outdir /home/thummala/graphsage-pytorch/res/wikics_crw_2020_mean --n_layers 2 --includenodefeats no --typewalk clusterteleport

python -m graphsage.main --dataset wikics --epochs 100 --rand_walk True --neighbours1 60 --neighbours2 60 --aggregator mean --attention normal --iter 3 --outdir /home/thummala/graphsage-pytorch/res/wikics_rw_6060_mean --n_layers 2 --includenodefeats no --typewalk default

python -m graphsage.main --dataset wikics --epochs 100 --rand_walk True --neighbours1 20 --neighbours2 20 --aggregator mean --attention normal --iter 3 --outdir /home/thummala/graphsage-pytorch/res/wikics_rw_2020_mean --n_layers 2 --includenodefeats no --typewalk default

python -m graphsage.main --dataset wikics --epochs 100 --rand_walk False --neighbours1 60 --neighbours2 60 --aggregator mean --attention normal --iter 3 --outdir /home/thummala/graphsage-pytorch/res/wikics_khop_6060_mean --n_layers 2 --includenodefeats no --typewalk default

python -m graphsage.main --dataset wikics --epochs 100 --rand_walk False --neighbours1 20 --neighbours2 20 --aggregator mean --attention normal --iter 3 --outdir /home/thummala/graphsage-pytorch/res/wikics_khop_2020_mean --n_layers 2 --includenodefeats no --typewalk default