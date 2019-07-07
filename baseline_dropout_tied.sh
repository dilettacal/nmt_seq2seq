#!/usr/bin/env bash

echo "Baseline model based on Sutskever"
echo "Fixed params: 300 embedding, hidden, LR=0.02, patience:10"
echo "Language combination: German>English, dataset: Europarl"
echo "No pretraiend, no tied"
echo "Test: Dropout values on 2 and 4 layers"

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --lr 0.002 --tok tok --tied False
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.50 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --lr 0.002 --tok tok --tied False
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.0 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --lr 0.002 --tok tok --tied False

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --lr 0.002 --tok tok --tied False
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.50 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --lr 0.002 --tok tok --tied False
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.0 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --lr 0.002 --tok tok --tied False

