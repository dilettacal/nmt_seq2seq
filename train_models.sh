#!/usr/bin/env bash

echo "Starting script"

echo "DROPOUT 0.25"

echo "Sutskever"

echo "Vocabulary: 30000"

echo "DE_EN"
python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 4 --dp 0.25 --reverse_input True --model_type s --reverse True --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000

echo "EN_DE"
python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 4 --dp 0.25 --reverse_input True --model_type s --reverse False --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000

echo "====================================="


