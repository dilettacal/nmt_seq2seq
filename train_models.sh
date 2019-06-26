#!/usr/bin/env bash

echo "Starting script"

echo "DROPOUT 0.25"

echo "Sutskever"

echo "Vocabulary: 30000"

echo "DE_EN"
python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 4 --dp 0.25 --reverse_input True --model_type s --reverse True --train 500000 --val 50000 --test 5000 --epochs 70

echo "EN_DE"
python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 4 --dp 0.25 --reverse_input True --model_type s --reverse False --train 500000 --val 50000 --test 5000 --epochs 70

echo "====================================="
echo "Vocabulary: 50000"

echo "DE_EN"
python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 4 --dp 0.25 --reverse_input True --model_type s --reverse True --train 500000 --val 50000 --test 5000 -v 50000

echo "EN_DE"
python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 4 --dp 0.25 --reverse_input True --model_type s --reverse False --train 500000 --val 50000 --test 5000 -v 50000

echo "===================================="

echo "CHO"

echo "Vocabulary: 30000"

echo "DE_EN"
python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 1 --dp 0.25 --reverse_input False --model_type c --reverse True --train 500000 --val 50000 --test 5000 --epochs 70

echo "EN_DE"
python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 1 --dp 0.25 --reverse_input False --model_type c --reverse False --train 500000 --val 50000 --test 5000 --epochs 70

echo "====================================="
echo "Vocabulary: 50000"

echo "DE_EN"
python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 1 --dp 0.25 --reverse_input False --model_type c --reverse True --train 500000 --val 50000 --test 5000 -v 50000 --epochs 70

echo "EN_DE"
python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 1 --dp 0.25 --reverse_input False --model_type c --reverse False --train 500000 --val 50000 --test 5000 -v 50000 --epochs 70