#!/usr/bin/env bash


echo "Starting script.........."

echo "EN_DE"

echo "Sutskever"

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.5 --reverse_input True --model_type s --reverse False --epochs 50 -v 30000 -b 32 --train 30000 --val 1000 --test 1000 --lr 0.001

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.25 --reverse_input True --model_type s --reverse False  --epochs 50 -v 30000 -b 32 --train 30000 --val 1000 --test 1000 --lr 0.001


echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.0 --reverse_input True --model_type s --reverse False --epochs 50 -v 30000 -b 32 --train 30000 --val 1000 --test 1000 --lr 0.001



echo "====================================="
echo "====================================="



echo "CUSTOM"

echo "Bidirectional"

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.5 --bi True --model_type custom --reverse False --epochs 50 -v 30000 -b 32 --train 30000 --val 1000 --test 1000 --lr 0.001

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.0 --bi True --model_type custom --reverse False --epochs 50 -v 30000 -b 32 --train 30000 --val 1000 --test 1000 --lr 0.001


echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.25 --bi True --model_type custom --reverse False --epochs 50 -v 30000 -b 32 --train 30000 --val 1000 --test 1000 --lr 0.001

echo "====================================="

echo "====================================="

