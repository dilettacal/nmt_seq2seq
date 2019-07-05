#!/usr/bin/env bash


echo "Starting script.........."

echo "DE-EN"

echo "Improvements"

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --lr 0.002 --corpus ""

echo "====================================="


echo "Bidirectional"

echo "====================================="

python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 4 --dp 0.25 --bi True --model_type custom --reverse True --epochs 80 -v 30000 -b 64 --lr 0.002 --corpus ""



echo "Attention experiments"


python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 4 --dp 0.25 --bi True --model_type custom --reverse True --epochs 80 -v 30000 -b 64 --lr 0.002 --corpus "" --attn dot

python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 4 --dp 0.25 --bi True --model_type custom --reverse True --epochs 80 -v 30000 -b 64 --lr 0.002 --corpus "" --attn additive

