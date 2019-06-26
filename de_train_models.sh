#!/usr/bin/env bash

echo "Starting script.........."

echo "DE_EN"

echo "Sutskever"

echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 4 --dp 0.0 --reverse_input True --reverse True --model_type s --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000

echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s  --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000

echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 4 --dp 0.5 --reverse_input True --reverse True --model_type s --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000


echo "====================================="
echo "====================================="


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 1 --dp 0.0 --reverse_input False --reverse True --model_type c --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 1 --dp 0.25 --reverse_input False --reverse True --model_type c --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 1 --dp 0.5 --reverse_input False --reverse True --model_type c --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000
echo "====================================="
echo "====================================="


echo "CUSTOM"

echo "Bidirectional"

echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 4 --dp 0.0 --bi True --model_type custom  --reverse True --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000

echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 4 --dp 0.25 --bi True --model_type custom --reverse True --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --lr 0.05 --nlayers 4 --dp 0.5 --bi True --model_type custom --reverse True --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000

echo "====================================="
echo "====================================="