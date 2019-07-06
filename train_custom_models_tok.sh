#!/usr/bin/env bash


echo "Starting script.........."

echo "DE-EN"

echo "Sutskever"

echo "====================================="

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --lr 0.002 --tok tok

echo "====================================="

echo "Improvements"

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64  --train 170000 --val 1000 --test 1000 --lr 0.002 --tok tok

echo "====================================="

echo "====================================="


echo "CUSTOM"

echo "Bidirectional"

echo "====================================="

python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 2 --dp 0.25 --bi True --model_type custom --reverse False --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --lr 0.002 --tok tok
echo "====================================="


echo "Improvements"

echo "====================================="

python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 4 --dp 0.25 --bi True --model_type custom --reverse False --epochs 80 -v 30000 -b 64  --train 170000 --val 1000 --test 1000 --lr 0.002 --tok tok
echo "====================================="

echo "====================================="



echo "Attention experiments"

echo "Additive"

### alterna en/de

echo "====================================="

python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 2 --dp 0.25 --bi False --model_type custom --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --attn additive --lr 0.002 --tok tok

python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 2 --dp 0.25 --bi True --model_type custom --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --attn additive --lr 0.002 --tok tok


##### 4 layers

python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 4 --dp 0.25 --bi False --model_type custom --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --attn additive --lr 0.002 --tok tok

python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 4 --dp 0.25 --bi True --model_type custom --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --attn additive --lr 0.002 --tok tok

echo "DOT"

python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 2 --dp 0.25 --bi False --model_type custom --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --attn dot --lr 0.002 --tok tok

python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 2 --dp 0.25 --bi True --model_type custom --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --attn dot --lr 0.002 --tok tok


##### 4 layers

python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 4 --dp 0.25 --bi False --model_type custom --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --attn dot --lr 0.002 --tok tok

python3 run_custom_nmt.py --hs 300 --emb 300  --nlayers 4 --dp 0.25 --bi True --model_type custom --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1000 --test 1000 --attn dot --lr 0.002 --tok tok


