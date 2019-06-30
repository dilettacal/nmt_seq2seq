#!/usr/bin/env bash



echo "Starting script.........."

echo "DE_EN"

echo "Sutskever"

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.5 --reverse_input True --reverse True --model_type s --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.0 --reverse_input True --reverse True --model_type s --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001

echo "====================================="

echo "Improvements"

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 4 --dp 0.5 --reverse_input True --reverse True --model_type s --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 4 --dp 0.0 --reverse_input True --reverse True --model_type s --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001

echo "====================================="

echo "====================================="


echo "CUSTOM"

echo "Bidirectional"


echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256  --nlayers 2 --dp 0.5 --bi True --model_type custom --reverse True --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256  --nlayers 2 --dp 0.0 --bi True --model_type custom  --reverse True --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256  --nlayers 2 --dp 0.25 --bi True --model_type custom --reverse True --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001
echo "====================================="

echo "Improvements"

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256  --nlayers 4 --dp 0.5 --bi True --model_type custom --reverse True --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256  --nlayers 4 --dp 0.0 --bi True --model_type custom  --reverse True --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256  --nlayers 4 --dp 0.25 --bi True --model_type custom --reverse True --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001
echo "====================================="

echo "====================================="

