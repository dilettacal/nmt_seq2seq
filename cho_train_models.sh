#!/usr/bin/env bash

echo "EN_DE"


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --nlayers 1 --dp 0.0 --reverse_input False --model_type c --reverse False --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --nlayers 1 --dp 0.25 --reverse_input False --model_type c --reverse False --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --nlayers 1 --dp 0.5 --reverse_input False --model_type c --reverse False --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000
echo "====================================="
echo "====================================="


echo "DE_EN"


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --nlayers 1 --dp 0.0 --reverse_input False --reverse True --model_type c --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --nlayers 1 --dp 0.25 --reverse_input False --reverse True --model_type c --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --nlayers 1 --dp 0.5 --reverse_input False --reverse True --model_type c --train 300000 --val 30000 --test 3000 --epochs 70 -v 30000
echo "====================================="
echo "====================================="