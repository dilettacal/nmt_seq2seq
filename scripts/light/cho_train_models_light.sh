#!/usr/bin/env bash

echo "EN_DE"


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py  --hs 512 --emb 256  --nlayers 1 --dp 0.5 --reverse_input False --model_type c --reverse False  --epochs 50 -v 30000 -b 128 --train 30000 --val 1000 --test 1000 --lr 0.001
echo "====================================="

python3 run_custom_nmt.py  --hs 512 --emb 256  --nlayers 1 --dp 0.0 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 30000 --val 1000 --test 1000 --lr 0.001
echo "====================================="

python3 run_custom_nmt.py  --hs 512 --emb 256  --nlayers 1 --dp 0.25 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 30000 --val 1000 --test 1000 --lr 0.001

echo "====================================="
echo "====================================="


echo "DE_EN"


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py  --hs 512 --emb 256  --nlayers 1 --dp 0.5 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 30000 --val 1000 --test 1000 --lr 0.001
echo "====================================="

python3 run_custom_nmt.py  --hs 512 --emb 256  --nlayers 1 --dp 0.0 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 30000 --val 1000 --test 1000 --lr 0.001
echo "====================================="

python3 run_custom_nmt.py  --hs 512 --emb 256  --nlayers 1 --dp 0.25 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 30000 --val 1000 --test 1000 --lr 0.001
echo "====================================="
echo "====================================="
