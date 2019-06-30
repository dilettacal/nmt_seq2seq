#!/usr/bin/env bash


echo "DE_EN"


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256  --nlayers 2 --dp 0.5 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.0 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256  --nlayers 2 --dp 0.25 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001
echo "====================================="
echo "====================================="


echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256  --nlayers 4 --dp 0.5 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 4 --dp 0.0 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256  --nlayers 4 --dp 0.25 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001
echo "====================================="
echo "====================================="

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256  --nlayers 4 --dp 0.5 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001 --bi True
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 4 --dp 0.0 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001  --bi True
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256  --nlayers 4 --dp 0.25 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001  --bi True
echo "====================================="
echo "====================================="

echo "====================================="

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.5 --reverse_input False --model_type c --reverse True  --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001 --bi True
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.0 --reverse_input False --model_type c --reverse True --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001 --bi True
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.25 --reverse_input False --model_type c --reverse True --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001 --bi True



echo "EN_DE"


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.5 --reverse_input False --model_type c --reverse False  --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.0 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.25 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001

echo "====================================="
echo "====================================="

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 4 --dp 0.5 --reverse_input False --model_type c --reverse False  --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 4 --dp 0.0 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 4 --dp 0.25 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001

echo "====================================="
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 4 --dp 0.5 --reverse_input False --model_type c --reverse False  --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001 --bi True
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 4 --dp 0.0 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001 --bi True
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 4 --dp 0.25 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001 --bi True

echo "====================================="

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.5 --reverse_input False --model_type c --reverse False  --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001 --bi True
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.0 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001 --bi True
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 2 --dp 0.25 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000  --lr 0.001 --bi True

