#!/usr/bin/env bash

cd ../..

#### Adding more data to the same settings as in light script

echo "More data and same settings as in the light script"

echo "EN_DE"


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 1 --dp 0.5 --reverse_input False --model_type c --reverse False  --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 1 --dp 0.0 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 1 --dp 0.25 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000

echo "====================================="
echo "====================================="


echo "DE_EN"


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py --hs 750 --emb 256 --nlayers 1 --dp 0.5 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000
echo "====================================="

python3 run_custom_nmt.py --hs 750 --emb 256 --nlayers 1 --dp 0.0 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000
echo "====================================="

python3 run_custom_nmt.py --hs 750 --emb 256 --nlayers 1 --dp 0.25 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000
echo "====================================="
echo "====================================="

echo "Improvements in the architecture"

echo "EN_DE"


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py --hs 750 --emb 300 --nlayers 1 --dp 0.5 --reverse_input False --model_type c --reverse False  --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 300 --nlayers 1 --dp 0.0 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 300 --nlayers 1 --dp 0.25 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000

echo "====================================="
echo "====================================="


echo "DE_EN"


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 1 --dp 0.5 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 1 --dp 0.0 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000
echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 256 --nlayers 1 --dp 0.25 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 -b 128 --train 90000 --val 3000 --test 3000
echo "====================================="
echo "====================================="