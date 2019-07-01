#!/usr/bin/env bash

#!/usr/bin/env bash


echo "Starting script.........."

echo "DE_EN"

echo "Sutskever"


echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --epochs 50 -v 30000 -b 64 --corpus ""
python3 run_custom_nmt.py --hs 512 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --epochs 50 -v 30000 -b 64 --corpus ""

echo "====================================="

python3 run_custom_nmt.py --hs 512 --emb 300 --nlayers 2 --dp 0.5 --reverse_input True --reverse True --epochs 50 -v 30000 -b 64 --corpus ""
python3 run_custom_nmt.py --hs 512 --emb 300 --nlayers 4 --dp 0.5 --reverse_input True --reverse True --epochs 50 -v 30000 -b 64 --corpus ""

echo "====================================="


python3 run_custom_nmt.py --hs 512 --emb 300  --nlayers 2 --dp 0.5 --bi True --model_type custom --reverse True --epochs 50 -v 30000 --corpus ""
python3 run_custom_nmt.py --hs 512 --emb 300  --nlayers 2 --dp 0.25 --bi True --model_type custom --reverse True --epochs 50 -v 30000 --corpus ""
python3 run_custom_nmt.py --hs 512 --emb 300  --nlayers 4 --dp 0.5 --bi True --model_type custom --reverse True --epochs 50 -v 30000 --corpus ""
python3 run_custom_nmt.py --hs 512 --emb 300  --nlayers 4 --dp 0.25 --bi True --model_type custom --reverse True --epochs 50 -v 30000 --corpus ""

echo "====================================="
