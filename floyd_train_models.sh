#!/usr/bin/env bash

pip install dill unidecode nltk sacremoses spacy
pip install git+https://github.com/amake/tmx2corpus.git

echo "EN_DE"


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --nlayers 1 --dp 0.0 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 --max_len 15 -b 128 --data_dir /floyd/input/nmt_dataset
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --nlayers 1 --dp 0.25 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 --max_len 15 -b 128 --data_dir /floyd/input/nmt_dataset
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --nlayers 1 --dp 0.5 --reverse_input False --model_type c --reverse False --epochs 50 -v 30000 --max_len 15 -b 128 --data_dir /floyd/input/nmt_dataset
echo "====================================="
echo "====================================="


echo "DE_EN"


echo "CHO"
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --nlayers 1 --dp 0.0 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 --max_len 15 -b 128 --data_dir /floyd/input/nmt_dataset
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --nlayers 1 --dp 0.25 --reverse_input False --reverse True --model_type c --epochs 50 -v 30000 --max_len 15 -b 128 --data_dir /floyd/input/nmt_dataset
echo "====================================="

python3 run_custom_nmt.py --hs 1000 --emb 500 --nlayers 1 --dp 0.5 --reverse_input False --reverse True --model_type c  --epochs 50 -v 30000 --max_len 15 -b 128 --data_dir /floyd/input/nmt_dataset
echo "====================================="
echo "====================================="
