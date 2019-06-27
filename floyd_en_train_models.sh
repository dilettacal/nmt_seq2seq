#!/usr/bin/env bash

echo "Starting script.........."

pip install dill unidecode nltk sacremoses spacy
pip install git+https://github.com/amake/tmx2corpus.git

echo "EN_DE"

echo "Sutskever"

echo "====================================="

python3 run_custom_nmt.py --hs 500 --emb 500 --nlayers 4 --dp 0.0 --reverse_input True --model_type s --reverse False --epochs 50 -v 30000 --max_len 15 --data_dir /floyd/input/nmt_data

echo "====================================="

python3 run_custom_nmt.py --hs 500 --emb 500 --nlayers 4 --dp 0.25 --reverse_input True --model_type s --reverse False --epochs 50 -v 30000 --max_len 15 --data_dir /floyd/input/nmt_data

echo "====================================="

python3 run_custom_nmt.py --hs 500 --emb 500 --nlayers 4 --dp 0.5 --reverse_input True --model_type s --reverse False --epochs 50 -v 30000 --max_len 15 --data_dir /floyd/input/nmt_data


echo "====================================="
echo "====================================="



echo "CUSTOM"

echo "Bidirectional"

echo "====================================="

python3 run_custom_nmt.py --hs 500 --emb 500 --nlayers 4 --dp 0.0 --bi True --model_type custom --reverse False --epochs 50 -v 30000 --max_len 15 --data_dir /floyd/input/nmt_data

echo "====================================="

python3 run_custom_nmt.py --hs 500 --emb 500 --nlayers 4 --dp 0.25 --bi True --model_type custom --reverse False --epochs 50 -v 30000 --max_len 15 --data_dir /floyd/input/nmt_data
echo "====================================="

python3 run_custom_nmt.py --hs 500 --emb 500 --nlayers 4 --dp 0.5 --bi True --model_type custom --reverse False --epochs 50 -v 30000 --max_len 15 --data_dir /floyd/input/nmt_data

echo "====================================="
echo "====================================="

