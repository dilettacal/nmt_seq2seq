#!/usr/bin/env bash

echo "Baseline model based on Sutskever"
echo "Fixed params: 300 embedding, hidden, patience:10"
echo "Language combination: German>English, dataset: Europarl"
echo "No pretraiend, no tied"
echo "Test: Dropout values on 2 and 4 layers"

echo "LR 2"

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.002 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.50 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.002 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.0 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.002 --tok tok --tied False --rnn gru

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.002 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.50 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.002 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.0 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.002 --tok tok --tied False --rnn gru


echo "Reducing learning rate to 0.0002"

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.50 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.0 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn gru

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.50 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.0 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn gru



echo "LR 3"

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.003 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.50 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.003 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.0 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.003 --tok tok --tied False --rnn gru

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.003 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.50 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.003 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.0 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.003 --tok tok --tied False --rnn gru


echo "Reducing learning rate to 0.0003"

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0003 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.50 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0003 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.0 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0003 --tok tok --tied False

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0003 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.50 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0003 --tok tok --tied False --rnn gru
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.0 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0003 --tok tok --tied False --rnn gru
