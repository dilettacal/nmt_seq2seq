#!/usr/bin/env bash

echo "Baseline LSTM"

echo "###############################################"
echo "0.002"
python3 train_model.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.002 --tok tok --tied False --rnn lstm --beam 1 --attn none

echo "###############################################"
echo "0.0001"
python3 train_model.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0001 --tok tok --tied False --rnn lstm --beam 1 --attn none

echo "###############################################"
echo "0.0002"
#python3 train_model.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn lstm --beam 1 --attn none


echo "###############################################"
echo "0.0002 - 4 Layers (Verfeinern)"
python3 train_model.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn lstm --beam 1 --attn none