#!/usr/bin/env bash

echo "Optional tests"

echo "Best config + 340000 data"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 340000 --val 2040 --test 2380 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot

echo "Best config + full train data"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 0 --val 5546 --test 6471 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot

# 5 % ca 15 minutes/epoch
echo "Best config + 5 %"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 128 --train 170000 --val 8500 --test 8500 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot