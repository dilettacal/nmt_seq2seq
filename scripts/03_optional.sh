#!/usr/bin/env bash

echo "Optional tests"

echo "Best config + full train data"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 0 --val 5546 --test 6471 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot

echo "Best config + 5 %"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 128 --train 170000 --val 8500 --test 8500 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot

echo "EN > DE"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse False --epochs 80 --v 30000 --b 128 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot