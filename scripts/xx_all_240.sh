#!/usr/bin/env bash

#  sudo screen -dm -L all_data ./scripts/xx_all_240.sh

echo "Training best attention models with 240 epochs"
echo "LSTM DE>EN"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 240 --v 30000 --b 64 --train 0 --val 0 --test 0  --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5 --attn dot
echo "LSTM EN > DE"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --reverse_input False --bi True --reverse False --epochs 240 --v 30000 --b 64 --train 0 --val 0 --test 0  --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5 --attn dot

echo "GRU DE>EN"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 240 --v 30000 --b 64 --train 0 --val 0 --test 0  --lr 0.0002 --tok tok --tied True --rnn gru --beam 5 --attn dot
echo "GRU EN>DE"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --reverse_input False --bi True --reverse False --epochs 240 --v 30000 --b 64 --train 0 --val 0 --test 0  --lr 0.0002 --tok tok --tied True --rnn gru --beam 5 --attn dot
