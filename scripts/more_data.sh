#!/usr/bin/env bash

echo "more data"
# ca. 10 minutes/epoch
#python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 1 --v 30000 --b 128 --train 170000 --val 5000 --test 5000 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn dot
#python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 1 --v 30000 --b 128 --train 170000 --val 5000 --test 5000 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot

# 5 % ca 15 minutes/epoch
python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 128 --train 170000 --val 8500 --test 8500 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn dot
python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 128 --train 170000 --val 8500 --test 8500 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot

# 10 %
#python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 1 --v 30000 --b 128 --train 170000 --val 17000 --test 17000 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn dot
#python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 1 --v 30000 --b 128 --train 170000 --val 17000 --test 17000 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot
