#!/usr/bin/env bash

echo "Saturate Attn"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --epochs 120 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn dot
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --epochs 120 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot

echo "Saturate Vanilla"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --epochs 120 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --epochs 120 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True
