#!/usr/bin/env bash

echo "Char based"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse True --bi True --epochs 30 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5 -c True
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse True --bi True --epochs 30 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn gru --beam 5 -c True
