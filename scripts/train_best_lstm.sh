#!/usr/bin/env bash

echo "Train best lstm"
echo "Saturate best BASELINE model"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 240 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True

echo "Saturate best model with optional methods"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 240 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --pretrained True --attn additive