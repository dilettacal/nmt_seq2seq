#!/usr/bin/env bash

echo "Baseline models with methods"
##########################  Best 0.0002, dp 0.25, layer 4,  BLEU: 15,88 for Beam 5 ################################################################################################################################################
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True

echo "Saturate best model"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 240 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True

echo "Optional experiments"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --pretrained True
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn dot
### local ###
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn additive
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --pretrained True --attn dot
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --pretrained True --attn additive
### local ###

echo "Saturate optional model"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 240 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --pretrained True --attn additive