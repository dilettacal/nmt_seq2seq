#!/usr/bin/env bash


#### baseline + 5

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 250 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn gru --beam 5
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 120 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0003 --tok tok --tied False --rnn gru --beam 5


### baseline + 5 + tied

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 250 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn gru --beam 5
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 120 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0003 --tok tok --tied True --rnn gru --beam 5


### enc bi

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse True --bi True --epochs 250 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn gru --beam 5
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse True --bi True --epochs 120 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0003 --tok tok --tied True --rnn gru --beam 5