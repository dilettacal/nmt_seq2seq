#!/usr/bin/env bash

echo "Best from baseline search"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 1
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 1

echo "Baseline models with weigth tying"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True

echo "Bidirectional 2 layers (final best vanilla model)"

python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True
python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True

echo "Bidirectional 2 layers + Attention dot (final best model)"
python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn dot
python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot
