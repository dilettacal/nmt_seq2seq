#!/usr/bin/env bash

echo "Optional tests"

echo "Best config + 340000 data"
python3 train_model.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn gru --beam 5 --attn dot
python3 train_model.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5--attn dot

echo "Best config + full train data"
python3 train_model.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn gru --beam 5 --attn dot
python3 train_model.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5--attn dot

echo "Best config + other valid and test"
python3 train_model.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn gru --beam 5 --attn dot
python3 train_model.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5--attn dot

echo "Best config + other langauge comb"
python3 train_model.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn gru --beam 5 --attn dot
python3 train_model.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5--attn dot