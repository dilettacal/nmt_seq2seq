#!/usr/bin/env bash

echo "Optional tests"
echo "Clip"
echo "Attention"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 128 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot --clip 5.0
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 128 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn dot --clip 5.0
echo "Vanilla"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 128 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn none --clip 5.0
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 128 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn none --clip 5.0

