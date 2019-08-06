#!/usr/bin/env bash

echo "All training data"

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 0 --val 5546 --test 6471 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn dot
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --epochs 80 -v 30000 -b 64 --train 0 --val 5546 --test 6471 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot



echo "test the model on other dataset"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --epochs 80 -v 30000 -b 64 --lr 0.0002 --tok tok --rnn lstm --beam 10 --tied True --attn none --corpus ""

echo "test the model on other dataset attention"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --epochs 80 -v 30000 -b 64 --lr 0.0002 --tok tok --rnn lstm --beam 10 --tied True --attn dot --corpus ""