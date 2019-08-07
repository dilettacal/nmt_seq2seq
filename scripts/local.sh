#!/usr/bin/env bash

echo "EN > DE (lstm) - Kleine Stichprobe "
#python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse False --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn dot
python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse False --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot


echo "END ##############################"

echo "OTHER DATASET"
echo "test the model on other dataset attention"
python3 run_custom_nmt.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --bi True --reverse True --epochs 80 --v 30000 --b 64 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot --corpus ""

