#!/usr/bin/env bash

echo "0.0002 - 4 Layers - Weight Tying - Beam 5 - Bidirectional - 4 Layers (decoder) - Attention: dot"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --tied True --rnn gru --beam 5 --attn dot
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5--attn dot


echo "###############################################"
echo "0.0002 - 4 Layers - Weight Tying - Beam 5"
python3 train_model.py --hs 300 --emb 300 --num_layers 4 --dp 0.25 --reverse_input True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --tied True --rnn gru --beam 5 --attn none
python3 train_model.py --hs 300 --emb 300 --num_layers 4 --dp 0.25 --reverse_input True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5 --attn none

echo "0.0002 - 4 Layers - Weight Tying - Beam 5 - Bidirectional - 4 Layers (decoder)"
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --tied True --rnn gru --beam 5 --attn none
python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5 --attn none