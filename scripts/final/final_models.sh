#!/usr/bin/env bash

# Here only the best models are retrained , also in the Language direciton EN > DE

#echo "Best baseline (Rev)"
# Best baseline (no bienc)
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --epochs 180 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --epochs 180 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True

# Best baseline (BiENc)
echo "Best baseline BiEnc"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 180 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 180 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True


echo "Best BiEnc pretrained"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 180 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --pretrained True
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 180 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --pretrained True


# Best Attention model
echo "Best Attention model"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 180 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn dot
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 180 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot

echo "Best Attention pretrained"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 180 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn gru --beam 5 --tied True --attn dot --pretrained True
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --epochs 180 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --rnn lstm --beam 5 --tied True --attn dot --pretrained True