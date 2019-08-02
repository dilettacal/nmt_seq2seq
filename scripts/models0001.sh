#!/usr/bin/env bash

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn gru --beam 1
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn gru --beam 1

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn lstm --beam 1
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn lstm --beam 1

echo "extend methods"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn gru --beam 5 --tied True
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn lstm --beam 5 --tied True


echo "BI 2"
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn gru --beam 5 --tied True
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn lstm --beam 5 --tied True


echo "optionals"
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn gru --beam 5 --tied True --pretrained True
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn lstm --beam 5 --tied True --pretrained True

#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn gru --beam 5 --tied True --attn dot
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn lstm --beam 5 --tied True --attn dot


#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn gru --beam 5 --tied True --attn additive
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn lstm --beam 5 --tied True --attn additive

echo "Attention and pretrained"
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn gru --beam 5 --tied True --pretrained True --attn dot
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn lstm --beam 5 --tied True --pretrained True --attn dot

#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn gru --beam 5 --tied True --pretrained True --attn additive
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn lstm --beam 5 --tied True --pretrained True --attn additive



echo "##########################"
echo "Standard bi4"
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn gru --beam 5 --tied True
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --bi True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.001 --tok tok --tied False --rnn lstm --beam 5 --tied True