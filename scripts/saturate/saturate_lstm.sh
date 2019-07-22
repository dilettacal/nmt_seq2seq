#!/usr/bin/env bash

#### 2 layers #####

### baseline model

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 160 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn lstm --beam 1
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 160 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0001 --tok tok --tied False --rnn lstm --beam 1


#### baseline + 5

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 160 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn lstm --beam 5
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 160 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0001 --tok tok --tied False --rnn lstm --beam 5


### baseline + 5 + tied

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 160 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 160 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0001 --tok tok --tied True --rnn lstm --beam 5

#### 4 layers #####

### baseline model

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 160 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn lstm --beam 1
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 160 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0001 --tok tok --tied False --rnn lstm --beam 1


#### baseline + 5

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 160 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn lstm --beam 5
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 160 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0001 --tok tok --tied False --rnn lstm --beam 5


### baseline + 5 + tied

python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 160 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 160 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0001 --tok tok --tied True --rnn lstm --beam 5
