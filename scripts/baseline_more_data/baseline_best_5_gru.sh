#!/usr/bin/env bash

echo "Baseline models with best parameters as baseline search"

#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.002 --tok tok --tied False --rnn gru --beam 5
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.002 --tok tok --tied False --rnn gru --beam 5

##########################  Best 0.0002, dp 0.25, layer 2, BLEU: 14,5 for Beam 5 ##################################################################################################################################################
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn gru --beam 5
##########################  Best 0.0002, dp 0.25, layer 4,  BLEU: 15,88 for Beam 5 ################################################################################################################################################
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0002 --tok tok --tied False --rnn gru --beam 5

#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0001 --tok tok --tied False --rnn gru --beam 5
#python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 170000 --val 1020 --test 1190 --lr 0.0001 --tok tok --tied False --rnn gru --beam 5

##########################  Best 0.0003, dp 0.25, layer 2, BLEU: 14,91 for Beam 5 ##################################################################################################################################################
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 340000 --val 2040 --test 2380 --lr 0.0003 --tok tok --tied False --rnn gru --beam 5
#########################  Best 0.0003, dp 0.25, layer 4, BLEU: 15,5 for Beam 5 ##################################################################################################################################################
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 340000 --val 2040 --test 2380 --lr 0.0003 --tok tok --tied False --rnn gru --beam 5

##########################  Best 0.0003, dp 0.25, layer 2, BLEU: 14,91 for Beam 5 ##################################################################################################################################################
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 510000 --val 3060 --test 3570 --lr 0.0003 --tok tok --tied False --rnn gru --beam 5
#########################  Best 0.0003, dp 0.25, layer 4, BLEU: 15,5 for Beam 5 ##################################################################################################################################################
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 510000 --val 3060 --test 3570 --lr 0.0003 --tok tok --tied False --rnn gru --beam 5



echo "Training on the whole dataset"

##########################  Best 0.0003, dp 0.25, layer 2, BLEU: 14,91 for Beam 5 ##################################################################################################################################################
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 2 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 0 --val 5546 --test 6471 --lr 0.0003 --tok tok --tied False --rnn gru --beam 5
#########################  Best 0.0003, dp 0.25, layer 4, BLEU: 15,5 for Beam 5 ##################################################################################################################################################
python3 run_custom_nmt.py --hs 300 --emb 300 --nlayers 4 --dp 0.25 --reverse_input True --reverse True --model_type s --epochs 80 -v 30000 -b 64 --train 0 --val 5546 --test 6471 --lr 0.0003 --tok tok --tied False --rnn gru --beam 5

