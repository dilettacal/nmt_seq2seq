from __future__ import absolute_import

import os
import time

import math
import torch
import numpy as np
from torchtext.data import Field
from torchtext.datasets import TranslationDataset

from project.utils.io import Seq2SeqDataset
from project.utils.utils import convert
from settings import DATA_DIR, DATA_DIR_RAW, DATA_DIR_PREPRO

SEED = 1234
torch.manual_seed(SEED)
np.random.seed(SEED)



from numpy import array
from numpy import argmax

# greedy decoder
def greedy_decoder(data):
	# index for largest probability each row
	return [torch.argmax(s).item() for s in data]



# beam search
def beam_search_decoder(data, k):
	sequences = [[list(), 1.0]]
	# walk over each step in sequence
	for row in data:
		all_candidates = list()
		# expand each current candidate
		for i in range(len(sequences)):
			seq, score = sequences[i]
			#print("Iterating through sentences for Row {}, step: {},  score: {}, sequence:{}".format(row, i, score, seq ))
			for j in range(len(row)):
				candidate = [seq + [j], score * -torch.log(row[j])]
				all_candidates.append(candidate)
				#print("Iterating through the rows for step {}. Row[j]: {}, Found candidates: {}. \nList of candidates {}".format(i,row[j], candidate, all_candidates))
		# order all candidates by score
		ordered = sorted(all_candidates, key=lambda tup:tup[1])
		# select k best
		sequences = ordered[:k]
	return sequences



if __name__ == '__main__':
    ### test datase
	src_field = Field(sequential=True)
	trg_field = Field(sequential=True, init_token="<s>", eos_token="</s>")
	start = time.time()
	train, val, test = Seq2SeqDataset.splits(path=os.path.join(DATA_DIR_PREPRO, "europarl", "de", "splits"), root="", exts=(".en", ".de"),
											 train="train", validation="val", test="test", fields=(src_field, trg_field), reduce= [500000,100000,10000])
	print("Duration:", convert(time.time()-start))

	### 802919 382342 89213
	print(len(train))
	print(len(val))
	print(len(test))

