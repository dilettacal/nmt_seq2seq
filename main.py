from __future__ import absolute_import

import os
import time

import math
import torch
import numpy as np
from torchtext import data
from torchtext.data import Field
from torchtext.datasets import TranslationDataset

from project.utils.io import Seq2SeqDataset, SrcField, TrgField
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
	SRC = SrcField()
	TRG = TrgField()
	start = time.time()
	train, val, test = Seq2SeqDataset.splits(path=os.path.join(DATA_DIR_PREPRO, "europarl", "de", "splits"), root="", exts=(".en", ".de"),
											 train="train", validation="val", test="test", fields=(SRC, TRG), reduce= [500000,100000,10000])
	print("Duration:", convert(time.time()-start))

	### 802919 382342 89213
	print(len(train))
	print(len(val))
	print(len(test))

	SRC.build_vocab(train, val, max_size=30000, min_freq=2)
	TRG.build_vocab(train, val, max_size=50000,  min_freq=2)
	# 50981 EN
#	# 146288 DE
	print(len(SRC.vocab))
	print(len(TRG.vocab))

	# Create iterators to process text in batches of approx. the same length
	train_iter = data.BucketIterator(train, batch_size=10, device="cuda", repeat=False,
									 sort_key=lambda x: (len(x.src), len(x.trg)), sort_within_batch=True, shuffle=True)
	val_iter = data.Iterator(val, batch_size=1, device="cuda", repeat=False, sort_key=lambda x: len(x.src))
	test_iter = data.Iterator(test, batch_size=1, device="cuda", repeat=False, sort_key=lambda x: len(x.src))


	first_train_batch = next(iter(train_iter))
	first_val_batch = next(iter(val_iter))
	first_test_batch = next(iter(test_iter))

	print("Train batch...")
	srcs = [' '.join([SRC.vocab.itos[i] for i in sent]) for sent in first_train_batch.src.t()]
	trgs = [' '.join([TRG.vocab.itos[i] for i in sent]) for sent in first_train_batch.trg.t()]

	print("First train batch:")
	all_togheter = list(zip(srcs, trgs))
	for elem in all_togheter:
		print(elem)

	print(SRC.reverse(first_train_batch.src))
	print(TRG.reverse(first_train_batch.trg))

	print("Val batch...")
	srcs = [' '.join([SRC.vocab.itos[i] for i in sent]) for sent in first_val_batch.src.t()]
	trgs = [' '.join([TRG.vocab.itos[i] for i in sent]) for sent in first_val_batch.trg.t()]

	print("First val batch:")
	all_togheter = list(zip(srcs, trgs))
	for elem in all_togheter:
		print(elem)







