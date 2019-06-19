from __future__ import absolute_import

import os

import math
import torch
import numpy as np

from settings import DATA_DIR, DATA_DIR_RAW

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
    from TMX2Corpus import tmx2corpus
    tmx2corpus.convert("")


