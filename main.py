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

	exit()

	# define a sequence of 10 words over a vocab of 5 words
	data = torch.Tensor([[0.1, 0.2, 0.3, 0.4, 0.5],
						 [0.5, 0.4, 0.3, 0.2, 0.1],
						 [0.1, 0.2, 0.3, 0.4, 0.5],
						 [0.5, 0.4, 0.3, 0.2, 0.1],
						 [0.1, 0.2, 0.3, 0.4, 0.5],
						 [0.5, 0.4, 0.3, 0.2, 0.1],
						 [0.1, 0.2, 0.3, 0.4, 0.5],
						 [0.5, 0.4, 0.3, 0.2, 0.1],
						 [0.1, 0.2, 0.3, 0.4, 0.5],
						 [0.5, 0.4, 0.3, 0.2, 0.1]])

	# decode sequence
	result = greedy_decoder(data)
	print(result)

	topK = torch.topk(data, 1)[1].squeeze().tolist()
	print(topK)

	print(sorted(beam_search_decoder(data, 4),reverse=True, key=lambda x: x[1]))


