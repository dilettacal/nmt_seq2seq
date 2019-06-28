import torch
import numpy as np

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

	words = dict({0: 'the', 1:'big', 2:'house', 3:'red', 4:'yellow'})

	data = [[0.1, 0.2, 0.3, 0.4, 0.5],
			[0.5, 0.4, 0.3, 0.2, 0.1],
			[0.1, 0.2, 0.3, 0.4, 0.5],
			[0.5, 0.4, 0.3, 0.2, 0.1],
			[0.1, 0.2, 0.3, 0.4, 0.5],
			[0.5, 0.4, 0.3, 0.2, 0.1],
			[0.1, 0.2, 0.3, 0.4, 0.5],
			[0.5, 0.4, 0.3, 0.2, 0.1],
			[0.1, 0.2, 0.3, 0.4, 0.5],
			[0.5, 0.4, 0.3, 0.2, 0.1]]
	data = torch.tensor(data)
	# decode sequence
	result = beam_search_decoder(data, 3)
	# print result
	for seq in result:
		print(seq)
		sentence = seq[0]
		sentence = [words.get(idx, "unk") for idx in sentence]
		print(sentence)


"""
[[4, 0, 4, 0, 4, 0, 4, 0, 4, 0], 0.025600863289563108]
[[4, 0, 4, 0, 4, 0, 4, 0, 4, 1], 0.03384250043584397]
[[4, 0, 4, 0, 4, 0, 4, 0, 3, 0], 0.03384250043584397]


My:
[[4, 0, 4, 0, 4, 0, 4, 0, 4, 0], tensor(0.0256)]
[[4, 0, 4, 0, 4, 0, 4, 0, 4, 1], tensor(0.0338)]
[[4, 0, 4, 0, 4, 0, 4, 0, 3, 0], tensor(0.0338)]
"""


