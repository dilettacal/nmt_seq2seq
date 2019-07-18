from project import get_full_path
from project.utils.preprocessing import SpacyTokenizer, FastTokenizer
import os
import spacy
import time

from project.utils.utils import Logger, convert_time_unit
from settings import DATA_DIR

if __name__ == '__main__':
	data = get_full_path(DATA_DIR, "preprocessed", "europarl", "de")

	src_lines_tok = [line.strip("\n") for line in
				 open(os.path.join(data, "bitext.tok.en"), mode="r",
					  encoding="utf-8").readlines() if line]
	trg_lines_tok = [line.strip("\n") for line in
				 open(os.path.join(data, "bitext.tok.de"), mode="r",
					  encoding="utf-8").readlines() if line]

	print(len(src_lines_tok))
	print(len(trg_lines_tok))
	for i, (s, t) in zip(src_lines_tok, trg_lines_tok):
		len_s = len(s.split(" "))
		len_t = len(t.split(" "))
		if s == "" or t == "":
			print("Sentence is empty")
			print(s, t)
		if len_s < 2 or len_t < 2:
			print("Sentence is short")
			print(s, t)

	exit()
	src_lines = [line.strip("\n") for line in
				 open(os.path.join(data, "bitext.en"), mode="r",
					  encoding="utf-8").readlines() if line]
	trg_lines = [line.strip("\n") for line in
				 open(os.path.join(data, "bitext.de"), mode="r",
					  encoding="utf-8").readlines() if line]


	print(len(src_lines), len(trg_lines))

	en_nlp = spacy.load("en")
	de_nlp = spacy.load("de")



	logger = Logger(".", file_name="test_nlp.log")

	testing_doc = []
	start = time.time()
	with en_nlp.disable_pipes('tagger', 'parser', "ner"):
		for i, doc in enumerate(en_nlp.pipe(src_lines, batch_size=1000)):
			testing_doc.append(doc)
			if i % 10 == 0:
				logger.log("Sequence: {}".format(i))
				logger.log("DOC: {}".format([tok.text for tok in doc]))
				logger.log("ORG: {}".format(src_lines[i]))
				logger.log("*"*100)

	print(convert_time_unit(time.time() - start))
	import numpy as np
	#np_src = np.array(src_lines)
	#np_docs = np.array(testing_doc)
	assert len(src_lines)==len(testing_doc)

	#print((np_src == np_docs).all())