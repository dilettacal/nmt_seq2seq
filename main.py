from project import get_full_path
from project.utils.preprocessing import SpacyTokenizer, FastTokenizer
import os
import spacy
import time

from project.utils.utils import Logger, convert
from settings import DATA_DIR

if __name__ == '__main__':
	data = get_full_path(DATA_DIR, "preprocessed", "europarl", "de")

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

	print(convert(time.time()-start))
	import numpy as np
	#np_src = np.array(src_lines)
	#np_docs = np.array(testing_doc)
	assert len(src_lines)==len(testing_doc)

	#print((np_src == np_docs).all())