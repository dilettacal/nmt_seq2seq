# Neural Machine Translation - Seq2Seq in PyTorch
Repository containing the code to my bachelor thesis about Neural Machine Translation

My Bachelor Thesis is about Neural Machine Translation. Main topics discussed are Encoder-Decoder architectures or Sequence-to-Sequence models in their simplest form. As baseline models the implementations from following papers are proposed:
- I. Sutskever et al. (2014), 
- Y. Cho et al. (2014),

More advanced implementations: t.b.d.

Following frameworks and libraries are used:
- PyTorch, as deep learning framework
- Torchtext, for handling with datasets and during mini-batch training
- Spacy, for tokenization and/or corpus analytics
- Sacremoses for normalization/tokenization purposes
- Sacrebleu to compute the BLEU score
- pandas for handling with `tsv` or `csv` files.

Part of my code is based on existing implementations, which have been extended or modified to my task/dataset and so on.
The main references are following repositories:

- Ben Trevett pytorch-seq2seq repository
- Luke Melas Machine-Translation
- Tensorflow Tutorials, especially the number 21 about Machine Translation and the utility methods to download and prepare the europarl dataset.

Other references are cited in code comments.


## Goals

The primary goal of this Bachelor Thesis is to gain a basic overview about Encoder-Decoder architectures. 
State-of-the-Art systems are based on the newest Google Architecture, the Transformer Model, which is a sophisticated architecture made up by many components, which delivers by the time of writing the best results.

Pre-State-of-the-Art systems adopted an Attention-Mechanism which allowed to focus on the relevant part of the source sentences, and which were taken into consideration during the decoding phase.

The base of this mechanism can be found in the above mentioned papers by Sutskever and Cho. 

## Code references
The code in this repository is mainly based on this work by Luke Melas-Kyriazi: [Machine-Translation](https://github.com/lukemelas/Machine-Translation). The code has been readapted to meet my thesis requirements.

## How to use the program

## Experiments

## Conclusion
