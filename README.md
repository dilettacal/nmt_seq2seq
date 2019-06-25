# Neural Machine Translation - Seq2Seq in PyTorch
(**WORK IN PROGRESS**)
Repository containing the code to my bachelor thesis about Neural Machine Translation

My Bachelor Thesis is about Neural Machine Translation. 

Main topics discussed are Encoder-Decoder architectures or Sequence-to-Sequence models in their simplest form. 

As baseline models the implementations from following papers are proposed:
- I. Sutskever et al. (2014), 
- Y. Cho et al. (2014),


Following frameworks and libraries are used:
- PyTorch: 1.1.0
- torchtext: 0.3.1 (Data handling, vectorization, vocabularies)
- spacy: tokenization, preprocessing
    - Models required: German, English. Others optionally
- tmx2corpus: A library to convert tmx files to raw text files
- nltk: `corpus_bleu` function is used to compute the corpus bleu
- Datascience stack: numpy, pandas, matplotlib

For other requirements, please see the `requirements.txt` file.


## Goals

The primary goal of this Bachelor Thesis is to gain a basic overview about Encoder-Decoder architectures. 
State-of-the-Art systems are based on the newest Google Architecture, the Transformer Model, which is a sophisticated architecture made up by many components, which delivers by the time of writing the best results.

Pre-State-of-the-Art systems adopted an Attention-Mechanism which allowed to focus on the relevant part of the source sentences, and which were taken into consideration during the decoding phase.

The base of this mechanism can be found in the above mentioned papers by Sutskever and Cho. 

## Code references
**TODO**: Extend this part !!!!!!

The code in this repository is partially based on this work by Luke Melas-Kyriazi: [Machine-Translation](https://lukemelas.github.io/machine-translation.html). 

## How to use the program

## Experiments

## Conclusion
