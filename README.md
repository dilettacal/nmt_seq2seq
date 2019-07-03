# Neural Machine Translation - Seq2Seq in PyTorch
(**WORK IN PROGRESS**)
Repository containing the code to my bachelor thesis about Neural Machine Translation

My Bachelor Thesis is about Neural Machine Translation. 

Main topics discussed are Encoder-Decoder architectures or Sequence-to-Sequence models in their simplest form. 


##### TODO: model description


Following frameworks and libraries are used:
- PyTorch: 1.1.0
- torchtext: 0.3.1 (Data handling, vectorization, vocabularies)
- spacy: tokenization, preprocessing
    - Models required: German, English. Others optionally
- tmx2corpus: A library to convert tmx files to raw text files
- nltk: `corpus_bleu` function is used to compute the corpus bleu
- numpy, pandas, matplotlib


```bash
### Requirements for the project ####
pip install --user torch torchtext
pip install --user -U spacy
python -m spacy download en
python -m spacy download de
pip install nltk
pip install numpy pandas matplotlib

### Required to preprocess the tmx file ####
pip install git+https://github.com/amake/tmx2corpus.git

```
Or see the `setup.sh` script or the `requirements.txt` file.


## Goals

TODO: bla bla bla

## Code references
**TODO**: Extend this part !!!!!!
## How to use the program

In this program the Europarl dataset has been used. This dataset is available on the OPUS platform: http://opus.nlpl.eu/Europarl.php in different formats, either in `tmx` or in `txt` format.
Download the files from the matrix in the section "Statistics and TMX/Moses Downloads". 

For more information about Opus: http://www.lrec-conf.org/proceedings/lrec2012/pdf/463_Paper.pdf

The Europarl corpus is described here: http://www.statmt.org/europarl/

The `tmx` format is a very common format in the translation industry, as it acts as a database for translations, as it stores for each translation unit (`<tu>`) two segments, one for the source language and the second for the target language.
Segments stored in this xml files are unique, meaning that there should not be any repeated translations. The format should also assure that sentences are really aligned.

The `tmx` format is available at: http://opus.nlpl.eu/download.php?f=Europarl/v7/tmx/de-en.tmx.gz.

The `txt` format is available at: http://opus.nlpl.eu/download.php?f=Europarl/v7/moses/de-en.txt.zip (The zip file contains two separate files for the source (`.en`) and for the target language (e.g. `.de`)).

To preprocess the tmx corpus, you need to install the tmx2corpus library with `pip install git+https://github.com/amake/tmx2corpus.git`. The preprocessing is very easy and only takes few seconds.

Extract the file in a directory, e.g. "data" (default: "data/raw/europarl/de") and run `preprocess.py`, passing the arguments as described in the python script.

The model also runs with the Torchtext IWSLT corpus. Just pass `--corpus ""` as an argument. 
The IWSLT files are not tokenized, but used as they are. Torchtext will only split words based on white spaces.

Example:
```bash
python run_custom_nmt.py --corpus "" --train 170000 --val 1000 --test 1000 --nlayers 4 --bi True
```

## Experiments

## Conclusion
