#!/usr/bin/env bash

### Requirements for the project ####
pip install --user torch torchtext
pip install --user -U spacy
python -m spacy download en
python -m spacy download de
pip install nltk
pip install numpy pandas matplotlib

### Required to preprocess the tmx file ####
pip install git+https://github.com/amake/tmx2corpus.git
