#!/usr/bin/env bash

### create environment
python3 -m venv env
echo "Activating environment"
source "venv/bin/activate"

### Requirements for the project ####
pip install torch torchtext
pip install -U spacy
python3 -m spacy download en
python3 -m spacy download de
python3 -m spacy download xx #multilanguage support
pip install dill
pip install nltk
pip install numpy pandas matplotlib
pip install mock
pip install HTMLParser
