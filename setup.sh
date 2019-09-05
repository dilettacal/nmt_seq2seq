#!/usr/bin/env bash

# Uncomment following lines to automatically setup virtual environment
# sudo apt-get install python3-venv
# python3 -m venv env
# source "env/bin/activate"

# Must-Dependencies
pip install torch torchtext
pip install -U spacy
python3 -m spacy download en
python3 -m spacy download de
python3 -m spacy download xx #multilanguage model,
pip install dill
pip install nltk
pip install numpy pandas matplotlib
pip install HTMLParser