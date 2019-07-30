@echo off
:: Create virtualenv
:: Type these commands before running this script!
:: Install a new virtual environment
:: python -m virtualenv env
:: Then activate environment:
:: . ./env/Scripts/activate
:: Install dependencies

:: PyTorch stuff - Uncomment the right version
:: Cuda Version
:: pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl
:: CPU version
:: pip3 install https://download.pytorch.org/whl/cpu/torch-1.1.0-cp36-cp36m-linux_x86_64.whl
pip install torchtext

:: spaCy
pip install -U spacy
:: Standard training requires englis
python -m spacy download en
:: My model is trained german > english
python -m spacy download de
:: This is the multi-language model
python -m spacy download xx

:: Other models :::
:: python -m spacy download fr
:: python -m spacy download es
:: python -m spacy download pt
:: python -m spacy download it

:: Needed for bleu_score
pip install -U nltk
:: Uncomment if needed
:: python -m nltk.downloader all

:: Other stuff
:: dill is needed to serialize objects
pip install -U dill
pip install -U numpy matplotlib pandas seaborn scipy scikit-learn

