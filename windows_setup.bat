@echo off
:: Create virtualenv
:: python -m virtualenv env
:: activate environment:
:: ../env/Scripts/activate
:: Install dependencies

:: PyTorch stuff
pip3 install https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl
pip3 install https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-win_amd64.whl
pip install torchtext

:: NLP
pip install -U spacy
python -m spacy download en
python -m spacy download de
python -m spacy download fr
python -m spacy download es
python -m spacy download pt
python -m spacy download it
python -m spacy download xx

pip install -U nltk
python -m nltk.downloader all

:: Other stuff

pip install -U numpy matplotlib pandas seaborn scipy scikit-learn dill

