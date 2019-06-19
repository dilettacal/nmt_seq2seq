#!/bin/bash
# run: sudo bash create_env.sh env_name
# This script creates the environment: <project_dir>/env_name

if [ $# -eq 0 ]
  then
    echo "Provide environment name as argument!"
    exit 1
fi

project_dir=$(pwd)
echo "Project directory directory: $project_dir"

if [ $2 = "home"  ]
  then
    cd $HOME
    echo "Installing the environment in $(pwd) directory"
fi

### activate environment with: source path_to_env/env_name/bin/activate

echo "You provided environment name:" "$1"

env_name=$1

echo "Installing virtualenv if not installed"

sudo -H pip3 install --user virtualenv


echo "Creating environment"

sudo -H python3 -m venv $env_name

#sudo chmod -R o+rwx /$env_name

sudo -H $env_name/bin/python -m pip install --upgrade pip

echo "Installing deep learning frameworks"

sudo -H $env_name/bin/pip install torch torchtext

# if tensorflow and keras should be installed as well
#sudo -H $env_name/bin/pip install tensorflow keras

echo "Installing data science libraries"

sudo -H $env_name/bin/pip install scipy numpy matplotlib pandas

echo "Installing sacremoses and nltk libraries"

sudo -H $env_name/bin/pip install sacremoses nltk

# Optional nlp installations
# sudo -H $env_name/bin/pip install textblob textblob_de

# comment lines if you do not want to install a certain spacy model
# English and German models are used within the project!

echo "Installing spacy..."

sudo -H $env_name/bin/pip install -U spacy

# mandatory
sudo -H $env_name/bin/python -m spacy download en;
sudo -H $env_name/bin/python -m spacy download de;

#optional
sudo -H $env_name/bin/python -m spacy download fr;
sudo -H $env_name/bin/python -m spacy download it;
sudo -H $env_name/bin/python -m spacy download xx;

#echo "Installing library tmx2corpus from git..."
#sudo -H $env_name/bin/pip install git+https://github.com/amake/tmx2corpus.git

echo "Installing other requirements"

sudo -H $env_name/bin/pip install pipreqs atlas unidecode

### testing environment

echo "Testing some imports"
echo ""

$env_name/bin/python -c "import torch; import torchtext; print(torch.cuda.is_available()); import tmx2corpus; import numpy as np; import pandas as pd; import matplotlib.pyplot as plt; import nltk; from sacremoses import MosesTokenizer, MosesDetokenizer; import spacy; spacy.load('en_core_web_sm'); spacy.load('de_core_news_sm'); from nltk.tokenize import sent_tokenize, word_tokenize; print(sent_tokenize('Today I am happy. I do not know why')); print(word_tokenize('How are you?')); nlp = spacy.load('en_core_web_sm'); print([tok.text for tok in nlp('How are you?')]); import numpy as np; print(np.random.randint(2, size=10));"

echo "Installing requirements from file"

sudo -H $env_name/bin/pip install -r $project_dir/requirements.txt




