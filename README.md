# Neural Machine Translation - Seq2Seq in PyTorch

This repository contains the code to my Bachelor Thesis about Neural Machine Translation.about

To setup the environment, create a virtual environment, e.g. `python3 -m venv env` and activate it by `source env/bin/activate`.
Then run the Bash script `bash setup.sh` to install all required dependencies.about

To train the model on the Europarl-Dataset, you need to first preprocess the dataset. This is possible by running the script `preprocess.py`.
If you run the script without any arguments, the programm will download the corpus for the German language and preprocess it, by tokenizing it into words and creating the splits. All preprocessed files are stored in `data/preprocessed/europarl/de/splits/30/`

Then you can train the model with the script `train_model.py`. A list of all possible arguments can be displayed by typing: `python train_model.py --help`.
To train the model in the best configuration achieved, run following command:

GRU:
```python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --tied True --rnn gru --beam 5 --attn dot```

LSTM:
```python3 train_model.py --hs 300 --emb 300 --num_layers 2 --dp 0.25 --reverse_input False --bi True --reverse True --epochs 80 --v 30000 --b 64 --train 170000 --val 1020 --test 1190  --lr 0.0002 --tok tok --tied True --rnn lstm --beam 5--attn dot```

To translate from a trained model, use the script `translate.py`. The script accepts following arguments:
1. `--path`: The path to the trained model, e.g. `python train_model.py --path results/de_en/custom/lstm/2/bi/2019-08-11-10-30-31`. This will start the live translation mode.
2. `--file`: Add this argument, if you want to translate from a file. Argument should be a valid path.
3. `--beam`: Add this argument to setup a beam size which is different from 5 (default value)

The beam size can be changed during the live translation mode by typing `#<new_beam_size>`, e.g. `#10`.


If you do not want to use the Europarl dataset, just run the script `train_model.py` by passing an empty string for the argument `--corpus`. This will train the model on the IWSLT-Dataset (Ted Talks) of TorchText.


Enjoy!