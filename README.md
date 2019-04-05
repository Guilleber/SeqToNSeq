# Seq-to-NSeq model for multi-summary generation

This repository contains the source code for the paper "Seq-to-NSeq model for multi-summary generation".
Our model is based on a sequence-to-sequence with multiple decoders that learns to identify and reproduce multiple styles of summaries.
Our results show that this model can generate wide differences between the summaries.
Also, the styles of the decoders are consistent (e.g., the active/passive form of some summaries)

The "results" folder contains output examples from our model on Gigaword test set for the baseline (seq-to-seq), 2 decoders (seq-to-2seq) and 3 decoders (seq-to-3seq)

## Setup

The scripts run with python 3.5 and pytorch 0.3.1. Other versions may not work.

You also need the following libraries:
- Numpy
- Argparse
- Json

Ensure there exist a "data" folder inside the project folder. This folder should contain at least 2 files:
- train.src: the input sentences for training
- train.trg: the gold summaries for these sentences

Gigaword data for sentence summarization task can be found at https://github.com/harvardnlp/sent-summary

When you run the script for the first time, two additional json files are created.
They contain the words vocabulary, and you need to delete them manually if you change the training set.

You must also create a "models" folder at the same location.
This folder is used by "train.py" to store the model files at each epoch.

## Training

"parameters.py" is a configuration file containing hyperparameters for the model:
- "embedding\_size controls" the the dimension of embeddings vectors (default: 500).
- "encoder\_hidden\_size" controls" the encoder LSTM's hidden vector dimension (default: 250).
- "decoder\_hidden\_size" controls the decoder LSTM's hidden vector dimension (default: 500).
- "num\_encoder\_layers" controls the number of layers of the encoder (default: 2).
- "num\_decoder\_layers" controls the number of layers of the decoder (default: 2).
- "VAE" adds an additional variational layer in between encoder and decoder (default: False).
- "nb\_decoders" controls the number of decoders and thus the number of generated summaries at test time (default: 2).
Set this hyperparameter to 1 for the baseline.

By default, the parameters are set to be the ones used in the paper.

To train the model, run the following command:
```shell
python3 train.py --model-name <model name> --num-epochs 15 --use-cuda
```
Addtional optional parameters for "train.py" contain:
- "--pen": \alpha parameter to apply for "PEN" models (default: 0.0)
- "--batch-size": minibatch size (default: 64)
- "--learning-rate": initial learning rate (default: 1.0)

This script creates one model file "MyModel_e#.model" per epoch containing all model parameters.

## Generate summaries

To generate the summaries for a given set of input sentences, run the following commands:
```shell
python3 translate.py --model ./models/<model name>_e15.model --input-file <path to the sentences to summarize> --output-file <file to store the summaries> --decoder 0 --use-cuda
python3 translate.py --model ./models/<model name>_e15.model --input-file <path to the sentences to summarize> --output-file <file to store the summaries> --decoder 1 --use-cuda
```
The option "--decoder" specify which decoder to use. This parameter should be an integer in [0, nb_decoder[

If you want to evaluate our model using some ROUGE evaluation script, pls ensure to replace all "\<unk\>" in the output sentences by "UNK" due to some weird interactions between "\<unk\>" and the XML/HTML parser used in the original ROUGE evaluation script
