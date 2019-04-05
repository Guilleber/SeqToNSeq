import argparse
import os
import datetime
import json

import numpy as np
import torch as t
from torch.optim import SGD

import batcher
import model
import parameters

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RNNAutoEncoder')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA', help='use cuda (default: False)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR', help='dropout prob (default: 0.3)')
    parser.add_argument('--results-logs-path', default='./results/', metavar='RP', help='folder where to store the results (default: ./results/)')
    parser.add_argument('--input-file', metavar='IF', help='input file')
    parser.add_argument('--output-file', metavar='OF', help='output file')
    parser.add_argument('--model', metavar="MODEL", help='model file')
    parser.add_argument('--decoder', metavar="DEC", default=0, type=int, help='decoder to use for translation')

    args = parser.parse_args()

    batch_loader = batcher.Batcher("./data/")
    
    params = parameters.params
    params["vocab_size"] = batch_loader.vocab_size
    params["dropout"] = args.dropout

    model = model.RNNAutoEncoder(params)
    model.load_state_dict(t.load(args.model))
    if args.use_cuda:
        model.cuda()
    model.train()

    translate = model.translator(batch_loader)
    input_file = open(args.input_file, 'r')
    output_file = open(args.output_file, 'w')
    translate(input_file, output_file, use_cuda=args.use_cuda, max_decoding_loop=20, dec=args.decoder)
    input_file.close()
    output_file.close()
