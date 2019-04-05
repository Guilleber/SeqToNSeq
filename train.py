import argparse
import os
import datetime
import json
import math

import numpy as np
import torch as t
from torch.optim import SGD
from torch.optim import Adam

import batcher
import model
import parameters

kldmax = 0

def kld_fun(epoch, i):
    """
    This function return a coefficient to apply to the KLD term of the loss if using a variational layer. Starting from a very low value, it increases to reach 0.1 after 1 epoch.
    
    Args:
        epoch: the epoch of the training
        i: number of iterations inside the given epoch

    Returns:
        float: KLD term coefficient
    """

    global kldmax
    if epoch != 1:
        return kldmax
    else:
        return ((math.tanh((i-10000)/1000)+1)/2)*kldmax


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RNNAutoEncoder')
    parser.add_argument('--num-epochs', type=int, default=20, metavar='NE', help='number of epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS', help='mini-batch size (default: 64)')
    parser.add_argument('--pen', type=float, default=0.0)
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA', help='use cuda (default: False)')
    parser.add_argument('--learning-rate', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='DR', help='dropout prob (default: 0.3)')
    parser.add_argument('--kldmax', type=float, default=0.00001)
    parser.add_argument('--results-logs-path', default='./results/', metavar='RP', help='folder where to store the results (default: ./results/)')
    parser.add_argument('--models-path', default='./models/', metavar='MP', help='folder where to store the models (default: ./models/)')
    parser.add_argument('--model-name', default='model', metavar="MODEL", help='model name (default: model)')

    args = parser.parse_args()

    batch_loader = batcher.Batcher("./data/")
   
    #Create the dictionary of parameters for the model
    params = parameters.params
    params["vocab_size"] = batch_loader.vocab_size
    params["dropout"] = args.dropout
    if args.pen != 0.0:
        params["pen"] = args.pen
    else:
        params["pen"] = False
    kldmax = args.kldmax

    model = model.RNNAutoEncoder(params)
    if args.use_cuda:
        model = model.cuda()
    model.train()
   
    #Set the learning rates and create the optimizers
    learning_rate = args.learning_rate
    optimizer = SGD(model.parameters(), args.learning_rate)

    #Create the training and testing operators
    train_step = model.trainer(optimizer, batch_loader)
    test = model.tester(batch_loader)

    list_losses_train = []
    #list_mu_train = []
    #list_std_train = []
    #list_r1_train = []
    current_epoch = 1
    iteration = -1

    t.save(model.state_dict(), args.models_path + args.model_name + '_e0.model')

    while current_epoch <= args.num_epochs:
        kld_coef = kld_fun(current_epoch, iteration)
        if iteration%50 == 0:
            loss_train, metrics_train = train_step(args.batch_size, use_cuda=args.use_cuda, full_metrics=True, kld_coef=kld_coef)
            list_losses_train.append(loss_train.data.cpu().numpy()[0])
            #list_mu_train.append(float(metrics_train["mu"]))
            #list_std_train.append(float(metrics_train["std"]))
            #list_r1_train.append(float(metrics_train["r1"]))
            print("epoch " + str(current_epoch) + ", iteration " + str(iteration) + ": loss = " + str(loss_train.data.cpu().numpy()[0]) + ', acc = ' + str(metrics_train["acc"]) + '\n')
        else:
            _, metrics_train = train_step(args.batch_size, use_cuda=args.use_cuda, full_metrics=False, kld_coef=kld_coef)

        iteration += 1
        if metrics_train["epoch"] != current_epoch:
            if metrics_train["epoch"] >= 8:
                learning_rate = learning_rate/2
                optimizer = SGD(model.parameters(), learning_rate)
                train_step = model.trainer(optimizer, batch_loader)           

            current_epoch = metrics_train["epoch"]
            iteration = 0
            #print("------VALIDATION------\n")
            #_, metrics_valid = test(args.batch_size, use_cuda=args.use_cuda, max_decoding_loop=20, full_metrics=True, test_set='dev', full_batch=False)
            #for k in metrics_valid:
            #    print(k + ": " + str(metrics_valid[k]) + '\n')
            #print("----------------------\n")

            t.save(model.state_dict(), args.models_path + args.model_name + "_e" + str(current_epoch - 1) + ".model")

        with open(args.results_logs_path + args.model_name + "_e" + str(current_epoch - 1) + ".loss", 'w') as f:
            json.dump([float(i) for i in list_losses_train], f)
            f.close()

            #with open(args.results_logs_path + args.model_name + "_e" + str(current_epoch - 1) + ".r1", 'w') as f:
                #json.dump(list_r1_train, f)
                #f.close()
