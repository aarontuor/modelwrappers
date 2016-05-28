#!/usr/bin/env python2

from __future__ import print_function
import tensorflow as tf
import argparse
from antk.core import config
from antk.core import generic_model
from antk.core import loader
from antk.models import mfmodel

def return_parser():
    parser = argparse.ArgumentParser(description="For testing")
    parser.add_argument("datadir", metavar="DATA_DIRECTORY", type=str,
                        help="The directory where train, dev, and test data resides. ")
    parser.add_argument("config", metavar="CONFIG", type=str,
                        help="The config file for building the ant architecture.")
    parser.add_argument("initrange", metavar="INITRANGE", type=float,
                        help="A value determining the initial size of the weights.")
    parser.add_argument("kfactors", metavar="KFACTORS", type=int,
                        help="The rank of the low rank factorization.")
    parser.add_argument("lamb", metavar="LAMBDA", type=float,
                        help="The coefficient for l2 regularization")
    parser.add_argument("mb", metavar="MINIBATCH", type=int,
                        help="The size of minibatches for stochastic gradient descent.")
    parser.add_argument("learnrate", metavar="LEARNRATE", type=float,
                        help="The stepsize for gradient descent.")
    parser.add_argument("verbose", metavar="VERBOSE", type=bool,
                        help="Whether or not to print dev evaluations during training.")
    parser.add_argument("maxbadcount", metavar="MAXBADCOUNT", type=int,
                        help="The threshold for early stopping.")
    parser.add_argument("epochs", metavar="EPOCHS", type=int,
                        help="The maximum number of epochs to train for.")
    parser.add_argument("modelID", metavar="MODEL_ID", type=int,
                        help="A unique integer for saving model results during distributed runs model parameters.")
    parser.add_argument("random_seed", metavar="RANDOM_SEED", type=int,
                        help="For reproducible results.")
    parser.add_argument("eval_rate", metavar="EVAL_RATE", type=int,
                        help="How often (in terms of number of data points) to evaluate on dev.")
    parser.add_argument("lossfile", metavar="LOSSFILE", type=str,
                        help="Loss file for spearmint_condor $lossfn argument.")
    parser.add_argument("expname", metavar="EXPNAME", type=str,
                        help="Name of experiment (for resolving results path).")

    return parser
if __name__ == '__main__':

    args = return_parser().parse_args()
    data = loader.read_data_sets(args.datadir, hashlist=['item', 'user', 'ratings'])
    data.train.labels['ratings'] = loader.center(data.train.labels['ratings'])
    data.dev.labels['ratings'] = loader.center(data.dev.labels['ratings'])
    x = mfmodel.mf(data, args.config, initrange=args.initrange,
                        kfactors=args.kfactors,
                        lamb=args.lamb,
                        mb=args.mb,
                        learnrate=args.learnrate,
                        verbose=args.verbose,
                        maxbadcount=args.maxbadcount,
                        epochs=args.epochs,
                        random_seed=args.random_seed,
                        eval_rate=args.eval_rate)
    lfile = str(args.lossfile)
    out = open(lfile, 'w')

    x_err = x._best_dev_error
    if x_err > 100 or x_err == float('inf') or x_err == float('nan'):
        x_err = 100

    out.write(str(x_err))
    out.close()

    #if not(args.expname.endswith('/')):
    #    args.expname += "/"

    out = open("/home/hutch_research/workspace/tuora/results/"+args.expname+"/%f_%d_%f_%d_%f_%d_out.log"
               % (args.initrange,
                  args.kfactors,
                  args.lamb,
                  args.mb,
                  args.learnrate,
                  args.maxbadcount),
               'w')

    out.write("lamb=%f kfactors=%d learnrate=%f mb=%d initrange=%f epochs=%f "
              "avspe=%f bestdev=%f"
              % (args.lamb, args.kfactors, args.learnrate, args.mb,
                 args.initrange, x.completed_epochs, x.average_secs_per_epoch, x_err))
    out.close()
