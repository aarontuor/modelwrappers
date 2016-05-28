#!/usr/bin/env python2

from __future__ import print_function
import tensorflow as tf
import argparse

import os
os.system("cd antk")

from antk.core import config
from antk.core import generic_model
from antk.core import loader
from antk.models import mfmodel

parser = argparse.ArgumentParser(description="For testing")
parser.add_argument("datadir", metavar="DATA_DIRECTORY", type=str,
                    help="The directory where train, dev, and test data resides. ")
parser.add_argument("config", metavar="CONFIG", type=str,
                    help="The config file for building the ant architecture.")
parser.add_argument("lamb", metavar="LAMB", type=float,
                    help="Lambda hyperparameter")
parser.add_argument("kfactors", metavar="KFACTORS", type=int,
                    help="kfactors hyperparameter")
parser.add_argument("learnrate", metavar="LEARNRATE", type=float,
                    help="learn rate hyperparameter")
parser.add_argument("mbsize", metavar="MBSIZE", type=int,
                    help="minibatch size")
parser.add_argument("irange", metavar="IRANGE", type=float,
                    help="initrange hyperparameter")
parser.add_argument("lossfile", metavar="LOSSFILE", type=str,
                    help="loss file for spearmint")

#if __name__ == '__main__':
args = parser.parse_args()
#data = loader.read_data_sets("/home/hutch_research/skomsks/prep/ydata/out", hashlist=['item', 'user', 'ratings'])
data = loader.read_data_sets(args.datadir, hashlist=['item', 'user', 'ratings'])
data.train.labels['ratings'] = loader.center(data.train.labels['ratings'])
data.dev.labels['ratings'] = loader.center(data.dev.labels['ratings'])
x = mfmodel.mf(data,
               args.config,
               lamb=args.lamb,
               kfactors=args.kfactors,
               verbose=True,
               epochs=100,
               maxbadcount=20,
               mb=args.mbsize,
               initrange=args.irange)
lfile = str(args.lossfile)
out = open(lfile, 'w')
x_err = x._best_dev_error
if x_err > 100 or x_err == float('inf') or x_err == float('nan'):
    x_err = 100
out.write(str(x_err))
out.close()

out = open("/home/hutch_research/workspace/tuora/results/mf_webscope/%f_%d_%f_%d_%f_%f_out.log" % (args.lamb,
                                                                                                args.kfactors,
                                                                                                args.learnrate,
                                                                                                args.mbsize,
                                                                                                args.irange), 'w')
'''
Everytime a wrapper script is run it should print the hyperparameters the
average time per epoch(kept track of by antk), the number of epochs completed
(kept track of by antk) to a file in a subfolder for your particular run below
the results folder. This will be one line printed to a different file for each
experiment of the form

parametername1=value1 parametername2=value2 ..... epochs=num_epochs bestdev=bestdevvalue

'''
out.write("lamb=%f kfactors=%d learnrate=%f mb=%d initrange=%f epochs=%f \
           avspe=%f bestdev=%f" 
           % (args.lamb, args.kfactors, args.learnrate, args.mbsize,
               args.irange, x.epoch_counter, x.average_secs_per_epoch, x_err))
out.close()
