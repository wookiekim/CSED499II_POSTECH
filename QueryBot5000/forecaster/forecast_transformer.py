#!/usr/bin/env python3.6

import importlib
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init
import numpy as np
import math
import time
import os
import csv
import argparse
import matplotlib.pyplot as plt
import scipy as sp
import scipy.io
import shutil
import pickle

import statsmodels.api as sm

from datetime import datetime, timedelta

from sortedcontainers import SortedDict

import Utilities

from spectral import Two_Stage_Regression

from models import TRANSFORMER_Model

# ==============================================
# PROJECT CONFIGURATIONS
# ==============================================

PROJECTS = {
    "tiramisu": {
        "name": "tiramisu",
        "input_dir": "../data/online-tiramisu-clusters",
        "cluster_path": "../data/cluster-coverage/tiramisu-coverage.pickle",
        "output_dir":
        #"../result-interval-new/online-prediction/online-tiramisu-clusters-prediction",
        "../result/online-prediction/online-tiramisu-clusters-prediction",
    },
    "to_be_added": {
        
    }
}

# =========================================================================
# Batch class: Object for holding a batch of data with mask during training
# >> Holds the src and target sentences for trainiing
# >> and constructs the masks
# =========================================================================
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()
    
    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

# ==========================================================================
# run_epoch
# > generic training and scoring function to keep track of loss
# > pass generic loss compute function to handle parameter updates
# ==========================================================================

def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, 
                            batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                    (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens

# ===============================================================
# Class NoamOpt
# Adam optimizer 
# ===============================================================
class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(model):
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
            torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

# ==================================================
# Args class: Object to hold parameters for learning
# ==================================================
class Args:
    def __init__(self):
        # Learning rate
        self.lr = 1
        self.clip = 0.25
        # Adjust BPTT size accordingly with interval size
        self.bptt = {1: 240, 5:200, 10:120, 20:90, 30:60, 60:48, 120:30}
        self.dropout = 0.2
        self.tied = False
        self.cuda = False
        self.log_interval = 50
        self.save = 'model.pt'
        self.nRFF = 25

# =======================================================================
# LoadData 
# =======================================================================
def LoadData(file_path, aggregate):
    trajs = dict()

    datetime_format = "%Y-%m-%d %H:%M:%S" # Strip milliseconds ".%f"
    for csv_file in sorted(os.listdir(file_path)):
        print(csv_file)

        cluster = int(os.path.splitext(csv_file)[0])
        trajs[cluster] = SortedDict()

        with open(file_path + "/" + csv_file, 'r') as f:
            reader = csv.reader(f)

            traj = list()
            date = list()

            for line in reader:
                count = float(line[1])
                ts = datetime.strptime(line[0], datetime_format)
                hour = ts.hour
                if aggregate > 60:
                    hour //= aggregate // 60
                time_stamp = datetime(ts.year, ts.month, ts.day, hour, ts.minute - (ts.minute %
                    aggregate), 0)

                if not time_stamp in trajs[cluster]:
                    trajs[cluster][time_stamp] = 0

                trajs[cluster][time_stamp] += count

    return trajs

# =========================================
# GeneratePair 
# =========================================
def GeneratePair(data, horizon, input_dim):
    n = data.shape[0]
    m = data.shape[1]

    x = []
    y = []

    for i in range(n - horizon - input_dim + 1):
        x.append(data[i:i + input_dim].flatten())
        y.append(data[i + input_dim + horizon - 1])

    return (np.array(x), np.array(y))

# ===============
# GetMatrix
# ===============
def GetMatrix(x):
    xx = x.T.dot(x)
    xx += np.identity(xx.shape[0])
    return np.linalg.inv(xx).dot(x.T)

# =================
# Training 
# =================
def Training(x, y):
    params = []
    for j in range(y.shape[1]):
        params.append(x.dot(y[:, j]))

    return params
# =============================================================================
# Testing 
# =============================================================================
def Testing(params, x):
    y_hat = None

    for j in range(len(params)):
        y = x.dot(params[j])

        y = y.reshape((-1, 1))

        if y_hat is None:
            y_hat = y
        else:
            y_hat = np.concatenate((y_hat, y), axis = 1)

    return y_hat

# ==============================================================================
# train_pass
# ==============================================================================
def train_pass(args, method, model, data, criterion, lr, bptt, clip, log_interval):

    if method == 'transformer':
        pass
# ==============================================================================
# evalute_pass 
# ==============================================================================
def evaluate_pass(args, method, model, data, criterion, bptt):

    if method == 'transformer':    
        return (total_loss / (batch+1)), y, y_hat
# ==============================================================================
# GetModel 
# ==============================================================================
def GetModel(args, data, method):
    train = batchify(data, args.batch_size)

    ntokens = train.shape[2]
    bsz = train.shape[1]

    #%% BUILD THE MODEL

    if method == "rnn":
        model = RNN_Model.RNN_Model(args.model, ntokens, args.nRFF, args.nhid, args.nlayers, args.dropout, args.tied)

    if method == "ar" or method == 'arma':
        return LinearModel()

    if method == "kr":
        return KernelRegressionModel()
    
    return model

# ==============================================================================
# GetMultiData
# ==============================================================================
def GetMultiData(trajs, clusters, date, num_days, interval, num_mins, aggregate):
    date_list = [date - timedelta(minutes = x) for x in range(num_days * interval * aggregate,
        -num_mins, -aggregate)]

    traj = []

    for date in date_list:
        obs = []
        for c in clusters:
            if c in trajs:
                data_date = next(trajs[c].irange(maximum = date, inclusive = (True, False), reverse =
                    True), None)
            else:
                data_date = None
                print("cluster %d is not in trajs!!!", c)

            if data_date is None:
                data_point = 0
            else:
                data_point = trajs[c][data_date]
            obs.append(data_point)

        traj.append(obs)

    traj = np.array(traj)

    return traj
# ===================================================
# Normalize Function 
# ===================================================
def Normalize(data):
    # normalizing data
    data_min = 1 - np.min(data)
    data = np.log(data + data_min)
    data_mean = np.mean(data)
    data -= data_mean
    data_std = np.std(data)
    data /= data_std

    return data, data_min, data_mean, data_std
# ====================================================
# Predict function 
# ====================================================
def Predict(args, config, top_cluster, trajs, method):

    for date, cluster_list in top_cluster[args.start_pos // args.interval:- max(args.horizon //
            args.interval, 1)]:
        # Training delta
        first_date = top_cluster[0][0]
        train_delta_intervals = min(((date - first_date).days * 1440 + (date - first_date).seconds // 60
            ) // (args.aggregate * args.interval), args.training_intervals)
        #print(train_delta_intervals)
        #print(date, first_date)
        # Predict delta
        predict_delta_mins = args.horizon * args.aggregate

        print(date, first_date, date + timedelta(minutes = predict_delta_mins))

        clusters = next(zip(*cluster_list))[:args.top_cluster_num]

        data = GetMultiData(trajs, clusters, date, train_delta_intervals, args.interval, predict_delta_mins, args.aggregate)


        data, data_min, data_mean, data_std = Normalize(data)

        #print(data)
        print(data.shape)
        #print(args.interval, args.horizon)
        train_data = data[:-args.interval - args.horizon]
        print(train_data.shape)
        test_data = data[-(args.paddling_intervals * args.interval + args.horizon + args.interval):]
        print(test_data.shape)

        model = GetModel(args, train_data, method)

        criterion = nn.MSELoss()

        # Loop over epochs.
        for epoch in range(1, args.epochs + 1):
            print('epoch: ', epoch)
            epoch_start_time = time.time()
            lr = args.lr
            if epoch > 100:
                lr = 0.2
            train_pass(args, method, model, train_data, criterion, lr, args.bptt, args.clip, args.log_interval)
            print('about to evaluate: ')
            val_loss, y, y_hat, = evaluate_pass(args, method, model, test_data, criterion, args.bptt)
            Utilities.prettyPrint('Validation Loss: Epoch'+str(epoch), np.mean((y[-args.interval:] -
                y_hat[-args.interval:]) ** 2))

        # Run on test data.
        print('about to test')
        test_loss, y, y_hat= evaluate_pass(args, method, model, test_data, criterion, args.bptt)

        y = y[-args.interval:]
        y_hat = y_hat[-args.interval:]
        Utilities.prettyPrint('Test Loss', np.mean((y - y_hat) ** 2))
        Utilities.prettyPrint('Test Data Variance', np.mean(y ** 2))

        y = np.exp(y * data_std + data_mean) - data_min
        y_hat = np.exp(y_hat * data_std + data_mean) - data_min

        predict_dates = [date + timedelta(minutes = args.horizon * args.aggregate - x) for x in
                range(args.interval * args.aggregate, 0, -args.aggregate)]
        for i, c in enumerate(clusters):
            WriteResult(config['output_dir'] + str(c) + ".csv", predict_dates, y[:, i], y_hat[:, i])

# ======================================================================================================
# Write Result 
# ======================================================================================================
def WriteResult(path, dates, actual, predict):
    with open(path, "a") as csvfile:
        writer = csv.writer(csvfile, quoting = csv.QUOTE_ALL)
        for x in range(len(dates)):
            writer.writerow([dates[x], actual[x], predict[x]])
# ======================================================================================================
# Main 
# ======================================================================================================
def Main(config, method, input_dir, output_dir, cluster_path):

# ARGS setting ###################################################################
    args = Args()
    args.epochs = args.epochs[config["name"]]


    global NTOKENS
    NTOKENS = args.top_cluster_num

    args.horizon = horizon

    args.start_pos //= aggregate
    args.interval //= aggregate

    args.bptt = args.bptt[aggregate]
    args.batch_size = args.batch_size[aggregate]
    args.regress_dim = args.regress_dim[aggregate]

# Configure input, output directories ############################################

    input_dir = input_dir or config['input_dir']
    output_dir = output_dir or config['output_dir']
    cluster_path = cluster_path or config['cluster_path']

# Load data from input data ######################################################

    trajs = LoadData(input_dir, aggregate)

    with open(cluster_path, 'rb') as f:
        top_cluster, _ = pickle.load(f)
##################################################################################
# configure name of method, and from that the output dir name ####################
    method_name = method

    output_dir += "/agg-%s/horizon-%s/%s/" % (aggregate, horizon, method_name)

    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    config['output_dir'] = output_dir

# Call Predict function ###########################################################
    Predict(args, config, top_cluster, trajs, method)

# ==============================================
# call to Main()
# ==============================================
if __name__ == '__main__':
    aparser = argparse.ArgumentParser(description='Online timeseries prediction')
    aparser.add_argument('project', choices=PROJECTS.keys(), help='Data source type')
    aparser.add_argument('--method', help='Input Data Directory')
    aparser.add_argument('--input_dir', help='Input directory')
    aparser.add_argument('--output_dir', help='Output directory')
    aparser.add_argument('--cluster_path', help='Path of the clustering assignment')
    args = vars(aparser.parse_args())

    Main(PROJECTS[args['project']], args['method'], args['input_dir'], args['output_dir'], args['cluster_path'])
