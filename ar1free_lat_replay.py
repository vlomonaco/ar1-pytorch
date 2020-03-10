#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco. All rights reserved.                  #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 23-07-2019                                                             #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

""" Simple AR1* implementation in PyTorch with Latent Replay """

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

from data_loader import CORE50
import torch
import numpy as np
import copy
import datetime
import os
import json
from mobilenet import MyMobilenetV1
from utils import preprocess_imgs, pad_data, shuffle_in_unison, maybe_cuda,\
    get_accuracy, reset_weights, consolidate_weights, \
    set_consolidate_weights, examples_per_class, replace_bn_with_brn, \
    set_brn_to_train, change_brn_pars, shuffle_in_unison_pytorch
import tensorflow as tf

# Hardware setup
use_cuda = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# hyperparams
init_lr = 0.001
inc_lr = 0.0001
mb_size = 128
init_train_ep = 4
inc_train_ep = 4
init_update_rate = 0.01
inc_update_rate = 0.00001

# Tensorboard setup
exp_name = "brn_free_cwr_1500_v13"
log_dir = 'logs/' + exp_name
summary_writer = tf.summary.create_file_writer(log_dir)
tot_it_step = 0

# Create the dataset object
dataset = CORE50(root='/home/admin/Ior50N/128', scenario="nicv2_391")
preproc = preprocess_imgs

# Get the fixed test set
test_x, test_y = dataset.get_test_set()

# Model
model = MyMobilenetV1(pretrained=True)
replace_bn_with_brn(model, momentum=init_update_rate, r_d_max_inc_step=0.000164)
# replay_layer = "features.stage4.unit5.dw_conv.bn.bias"
model.saved_weights = {}
model.past_j = {i:0 for i in range(50)}
model.cur_j = {i:0 for i in range(50)}
latent_replay = False

# freeze_below_layer = "end_features.0.unit2.pw_conv.bn.bias"
# freeze_below_layer = "lat_features.4.unit6.pw_conv.bn.bias"
freeze_below_layer = "lat_features.4.unit6.pw_conv.bn.beta"
# freeze_below_layer = "end_features.0.unit2.pw_conv.bn.beta"
for name, param in model.named_parameters():
    # tells whether we want to use gradients for a given parameter
    param.requires_grad = False
    print("Freezing parameter " + name)
    if name == freeze_below_layer:
        break

optimizer = torch.optim.SGD(
    model.parameters(), lr=init_lr, momentum=0.9, weight_decay=0.0005
)
criterion = torch.nn.CrossEntropyLoss()

# replay pars
rm = None
rm_sz = 1500

# Saving hyper
hyper = json.dumps(
    {
        "mb_size": mb_size,
        "inc_train_ep": inc_train_ep,
        "init_train_ep": inc_train_ep,
        "init_lr": init_lr,
        "inc_lr": inc_lr,
        "rm_size": rm_sz,
        "init_update_rate": init_update_rate,
        "inc_update_rate": inc_update_rate
    }
)

with summary_writer.as_default():
    tf.summary.text("hyperparameters", hyper, step=0)

# loop over the training incremental batches
for i, train_batch in enumerate(dataset):

    if i == 1:
        change_brn_pars(
            model, momentum=inc_update_rate, r_d_max_inc_step=0, r_max=1.5,
            d_max=2.5)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=inc_lr, momentum=0.9, weight_decay=0.0005
        )

    train_x, train_y = train_batch
    train_x = preproc(train_x)

    if i == 0:
        cur_class = [int(o) for o in set(train_y)]
        model.cur_j = examples_per_class(train_y)
    else:
        cur_class = [int(o) for o in set(train_y).union(set(rm[1]))]
        model.cur_j = examples_per_class(list(train_y) + list(rm[1]))

    print("----------- batch {0} -------------".format(i))
    print("train_x shape: {}, train_y shape: {}"
          .format(train_x.shape, train_y.shape))

    model.eval()
    model.end_features.train()
    set_brn_to_train(model)
    print("setting to train..")
    model.output.train()

    reset_weights(model, cur_class)
    cur_ep = 0

    if i == 0:
        (train_x, train_y), it_x_ep = pad_data(
            [train_x, train_y], mb_size
        )

    shuffle_in_unison(
        [train_x, train_y], 0, in_place=True
    )

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    ave_loss = 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    if i == 0:
        train_ep = init_train_ep
    else:
        train_ep = inc_train_ep

    for ep in range(train_ep):

        print("training ep: ", ep)
        correct_cnt, ave_loss = 0, 0

        if i > 0:
            it_x_ep = train_x.size(0) // 21
            n2inject = 107
        else:
            n2inject = 0
        print("total sz:", train_x.size(0) + rm_sz)
        print("n2inject", n2inject)
        print("it x ep: ", it_x_ep)

        for it in range(it_x_ep):
            # print(it)
            start = it * (mb_size - n2inject)
            end = (it + 1) * (mb_size - n2inject)

            optimizer.zero_grad()

            x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)

            if i == 0:
                lat_mb_x = None
                y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)
                logits = model(x_mb)
            else:
                lat_mb_x = rm[0][it*n2inject: (it + 1)*n2inject]
                lat_mb_y = rm[1][it*n2inject: (it + 1)*n2inject]
                y_mb = maybe_cuda(
                    torch.cat((train_y[start:end], lat_mb_y), 0),
                    use_cuda=use_cuda)
                lat_mb_x = maybe_cuda(lat_mb_x, use_cuda=use_cuda)

            logits, lat_acts = model(
                x_mb, latent_input=lat_mb_x, return_lat_acts=True)

            # collect latent volumes only for the first ep
            if ep == 0:
                lat_acts = lat_acts.cpu().detach()
                if it == 0:
                    cur_acts = copy.deepcopy(lat_acts)
                else:
                    cur_acts = torch.cat((cur_acts, lat_acts), 0)

            _, pred_label = torch.max(logits, 1)
            correct_cnt += (pred_label == y_mb).sum()

            loss = criterion(logits, y_mb)
            ave_loss += loss.item()

            loss.backward()
            optimizer.step()

            acc = correct_cnt.item() / \
                  ((it + 1) * y_mb.size(0))
            ave_loss /= ((it + 1) * y_mb.size(0))

            if it % 10 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )

            # 1. Log scalar values (scalar summary) to TB
            tot_it_step +=1
            with summary_writer.as_default():
                tf.summary.scalar('train_loss', ave_loss, step=tot_it_step)
                tf.summary.scalar('train_accuracy', acc, step=tot_it_step)

        cur_ep += 1

    consolidate_weights(model, cur_class)

    # how many patterns to save for next iter
    h = min(rm_sz // (i + 1), cur_acts.size(0))
    print("h", h)

    print("cur_acts sz:", cur_acts.size(0))
    idxs_cur = np.random.choice(
        cur_acts.size(0), h, replace=False
    )
    rm_add = [cur_acts[idxs_cur], train_y[idxs_cur]]
    print("rm_add size", rm_add[0].size(0))

    # replace patterns in random memory
    if i == 0:
        rm = copy.deepcopy(rm_add)
    else:
        idxs_2_replace = np.random.choice(
            rm[0].size(0), h, replace=False
        )
        for j, idx in enumerate(idxs_2_replace):
            rm[0][idx] = copy.deepcopy(rm_add[0][j])
            rm[1][idx] = copy.deepcopy(rm_add[1][j])

    rm = shuffle_in_unison_pytorch(rm)

    set_consolidate_weights(model)
    ave_loss, acc, accs = get_accuracy(
        model, criterion, mb_size, test_x, test_y, preproc=preproc
    )

    # 1. Log scalar values (scalar summary) to TB
    with summary_writer.as_default():
        tf.summary.scalar('test_loss', ave_loss, step=i)
        tf.summary.scalar('test_accuracy', acc, step=i)

    # update number examples encountered over time
    for c, n in model.cur_j.items():
        model.past_j[c] += n

    print("---------------------------------")
    print("Accuracy: ", acc)
    print("---------------------------------")
