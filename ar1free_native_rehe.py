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
from data_loader import CORE50
import torch
import numpy as np
import copy
import datetime
import os
import json
from models.mobilenet import MyMobilenetV1
from utils import preprocess_imgs, pad_data, shuffle_in_unison, maybe_cuda,\
    get_accuracy, reset_weights, consolidate_weights, \
    set_consolidate_weights, examples_per_class, replace_bn_with_brn, \
    set_brn_to_train, change_brn_pars, shuffle_in_unison_pytorch, freeze_up_to
from torch.utils.tensorboard import SummaryWriter

# Hardware setup
use_cuda = True
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# hyperparams
init_lr = 0.0001
inc_lr = 0.0003
mb_size = 128
init_train_ep = 4
inc_train_ep = 4
init_update_rate = 0.01
inc_update_rate = 0.0001
max_r_max = 1.25
max_d_max = 0.5
inc_step = 4.1e-05
rm_sz = 1500
momentum = 0.9
l2 = 0.0005
freeze_below_layer = "lat_features.19.conv.weight"
rm = None

# Tensorboard setup
exp_name = "ar1free_bn_native_frozen_below_v3"
comment = ""

log_dir = 'logs/' + exp_name
writer = SummaryWriter(log_dir)
tot_it_step = 0

# Create the dataset object
dataset = CORE50(root='/home/admin/ssd_data/core50', scenario="nicv2_391")
preproc = preprocess_imgs

# Get the fixed test set
test_x, test_y = dataset.get_test_set()

# Model
model = MyMobilenetV1(pretrained=True, latent_layer_num=19)
# replace_bn_with_brn(
#     model, momentum=init_update_rate, r_d_max_inc_step=inc_step,
#     max_r_max=max_r_max, max_d_max=max_d_max
# )
model.saved_weights = {}
model.past_j = {i:0 for i in range(50)}
model.cur_j = {i:0 for i in range(50)}

# optimizer
optimizer = torch.optim.SGD(
    model.parameters(), lr=init_lr, momentum=momentum, weight_decay=l2
)
criterion = torch.nn.CrossEntropyLoss()

hyper = json.dumps(
    {
        "mb_size": mb_size,
        "inc_train_ep": inc_train_ep,
        "init_train_ep": inc_train_ep,
        "init_lr": init_lr,
        "inc_lr": inc_lr,
        "rm_size": rm_sz,
        "init_update_rate": init_update_rate,
        "inc_update_rate": inc_update_rate,
        "momentum": momentum,
        "l2": l2,
        "max_r_max": max_r_max,
        "max_d_max": max_d_max,
        "r_d_inc_step": inc_step,
        "freeze_below_layer": freeze_below_layer,
        "comment": comment
    }
)

writer.add_text("parameters", hyper, 0)

# loop over the training incremental batches
for i, train_batch in enumerate(dataset):

    if i == 1:
        if freeze_below_layer:
            freeze_up_to(model, freeze_below_layer)
        # change_brn_pars(
        #     model, momentum=inc_update_rate, r_d_max_inc_step=0, r_max=1.25,
        #     d_max=0.5)
        optimizer = torch.optim.SGD(
            model.parameters(), lr=inc_lr, momentum=momentum, weight_decay=l2
        )

    train_x, train_y = train_batch
    train_x = preproc(train_x)

    # saving patterns for next iter
    h = min(rm_sz // (i + 1), train_x.shape[0])
    # h = rm_sz // (i + 1)
    print("h", h)
    print("train x sz: ", train_x.shape[0])

    idxs_cur = np.random.choice(
        train_x.shape[0], h, replace=False
    )
    # idxs_cur = np.random.choice(
    #     train_x.shape[0], h, replace=True
    # )
    rm_add = [train_x[idxs_cur], train_y[idxs_cur]]

    print("rm_add size", rm_add[0].shape[0])

    # adding eventual replay patterns to the current batch
    if i > 0:
        print("rm size", rm[0].shape[0])
        train_x = np.concatenate((train_x, rm[0]))
        train_y = np.concatenate((train_y, rm[1]))

    cur_class = [int(o) for o in set(train_y)]
    model.cur_j = examples_per_class(train_y)

    print("----------- batch {0} -------------".format(i))
    print("train_x shape: {}, train_y shape: {}"
          .format(train_x.shape, train_y.shape))

    if i == 0:
        # model.eval()
        # set_brn_to_train(model)
        # model.end_features.train()
        # model.output.train()
        model.train()
    else:
        model.eval()
        model.end_features.train()
        model.output.train()

    reset_weights(model, cur_class)
    cur_ep = 0

    (train_x, train_y), it_x_ep = pad_data([train_x, train_y], mb_size)
    shuffle_in_unison([train_x, train_y], in_place=True)

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

        for it in range(it_x_ep):

            start = it * mb_size
            end = (it + 1) * mb_size

            optimizer.zero_grad()

            x_mb = maybe_cuda(train_x[start:end], use_cuda=use_cuda)
            y_mb = maybe_cuda(train_y[start:end], use_cuda=use_cuda)

            logits = model(x_mb)

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
            tot_it_step +=1
            writer.add_scalar('train_loss', ave_loss, tot_it_step)
            writer.add_scalar('train_accuracy', acc, tot_it_step)

        cur_ep += 1

    consolidate_weights(model, cur_class)

    # replace patterns in random memory
    if i == 0:
        rm = copy.deepcopy(rm_add)
    else:
        # shuffle_in_unison(rm, seed=0, in_place=True)
        idxs_2_replace = np.random.choice(
            rm[0].shape[0], h, replace=False
        )
        for j, idx in enumerate(idxs_2_replace):
            rm[0][idx] = copy.deepcopy(rm_add[0][j])
            rm[1][idx] = copy.deepcopy(rm_add[1][j])

    set_consolidate_weights(model)
    ave_loss, acc, accs = get_accuracy(
        model, criterion, mb_size, test_x, test_y, preproc=preproc
    )

    # 1. Log scalar values (scalar summary) to TB
    writer.add_scalar('test_loss', ave_loss, i)
    writer.add_scalar('test_accuracy', acc, i)

    # update number examples encountered over time
    for c, n in model.cur_j.items():
        model.past_j[c] += n

    print("past_j:", model.past_j)
    count = [0] * 50
    for j in rm[1]:
        count[int(j)] +=1
    print(count)

    print("---------------------------------")
    print("Accuracy: ", acc)
    print("---------------------------------")

writer.close()
