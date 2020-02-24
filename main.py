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
from pytorchcv.model_provider import get_model as ptcv_get_model
from utils import preprocess_imgs, pad_data, shuffle_in_unison, maybe_cuda,\
    get_accuracy, reset_weights, consolidate_weights, set_consolidate_weights

# Hardware setup
use_cuda = True

# Create the dataset object
dataset = CORE50(root='/home/admin/Ior50N/128', scenario="nicv2_79")
preproc = preprocess_imgs

# Get the fixed test set
test_x, test_y = dataset.get_test_set()

# Model
model = ptcv_get_model("mobilenet_w1", pretrained=True)
model.features.final_pool = torch.nn.AvgPool2d(4)
model.output = torch.nn.Linear(1024, 50, bias=True)
replay_layer = "features.stage4.unit5.dw_conv.bn.bias"
model.saved_weights = {}

# freezing layers below threshold
freeze_below_layer = "features.stage5.unit2.pw_conv.bn.bias"
for name, param in model.named_parameters():
    # tells whether we want to use gradients for a given parameter
    param.requires_grad = False
    print("Freezing parameter " + name)
    if name == freeze_below_layer:
        break

# optimizer
optimizer = torch.optim.SGD(model.output.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
mb_size = 128
inc_train_ep = 4
init_train_ep = 4

# replay pars
rm = None
rm_sz = 2900

# loop over the training incremental batches
for i, train_batch in enumerate(dataset):

    train_x, train_y = train_batch
    train_x = preproc(train_x)

    # saving patterns for next iter
    h = rm_sz // (i + 1)
    idxs_cur = np.random.choice(
        train_x.shape[0], h, replace=False
    )
    rm_add = [train_x[idxs_cur], train_y[idxs_cur]]
    print("h", h)
    print("rm_add size", rm_add[0].shape[0])

    # adding eventual replay patterns to the current batch
    if i > 0:
        print("rm size", rm[0].shape[0])
        train_x = np.concatenate((train_x, rm[0]))
        train_y = np.concatenate((train_y, rm[1]))

    cur_class = [int(o) for o in set(train_y)]

    print("----------- batch {0} -------------".format(i))
    print("train_x shape: {}, train_y shape: {}"
          .format(train_x.shape, train_y.shape))

    # if i == 1:
    #     optimizer = torch.optim.SGD(model.output.parameters(), lr=0.0001)

    model.eval()
    model.output.train()
    # model.train()
    reset_weights(model, cur_class)
    cur_ep = 0


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

            if it % 100 == 0:
                print(
                    '==>>> it: {}, avg. loss: {:.6f}, '
                    'running train acc: {:.3f}'
                        .format(it, ave_loss, acc)
                )

        cur_ep += 1

    consolidate_weights(model, cur_class)

    # replace patterns in random memory
    if i == 0:
        rm = copy.deepcopy(rm_add)
    else:
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
    print("Accuracy: ", acc)
