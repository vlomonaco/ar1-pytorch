#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2019. Vincenzo Lomonaco. All rights reserved.                  #
# Copyrights licensed under the CC BY 4.0 License.                             #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 8-11-2019                                                              #
# Author: Vincenzo Lomonaco                                                    #
# E-mail: vincenzo.lomonaco@unibo.it                                           #
# Website: vincenzolomonaco.com                                                #
################################################################################

"""
General useful functions for machine learning with Pytorch.
"""

# Python 2-3 compatible
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import torch
from torch.autograd import Variable
from batch_renormalization import BatchRenormalization2D
from batch_renorm import BatchRenorm2d


def shuffle_in_unison(dataset, seed=None, in_place=False):
    """
    Shuffle two (or more) list in unison. It's important to shuffle the images
    and the labels maintaining their correspondence.

        Args:
            dataset (dict): list of shuffle with the same order.
            seed (int): set of fixed Cifar parameters.
            in_place (bool): if we want to shuffle the same data or we want
                             to return a new shuffled dataset.
        Returns:
            list: train and test sets composed of images and labels, if in_place
                  is set to False.
    """

    if seed:
        np.random.seed(seed)
    rng_state = np.random.get_state()
    new_dataset = []
    for x in dataset:
        if in_place:
            np.random.shuffle(x)
        else:
            new_dataset.append(np.random.permutation(x))
        np.random.set_state(rng_state)

    if not in_place:
        return new_dataset

def shuffle_in_unison_pytorch(dataset, seed=None):

    shuffled_dataset = []
    perm = torch.randperm(dataset[0].size(0))
    if seed:
        torch.manual_seed(seed)
    for x in dataset:
        shuffled_dataset.append(x[perm])

    return shuffled_dataset

def pad_data(dataset, mb_size):
    """
    Padding all the matrices contained in dataset to suit the mini-batch
    size. We assume they have the same shape.

        Args:
            dataset (str): sets to pad to reach a multile of mb_size.
            mb_size (int): mini-batch size.
        Returns:
            list: padded data sets
            int: number of iterations needed to cover the entire training set
                 with mb_size mini-batches.
    """

    num_set = len(dataset)
    x = dataset[0]
    # computing test_iters
    n_missing = x.shape[0] % mb_size
    if n_missing > 0:
        surplus = 1
    else:
        surplus = 0
    it = x.shape[0] // mb_size + surplus

    # padding data to fix batch dimentions
    if n_missing > 0:
        n_to_add = mb_size - n_missing
        for i, data in enumerate(dataset):
            dataset[i] = np.concatenate((data[:n_to_add], data))
    if num_set == 1:
        dataset = dataset[0]

    return dataset, it


def get_accuracy(model, criterion, batch_size, test_x, test_y, use_cuda=True,
                 mask=None, preproc=None):
    """
    Test accuracy given a model and the test data.

        Args:
            model (nn.Module): the pytorch model to test.
            criterion (func): loss function.
            batch_size (int): mini-batch size.
            test_x (tensor): test data.
            test_y (tensor): test labels.
            use_cuda (bool): if we want to use gpu or cpu.
            mask (bool): if we want to maks out some classes from the results.
        Returns:
            ave_loss (float): average loss across the test set.
            acc (float): average accuracy.
            accs (list): average accuracy for class.
    """

    model.eval()

    correct_cnt, ave_loss = 0, 0
    model = maybe_cuda(model, use_cuda=use_cuda)

    num_class = int(np.max(test_y) + 1)
    hits_per_class = [0] * num_class
    pattern_per_class = [0] * num_class
    test_it = test_y.shape[0] // batch_size + 1

    test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
    test_y = torch.from_numpy(test_y).type(torch.LongTensor)

    if preproc:
        test_x = preproc(test_x)

    for i in range(test_it):
        # indexing
        start = i * batch_size
        end = (i + 1) * batch_size

        x = maybe_cuda(test_x[start:end], use_cuda=use_cuda)
        y = maybe_cuda(test_y[start:end], use_cuda=use_cuda)

        logits = model(x)

        if mask is not None:
            # we put an high negative number so that after softmax that prob
            # will be zero and not contribute to the loss
            idx = (torch.FloatTensor(mask).cuda() == 0).nonzero()
            idx = idx.view(idx.size(0))
            logits[:, idx] = -10e10

        loss = criterion(logits, y)
        _, pred_label = torch.max(logits.data, 1)
        correct_cnt += (pred_label == y.data).sum()
        ave_loss += loss.item()

        for label in y.data:
            pattern_per_class[int(label)] += 1

        for i, pred in enumerate(pred_label):
            if pred == y.data[i]:
                hits_per_class[int(pred)] += 1

    accs = np.asarray(hits_per_class) / \
           np.asarray(pattern_per_class).astype(float)

    acc = correct_cnt.item() * 1.0 / test_y.size(0)

    ave_loss /= test_y.size(0)

    return ave_loss, acc, accs


def train_net(optimizer, model, criterion, mb_size, x, y, t,
              train_ep, preproc=None, use_cuda=True, mask=None,
              record_stats=False):
    """
    Train a Pytorch model from pre-loaded tensors.

        Args:
            optimizer (object): the pytorch optimizer.
            model (object): the pytorch model to train.
            criterion (func): loss function.
            mb_size (int): mini-batch size.
            x (tensor): train data.
            y (tensor): train labels.
            t (int): task label.
            train_ep (int): number of training epochs.
            preproc (func): test iterations.
            use_cuda (bool): if we want to use gpu or cpu.
            mask (bool): if we want to maks out some classes from the results.
            record_stats (bool): if we want to save collect sparsity stats.
        Returns:
            ave_loss (float): average loss across the train set.
            acc (float): average accuracy over training.
            stats (dict): dictionary of several stats collected.
    """

    cur_ep = 0
    cur_train_t = t
    stats = {'perc_on_avg': [], 'perc_on_std': [], 'on_idxs': []}

    if preproc:
        x = preproc(x)

    (train_x, train_y), it_x_ep = pad_data(
        [x, y], mb_size
    )

    shuffle_in_unison(
        [train_x, train_y], 0, in_place=True
    )

    model = maybe_cuda(model, use_cuda=use_cuda)
    acc = None
    ave_loss = 0

    train_x = torch.from_numpy(train_x).type(torch.FloatTensor)
    train_y = torch.from_numpy(train_y).type(torch.LongTensor)

    for ep in range(train_ep):

        model.active_perc_list = []
        model.train()

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

        if record_stats:
            perc_on_avg = np.mean(model.active_perc_list)
            perc_on_std = np.std(model.active_perc_list)

            print(
                "Average active units: {}%, std {}%: "
                .format(perc_on_avg, perc_on_std)
            )

            stats['perc_on_avg'].append(perc_on_avg)
            stats['perc_on_std'].append(perc_on_std)

    stats['on_idxs'] = model.on_idxs

    return ave_loss, acc, stats


def preprocess_imgs(img_batch, scale=True, norm=True, channel_first=True):
    """
    Here we get a batch of PIL imgs and we return them normalized as for
    the pytorch pre-trained models.

        Args:
            img_batch (tensor): batch of images.
            scale (bool): if we want to scale the images between 0 an 1.
            channel_first (bool): if the channel dimension is before of after
                                  the other dimensions (width and height).
            norm (bool): if we want to normalize them.
        Returns:
            tensor: pre-processed batch.

    """

    if scale:
        # convert to float in [0, 1]
        img_batch = img_batch / 255

    if norm:
        # normalize
        img_batch[:, :, :, 0] = ((img_batch[:, :, :, 0] - 0.485) / 0.229)
        img_batch[:, :, :, 1] = ((img_batch[:, :, :, 1] - 0.456) / 0.224)
        img_batch[:, :, :, 2] = ((img_batch[:, :, :, 2] - 0.406) / 0.225)

    if channel_first:
        # Swap channel dimension to fit the caffe format (c, w, h)
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))

    return img_batch


def maybe_cuda(what, use_cuda=True, **kw):
    """
    Moves `what` to CUDA and returns it, if `use_cuda` and it's available.

        Args:
            what (object): any object to move to eventually gpu
            use_cuda (bool): if we want to use gpu or cpu.
        Returns
            object: the same object but eventually moved to gpu.
    """

    if use_cuda is not False and torch.cuda.is_available():
        what = what.cuda()
    return what


def test_multitask(
        model, test_set, mb_size, preproc=None, use_cuda=True, multi_heads=[],
        mask=False):
    """
    Test a model considering that the test set is composed of multiple tests
    one for each task.

        Args:
            model (nn.Module): the pytorch model to test.
            test_set (list): list of (x,y,t) test tuples.
            mb_size (int): mini-batch size.
            preproc (func): image preprocess function.
            use_cuda (bool): if we want to use gpu or cpu.
            mask (bool): if we want to maks out some classes from the results.
            multi_heads (list): ordered list of "heads" to be used for each
                                task.
        Returns:
            stats (float): collected stasts of the test including average and
                           per class accuracies.
    """

    model.eval()

    acc_x_task = []
    stats = {'accs': []}

    for (x, y), t in test_set:

        # reduce test set 20 substampling
        # idx = range(0, y.shape[0], 20)
        # x = np.take(x, idx, axis=0)
        # y = np.take(y, idx, axis=0)

        if preproc:
            x = preproc(x)

        (test_x, test_y), it_x_ep = pad_data(
            [x, y], mb_size
        )

        if multi_heads != [] and len(multi_heads) > t:
            # we can use the stored head
            print("Using head: ", t)
            model.classifier = multi_heads[t]

        model = maybe_cuda(model, use_cuda=use_cuda)
        acc = None

        test_x = torch.from_numpy(test_x).type(torch.FloatTensor)
        test_y = torch.from_numpy(test_y).type(torch.LongTensor)

        correct_cnt, ave_loss = 0, 0

        with torch.no_grad():

            minclass = np.min(y)
            maxclass = np.max(y)

            for it in range(it_x_ep):

                start = it * mb_size
                end = (it + 1) * mb_size

                x_mb = maybe_cuda(test_x[start:end], use_cuda=use_cuda)
                y_mb = maybe_cuda(test_y[start:end], use_cuda=use_cuda)
                logits = model(x_mb)

                if mask:
                    totclass = logits.size(1)
                    mymask = np.asarray([0] * totclass)
                    mymask[minclass:maxclass+1] = 1
                    mymask = maybe_cuda(torch.from_numpy(mymask),
                                        use_cuda=use_cuda)
                    # print(mymask)
                    logits = mymask * logits

                _, pred_label = torch.max(logits, 1)
                correct_cnt += (pred_label == y_mb).sum()

            acc = correct_cnt.item() / test_y.shape[0]

        print('TEST Acc. Task {}==>>> acc: {:.3f}'.format(t, acc))
        acc_x_task.append(acc)
        stats['accs'].append(acc)

    print("------------------------------------------")
    print("Avg. acc:", np.mean(acc_x_task))
    print("------------------------------------------")

    # reset the head for the next batch
    if multi_heads:
        print("classifier reset...")
        model.reset_classifier()

    return stats

def replace_bn_with_brn(
        m, name="", momentum=0.1, r_d_max_inc_step=0.0001, r_max=1.0,
        d_max=0.0, max_r_max=3.0, max_d_max=5.0):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == torch.nn.BatchNorm2d:
            # print('replaced: ', name, attr_str)
            setattr(m, attr_str,
                    BatchRenormalization2D(
                        target_attr.num_features,
                        gamma=target_attr.weight,
                        beta=target_attr.bias,
                        running_mean=target_attr.running_mean,
                        running_var=target_attr.running_var,
                        eps=target_attr.eps,
                        momentum=momentum,
                        r_d_max_inc_step=r_d_max_inc_step,
                        r_max=r_max,
                        d_max=d_max,
                        max_r_max=max_r_max,
                        max_d_max=max_d_max
                        )
                    # BatchRenormalization2D(target_attr.num_features)
                    # BatchRenorm2d(
                    #     target_attr.num_features,
                    #     weight=target_attr.weight,
                    #     bias=target_attr.bias,
                    #     running_mean=target_attr.running_mean,
                    #     running_std=target_attr.running_var,
                    #     eps=target_attr.eps,
                    #     momentum=target_attr.momentum,
                    #     num_batches_tracked=target_attr.num_batches_tracked
                    #     )
                    )
    for n, ch in m.named_children():
        replace_bn_with_brn(ch, n, momentum, r_d_max_inc_step, r_max, d_max,
                            max_r_max, max_d_max)

def change_brn_pars(
        m, name="", momentum=0.1, r_d_max_inc_step=0.0001, r_max=1.0,
        d_max=0.0):
    for attr_str in dir(m):
        target_attr = getattr(m, attr_str)
        if type(target_attr) == BatchRenormalization2D:
            target_attr.momentum = torch.tensor((momentum), requires_grad=False)
            target_attr.r_max = torch.tensor(r_max, requires_grad=False)
            target_attr.d_max = torch.tensor(d_max, requires_grad=False)
            target_attr.r_d_max_inc_step = r_d_max_inc_step

    for n, ch in m.named_children():
        change_brn_pars(ch, n, momentum, r_d_max_inc_step, r_max, d_max)

def consolidate_weights(model, cur_clas):
    """ Mean-shift for the target layer weights"""

    with torch.no_grad():
        # print(model.classifier[0].weight.size())
        globavg = np.average(model.output.weight.detach()
                             .cpu().numpy()[cur_clas])
        # print("GlobalAvg: {} ".format(globavg))
        for c in cur_clas:
            w = model.output.weight.detach().cpu().numpy()[c]

            if c in cur_clas:
                new_w = w - globavg
                # if len(cur_clas) == 10:
                #     print("This is first batch!")
                #     new_w *= 4
                if c in model.saved_weights.keys():
                    wpast_j = np.sqrt(model.past_j[c] / model.cur_j[c])
                    # wpast_j = model.past_j[c] / model.cur_j[c]
                    model.saved_weights[c] = (model.saved_weights[c] * wpast_j
                     + new_w) / (wpast_j + 1)
                else:
                    model.saved_weights[c] = new_w

            # debug
            # print(
            #     "C: " + str(c) + "  -  Avg W: " + str(np.average(w)) +
            #     " Std W: " + str(np.std(w)) + " Max W: " + str(np.max(w))
            # )

def set_consolidate_weights(model):
    """ set trained weights """

    with torch.no_grad():
        for c, w in model.saved_weights.items():
            model.output.weight[c].copy_(
                torch.from_numpy(model.saved_weights[c])
            )


def reset_weights(model, cur_clas):
    """ reset weights"""

    with torch.no_grad():
        model.output.weight.fill_(0.0)
        # model.output.weight.copy_(
        #     torch.zeros(model.output.weight.size())
        # )
        for c, w in model.saved_weights.items():
            if c in cur_clas:
                model.output.weight[c].copy_(
                    torch.from_numpy(model.saved_weights[c])
                )

def examples_per_class(train_y):
    count = {i:0 for i in range(50)}
    for y in train_y:
        count[int(y)] +=1

    return count

def set_brn_to_train(m, name=""):
        for attr_str in dir(m):
            target_attr = getattr(m, attr_str)
            if type(target_attr) == BatchRenormalization2D:
                target_attr.train()
                # print("setting to train..")
        for n, ch in m.named_children():
            set_brn_to_train(ch, n)


def freeze_up_to(model, freeze_below_layer):
    for name, param in model.named_parameters():
        # tells whether we want to use gradients for a given parameter
        if "bn" not in name:
            param.requires_grad = False
            print("Freezing parameter " + name)
        if name == freeze_below_layer:
            break


if __name__ == "__main__":

    from mobilenet import MyMobilenetV1
    model = MyMobilenetV1(pretrained=True)

    replace_bn_with_brn(model, "net")



