#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import sys
import os, time

import train_utils
import random as rnd

# ------ REHEARSAL ------
ExtMem = [[],[]]
ExtMemSize = 0

def allocate_memory(num_patterns, size_x, size_y):
    global ExtMem
    global ExtMemSize

    assert size_y == 1, "Reharshal Memory: size_y must be 1!"
    ExtMem =[   np.zeros((num_patterns, size_x), dtype = np.float32),
                np.zeros((num_patterns), dtype = np.int32),
            ]
    ExtMemSize = num_patterns

    print("Reharshal Memory created for %d pattern and size %d" % (num_patterns, num_patterns*(size_x+size_y)))

# Seleziona random da n patterns per rimpiazzare pattern random della memoria esterna
def update_memory(train_x, train_y, batch):
    global ExtMem
    global ExtMemSize

    # n_cur_batch = ExtMemSize // (batch + 1)    # pattern dal batch corrente (al batch 0, pari alla lunghezza del batch)
    # n_cur_batch = (ExtMemSize // np.sqrt((batch + 1))).astype(np.int)    # pattern dal batch corrente (al batch 0, pari alla lunghezza del batch)
    n_cur_batch = ExtMemSize // (batch + 1)    # pattern dal batch corrente (al batch 0, pari alla lunghezza del batch)
    if n_cur_batch > train_x.shape[0]:
        n_cur_batch = train_x.shape[0]
    n_ext_mem = ExtMemSize - n_cur_batch       # i rimanenti dalla memoria

    assert n_ext_mem >= 0, "Rehearsal: n_ext_mem should never be less than 0!"
    assert n_cur_batch <= train_x.shape[0], "Rehearsal: non enough pattern to get in current batch!"
    #assert batch > 0 or ExtMemSize <= train_x.shape[0], "Rehearsal: for Batch 0, the batch size must be large enough to fill the external memory."

    idxs_cur = np.random.choice(train_x.shape[0], n_cur_batch, replace=False)   # indici pattern scelti in current batch

    if n_ext_mem == 0:      # tutto dal batch corrente
        ExtMem = [train_x[idxs_cur], train_y[idxs_cur]]
    else:
        idxs_ext = np.random.choice(ExtMemSize, n_ext_mem, replace=False)
        ExtMem= [   np.concatenate((train_x[idxs_cur], ExtMem[0][idxs_ext])),
                    np.concatenate((train_y[idxs_cur], ExtMem[1][idxs_ext]))
                ]


# ritorna tutta la memoria esterna
def get_samples():
    global ExtMem

    return ExtMem[0], ExtMem[1] 
