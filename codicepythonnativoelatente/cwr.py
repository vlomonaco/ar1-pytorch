#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

import sys
import os, time

import train_utils
import random as rnd

# ------ CWR ------

# call after the model weights have been randomly generated
def copy_initial_weights(net, layers_to_copy, num_classes):
    
    rand_w = {}   # dictionary to store randow weights
    cons_w = {}   # dictionary to store consolidated weights
    for layer in layers_to_copy:    # there can be more than one class-specific layer as in GoogleNet
        for c in range(num_classes):
            rand_w[layer,c] = (np.array(net.params[layer][0].data[c]), net.params[layer][1].data[c])    # each entry is a pair -> weights + biases (the second is a scalar value) 
            cons_w[layer,c] = (np.array(net.params[layer][0].data[c]), net.params[layer][1].data[c])
    return rand_w, cons_w

#def equal_weights(net, layers_to_copy, num_classes, source_weights):
#    for layer in layers_to_copy:
#        for c in range(num_classes):
#            (weights, bias) = source_weights[layer,c]
#            if (not np.array_equal(net.params[layer][0].data[c], weights)):
#                return False
#            if (not np.array_equal(net.params[layer][1].data[c], bias)):
#                return False
#    return True

def load_weights(net, layers_to_copy, num_classes, source_weights):
    for layer in layers_to_copy:
        for c in range(num_classes):
            (weights, bias) = source_weights[layer,c]
            net.params[layer][0].data[c] = weights  # makes a copy when assigning to an external objects 
            net.params[layer][1].data[c] = bias


def load_weights_nic(net, layers_to_copy, train_y, cons_w):
    class_to_load = np.unique(train_y.astype(np.int))
    for layer in layers_to_copy:
        for c in class_to_load:
            (w, b) = cons_w[layer, c]
            net.params[layer][0].data[c] = w  # makes a copy when assigning to an external objects 
            net.params[layer][1].data[c] = b


sys.path.insert(0, 'D:/PRJ/CNN/CaffePrj/Basic/Inference')
import visualization

def consolidate_weights_cwr(net, layers_to_copy, train_y, cons_w, contribution, class_updates = None):
    class_to_consolidate = np.unique(train_y.astype(np.int))
    for layer in layers_to_copy:
        for c in class_to_consolidate:
            (prev, _) = cons_w[layer,c]
            updated = (prev * class_updates[c] + net.params[layer][0].data[c]) / (class_updates[c] + 1)    # Works for both NC and NIC. For NC class_updates[c] is here always 0
            cons_w[layer,c] = (updated * contribution, net.params[layer][1].data[c])  # multiplication induce a copy
            #print("C ", c, "  -  Prev: ", class_updates[c])

    # # if class_to_consolidate.shape[0]==10: return;
    for layer in layers_to_copy:
        print()
        for c in class_to_consolidate:
            (w, b) = cons_w[layer,c]
            print("C ", c, "  -  Avg W: ", np.average(w), " Std W: ", np.std(w), " Max W: ", np.max(w), " - B: ", b)
            # visualization.WeightHistograms(w,100,'d:/Temp/'+layer.replace('/','_')+'_'+str(c)+'.png')


def _consolidate_weights_cwr_plus(net, layers_to_copy, train_y, cons_w, class_updates = None):
    class_to_consolidate = np.unique(train_y.astype(np.int))
    for layer in layers_to_copy:
        globavg = np.average(net.params[layer][0].data[class_to_consolidate])
        # print("GlobalAvg (", layer, ") ", globavg)
        for c in class_to_consolidate:
            (prev, _) = cons_w[layer,c]
            w = net.params[layer][0].data[c]
            prev_weight = np.sqrt(class_updates[c])
            #prev_weight = class_updates[c]
            updated = (prev * prev_weight + w-globavg) / (prev_weight + 1)    # Works for both NC and NIC. For NC class_updates[c] is here always 0
            # cons_w[layer,c] = (w-np.average(w), 0)  # multiplication induce a copy
            cons_w[layer,c] = (updated, net.params[layer][1].data[c])  # multiplication induce a copy

    # # if class_to_consolidate.shape[0]==10: return;
    for layer in layers_to_copy:
        print()
        for c in class_to_consolidate:
            (w, b) = cons_w[layer,c]
            print("C ", c, "  -  Avg W: ", np.average(w), " Std W: ", np.std(w), " Max W: ", np.max(w), " - B: ", b)

    #for layer in layers_to_copy:
    #    Wtot = np.zeros((class_to_consolidate.shape[0], net.params[layer][0].data[class_to_consolidate[0]].shape[0]), dtype = np.float32)
    #    row = 0
    #    for c in class_to_consolidate:
    #        (w,b) = cons_w[layer,c]
    #        Wtot[row] = np.copy(w)
    #        row+=1 
    #    sum = Wtot.sum() / class_to_consolidate.shape[0]      
    #    print("\n SUM (abs): ", sum)

def consolidate_weights_cwr_plus(net, layers_to_copy, class_to_consolidate, class_freq, prev_freq, cons_w):
    for layer in layers_to_copy:
        globavg = np.average(net.params[layer][0].data[class_to_consolidate])
        # print("GlobalAvg (", layer, ") ", globavg)
        for idx, c in enumerate(class_to_consolidate):
            (prev, _) = cons_w[layer,c]
            w = net.params[layer][0].data[c]
            prev_weight = np.sqrt(prev_freq[c]/class_freq[idx])
            #prev_weight = prev_freq[c]/class_freq[idx]
            updated = (prev * prev_weight + w-globavg) / (prev_weight + 1)    # Works for both NC and NIC. For NC class_updates[c] is here always 0
            cons_w[layer,c] = (updated, net.params[layer][1].data[c])  # multiplication induce a copy

    # # if class_to_consolidate.shape[0]==10: return;
    for layer in layers_to_copy:
        print()
        for c in class_to_consolidate:
            (w, b) = cons_w[layer,c]
            print("C ", c, "  -  Avg W: ", np.average(w), " Std W: ", np.std(w), " Max W: ", np.max(w), " - B: ", b)


def set_brn_past_weight(net, weight):
    for layer, param in net.params.items():
        layer_pos = list(net._layer_names).index(layer)   # posizione del layer
        if net.layers[layer_pos].type == 'BatchReNorm':
            assert len(param) == 4, "For BRN layers we expect four blobs: Mean, Variance, PastWeight, Iter!"
            scale = weight/net.params[layer][2].data[0]
            net.params[layer][0].data[...] *= scale
            net.params[layer][1].data[...] *= scale
            net.params[layer][2].data[0] = weight
            

#

# ------ CWR+ -------
# -------------------------------
# --- To manipulate LR -> add to _Caffe.cpp in pyCaffe ---
#  #ifdef EXPOSE_LR
#       .add_property("_params_lr", bp::make_function(
#           &Net<Dtype>::params_lr, bp::return_internal_reference<>()))
#  #endif

def zeros_cwr_layer_bias_lr(net, layers_to_copy, force_weights_lr_mult = -1):
    lr_entry = 0
    for layer, param in net.params.items():
        blobs = len(param)
        if layer in layers_to_copy:
            assert blobs == 2, "For CwR layers we expect two blobs: W and b!"
            if force_weights_lr_mult > 0:
                net._params_lr[lr_entry] = force_weights_lr_mult     # Force multiplier for weights LR in CWR layers!!! useful to increase this LR with respect to lower levels
            net._params_lr[lr_entry+1] = 0  
        lr_entry += blobs

# set all learning rates to 0 except for cwr layers
def zeros_non_cwr_layers_lr(net, layers_to_copy):
    lr_entry = 0
    for layer, param in net.params.items():
        blobs = len(param)
        if layer not in layers_to_copy:
            for pos in range(lr_entry,lr_entry+blobs):
                net._params_lr[pos] = 0       # tutti i blob parametri. Non posso usare slicing perchè è un vettore C++ (non esposto come np array)
        lr_entry += blobs

# set learning rates to 0 for cwr layers
#def zeros_cwr_layers_lr(net, layers_to_copy):
#    lr_entry = 0
#    for layer, param in net.params.items():
#        blobs = len(param)
#        if layer in layers_to_copy:
#            for pos in range(lr_entry,lr_entry+blobs):
#                net._params_lr[pos] = 0  
#        lr_entry += blobs

# reset all learning rates 
#def set_all_level_learning_rates(net, lr):
#    lr_entry = 0
#    for layer, param in net.params.items():
#        blobs = len(param)
#        for pos in range(lr_entry,lr_entry+blobs):
#            net._params_lr[pos] = 0       # tutti i blob parametri. Non posso usare slicing perchè è un vettore C++ (non esposto come np array)
#        lr_entry += blobs

def init_consolidated_weights(net, layers_to_copy, num_classes):
    cons_w = {}   # dictionary to store consolidated weights
    for layer in layers_to_copy:    # there can be more than one class-specific layer as in GoogleNet
        for c in range(num_classes):
            cons_w[layer,c] = (np.zeros_like(net.params[layer][0].data[c]), 0)
    return cons_w
   
def reset_weights(net, layers_to_copy, num_classes):
    for layer in layers_to_copy:
        for c in range(num_classes):
            net.params[layer][0].data[c] = 0  
            net.params[layer][1].data[c] = 1  # ... non cambia praticamente nulla anche se nel proto della rete era messo a 1


# perde 6-7% rispetto a maximal head. Se poi uso boost iniziale del learning rate riparametrizzando perdo 3% circa.
def reset_weights_single_head(net, layers_to_copy, num_classes, train_y):
    class_to_consolidate = np.unique(train_y.astype(np.int))
    # max_class = class_to_consolidate.max()
    for layer in layers_to_copy:
        for c in range(num_classes):
            net.params[layer][0].data[c] = -100  #-0.001 * rnd.random() #-100
            net.params[layer][1].data[c] = 0
        for c in class_to_consolidate:
        # start_class = np.min(class_to_consolidate)    
        # for c in range(start_class, start_class+10):
            net.params[layer][0].data[c] = 0  
        # for c in range(max_class+1):
        #    net.params[layer][0].data[c] = 0  

# sembra perdere poco rispetto a maximal head (circa 2% su run0 Caffenet)
def dynamic_head_expansion_cwr(net, layers_to_copy, num_classes, train_y):
    class_to_consolidate = np.unique(train_y.astype(np.int))
    min_class = 0
    max_class = class_to_consolidate.max()
    for layer in layers_to_copy:
        for c in range(min_class, max_class+1):
            net.params[layer][0].data[c] = 0
        for c in range(max_class+1, 50):   # usare num_classes!?!
            net.params[layer][0].data[c] = -100
            net.params[layer][1].data[c] = 0

