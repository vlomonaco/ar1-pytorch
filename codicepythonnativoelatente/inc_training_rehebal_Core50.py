# Reharsal with Balanced minibatches (only Naive, CWR* and AR1)

import sys, os, time
import numpy as np    

#caffe_root = 'D:/PRJ/CaffeGpu_/'
caffe_root = 'D:/PRJ/CaffeCuda10/'
sys.path.insert(0, caffe_root + 'python')
os.environ["GLOG_minloglevel"] = "1"  # limit logging (0 - debug, 1 - info (still a LOT of outputs), 2 - warnings, 3 - errors)
import caffe

# per il parsing dei prototxt
from caffe.proto import caffe_pb2
import google.protobuf.text_format as txtf

from sklearn.metrics import confusion_matrix

sys.path.insert(0, 'D:/PRJ/CNN/CaffePrj/Basic/Inference')
sys.path.insert(0, 'D:/PRJ/CNN/CaffePrj/Basic/Training')
import visualization
import filelog
import train_utils
import lwf, ewc, cwr, syn
import rehearsal

def main_Core50(conf, run, close_at_the_end = False):
   
    # prepare configurations file
    conf['solver_file_first_batch'] = conf['solver_file_first_batch'].replace('X', conf['model'])
    conf['solver_file'] = conf['solver_file'].replace('X', conf['model'])
    conf['init_weights_file'] = conf['init_weights_file'].replace('X', conf['model'])
    conf['tmp_weights_file'] = conf['tmp_weights_file'].replace('X', conf['model'])
    train_filelists = conf['train_filelists'].replace('RUN_X', run)
    test_filelist = conf['test_filelist'].replace('RUN_X', run)
    run_on_the_fly = False if run == 'run0' else True     # for run 0 store/load binary files, for the rest of runs read single files (slower but save disk)

    # To change if needed the network prototxt
    # train_utils.modify_net_prototxt("D:/PRJ/CNN/Experiments/Core50/NIC/NIC_MobileNetV1dw_froz_brn3.prototxt", "D:/PRJ/CNN/Experiments/Core50/NIC/NIC_MobileNetV1dw_froz_brn3b.prototxt")
    # train_utils.modify_net_prototxt("D:/PRJ/CNN/Experiments/Core50/NIC/NIC_MobileNetV1dw_first_batch_brn2.prototxt", "D:/PRJ/CNN/Experiments/Core50/NIC/NIC_MobileNetV1dw_first_batch_brn2b.prototxt")
    
    # train_utils.modify_net_prototxt("D:/PRJ/CNN/Experiments/Core50/NIC/NIC_MobileNetV1dw_froz.prototxt", "D:/PRJ/CNN/Experiments/Core50/NIC/NIC_MobileNetV1dw_froz_bn.prototxt")
    # train_utils.modify_net_prototxt("D:/PRJ/CNN/Experiments/Core50/NC/NC_CaffeNet.prototxt", "D:/PRJ/CNN/Experiments/Core50/NC/NC_CaffeNet_b1.prototxt")
    # train_utils.modify_net_prototxt("D:/PRJ/CNN/Experiments/Core50/Cumulative/Cumulative_MobileNetV1_first_batch.prototxt", "D:/PRJ/CNN/Experiments/Core50/Cumulative/Cumulative_MobileNetV1dw_first_batch.prototxt")
    # train_utils.modify_net_prototxt("D:/PRJ/CNN/Experiments/Core50/NI/NI_MobileNetV1dw.prototxt", "D:/PRJ/CNN/Experiments/Core50/NI/NI_MobileNetV1dw_frozAll.prototxt")

    # parse solver
    #  for more details see - https://stackoverflow.com/questions/31823898/changing-the-solver-parameters-in-caffe-through-pycaffe
    print('Solver proto: ',conf['solver_file_first_batch'])
    solver_param = caffe_pb2.SolverParameter()
    with open(conf['solver_file_first_batch']) as f:
        txtf.Merge(str(f.read()), solver_param)
    net_prototxt = solver_param.net
    print('Net proto: ',net_prototxt)

    # recover class labels
    if conf['class_labels'] != '':
        label_str = np.loadtxt(conf['class_labels'], dtype=bytes, delimiter="\n").astype(str)     # more complex than a simple loadtxt because of unicode representation in python 3
            
    # recovering minibatch size from net proto
    train_minibatch_size, test_minibatch_size = train_utils.extract_minibatch_size_from_prototxt_with_input_layers(net_prototxt)  
    print(' test minibatch size: ', test_minibatch_size)
    print(' train minibatch size: ', train_minibatch_size)
    
    # is the network using target vectors (besides the labels)?
    need_target = train_utils.net_use_target_vectors(net_prototxt)

    # recovering test set
    print ("Recovering Test Set: ", test_filelist, " ...")    
    start = time.time()
    test_x, test_y = train_utils.get_data(test_filelist, conf['db_path'], conf['exp_path'], on_the_fly=run_on_the_fly, verbose = conf['verbose'])   
    assert(test_x.shape[0] == test_y.shape[0])
    if conf['num_classes'] == 10:  # Category based classification
        test_y = test_y // 5
    test_y = test_y.astype(np.float32)
    test_patterns = test_x.shape[0]
    test_x, test_y, test_iterat = train_utils.pad_data(test_x, test_y, test_minibatch_size)
    print (' -> %d patterns of %d classes (%.2f sec.)' % (test_patterns,len(np.unique(test_y)),time.time()-start))
    print (' -> %.2f -> %d iterations for full evaluation' % (test_patterns / test_minibatch_size, test_iterat))
    

    # recover training patterns in batches (by now assume the same number in all batches) 
    batch_count = conf['num_batches']
    train_patterns = train_utils.count_lines_in_batches(batch_count,train_filelists)
    train_iterations_per_epoch = np.zeros(batch_count, int)
    train_iterations = np.zeros(batch_count, int)
    test_interval_epochs = conf['test_interval_epochs']
    test_interval = np.zeros(batch_count, float)
    for batch in range(batch_count):
        if conf["rehearsal"] and batch > 0:
            train_patterns[batch] += conf["rehearsal_memory"]
        train_iterations_per_epoch[batch] = int(np.ceil(train_patterns[batch] / train_minibatch_size))     
        test_interval[batch] = test_interval_epochs * train_iterations_per_epoch[batch]
        if (batch == 0):
            train_iterations[batch] = train_iterations_per_epoch[batch] * conf['num_epochs_first_batch']
        else:
            train_iterations[batch] = train_iterations_per_epoch[batch] * conf['num_epochs']
        print ("Batch %2d: %d patterns, %d iterations (%d iter. per epochs - test every %.1f iter.)" \
             % (batch, train_patterns[batch],  train_iterations[batch], train_iterations_per_epoch[batch], test_interval[batch]))
    
    # Create evaluation points
    # iterations which are boundaries of batches
    batch_iter = [0]  
    iter = 0
    for batch in range(batch_count):    
        iter += train_iterations[batch]
        batch_iter.append(iter)
    # iterations where the network will be evaluated
    eval_iters = [1]   # start with 1 (insted of 0) because the test net is aligned to the train one after solver.step(1)
    for batch in range(batch_count):    
        start = batch_iter[batch]
        end = batch_iter[batch+1]
        start += test_interval[batch]
        while start < end:
            eval_iters.append(int(start))
            start += test_interval[batch]
        eval_iters.append(end)
    # iterations which are epochs in the evaluation range
    epochs_iter = []
    for batch in range(batch_count):                
        start = batch_iter[batch]
        end = batch_iter[batch+1]
        start += train_iterations_per_epoch[batch]
        while start <= end:
            epochs_iter.append(int(start))
            start += train_iterations_per_epoch[batch]

    prev_train_loss = np.zeros(len(eval_iters))
    prev_test_acc = np.zeros(len(eval_iters))
    prev_exist = filelog.TryLoadPrevTrainingLog(conf['train_log_file'], prev_train_loss, prev_test_acc)
    train_loss = np.copy(prev_train_loss)  # copy allows to correctly visualize the graph in case we start from initial_batch > 0
    test_acc = np.copy(prev_test_acc)      # as above
    train_acc = np.zeros(len(eval_iters))    

    epochs_tick = False if batch_count > 30 else True  # for better visualization
    visualization.Plot_Incremental_Training_Init('Incremental Training', eval_iters, epochs_iter, batch_iter, train_loss, test_acc, 5, conf['accuracy_max'], prev_exist, prev_train_loss, prev_test_acc, show_epochs_tick = epochs_tick)
    filelog.Train_Log_Init(conf['train_log_file'])
    filelog.Train_LogDetails_Init(conf['train_log_file'])

    start_train = time.time()
    eval_idx = 0   # counter of evaluation
    global_eval_iter = 0  # counter of global iteration
    first_round = True
    initial_batch = conf['initial_batch']    
    if  initial_batch > 0:  # move forward by skipping unnecessary evaluation
        global_eval_iter = batch_iter[initial_batch]
        while eval_iters[eval_idx] < global_eval_iter:
            eval_idx += 1
        eval_idx += 1  # one step more (eval on the entry baoundary of a batch is skipped)   

    for batch in range(initial_batch, batch_count):

        print ('\nBATCH = {:2d} ----------------------------------------------------'.format(batch))
        
        if batch == 0:
            solver = caffe.get_solver(conf['solver_file_first_batch'])   # load solver and create net
            if conf['init_weights_file'] !='':
                solver.net.copy_from(conf['init_weights_file'])
                print('Network created and Weights loaded from: ', conf['init_weights_file'])
                # Test
                solver.share_weights(solver.test_nets[0])  # necessario se non usi step(1)
                print('Weights shared with Test Net')
                # solver.test_nets[0].copy_from(conf['init_weights_file'])  # caffe la sincronizza solo dopo primo solve(1). Li carico esplicitamente
                accuracy, _ , pred_y = train_utils.test_network_with_accuracy_layer(solver, test_x, test_y, test_iterat, test_minibatch_size, prediction_level_Model[conf['model']], return_prediction = True)
            
            if conf['strategy'] in ['cwr+','ar1']:
                cwr.zeros_cwr_layer_bias_lr(solver.net, cwr_layers_Model[conf['model']])
                class_updates = np.zeros(conf['num_classes'], dtype=np.float32)
                cons_w = cwr.init_consolidated_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])    # allocate space for consolidated weights and initialze to 0
                cwr.reset_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])   # reset weights to 0 (done here for the first batch to keep initial stats correct)                 

            # ZERO INIT PESI LIVELLO OUT
            # cwr.reset_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])   # reset weights to 0 (done here for the first batch to keep initial stats correct)                 

            if conf['strategy'] in ['ar1']:
                ewcData, synData = syn.create_syn_data(solver.net)   # ewcData stores optimal weights + normalized fisher; trajectory store unnormalized summed grad*deltaW
                        
            rehearsal.allocate_memory(conf['rehearsal_memory'],test_x[0].size,1)

            # Blob 0 for L1 loss (sparsification)
            # solver.net.blobs['dummy_Conv5_4_zero'].data[...] = np.zeros_like(solver.net.blobs['conv5_4/dw'].data)

        elif batch == 1:
            solver = caffe.get_solver(conf['solver_file'])   # load solver and create net
            solver.net.copy_from(conf['tmp_weights_file'])
            print('Network created and Weights loaded from: ', conf['tmp_weights_file'])
            solver.share_weights(solver.test_nets[0]) # necessario se non usi step(1)
            print('Weights shared with Test Net')

            if conf['strategy'] in ['cwr+']:
                cwr.zeros_non_cwr_layers_lr(solver.net, cwr_layers_Model[conf['model']])   # blocca livelli sotto

            if conf['strategy'] in ['cwr+', 'ar1']:
                if 'cwr_lr_mult' in conf.keys() and conf['cwr_lr_mult']!=1:
                    cwr.zeros_cwr_layer_bias_lr(solver.net, cwr_layers_Model[conf['model']], force_weights_lr_mult = conf['cwr_lr_mult'])
                else:
                    cwr.zeros_cwr_layer_bias_lr(solver.net, cwr_layers_Model[conf['model']])

            # riscala peso passato
            cwr.set_brn_past_weight(solver.net, 10000)

        #only the first round
        if first_round:
            if batch == 1 and (conf['strategy'] in ['ewc','cwr', 'cwr+', 'syn', 'ar1']):
                print('Cannot start from batch 1 in ', conf['strategy'], ' strategy!')
                sys.exit(0)
            visualization.PrintNetworkArchitecture(solver.net)
            # if accuracy layer is defined in the prototxt also in TRAIN mode -> log also train accuracy (not in the plot)
            try:
                report_train_accuracy = True
                err = solver.net.blobs['accuracy'].num  # assume this is stable for prototxt of successive batches
            except: 
                report_train_accuracy = False
            first_round = False
            if conf['compute_param_stats']: 
                param_change = {}
                param_stats = train_utils.stats_initialize_param(solver.net)
                #nonzero_activations = train_utils.stats_activations_initialize(solver.net)
            
        # training data
        current_train_filelist = train_filelists.replace('XX', str(batch).zfill(2))
        print ("Recovering training data: ", current_train_filelist, " ...")
        load_start = time.time()        
        batch_x, batch_y = train_utils.get_data(current_train_filelist, conf['db_path'], conf['exp_path'], on_the_fly=run_on_the_fly, verbose = conf['verbose'])
        print ("Done.")
        if conf['num_classes'] == 10:  # Category based classification
            batch_y = batch_y // 5
        
        batch_t = train_utils.compute_one_hot_vectors(batch_y, conf['num_classes'])        
        rehe_x, rehe_y = rehearsal.get_samples()            
        rehe_t = train_utils.compute_one_hot_vectors(rehe_y, conf['num_classes'])

        if batch == 0: classes_in_cur_train = batch_y.astype(np.int)
        else: classes_in_cur_train = np.concatenate((batch_y.astype(np.int),rehe_y.astype(np.int)))
        unique_y, y_freq = np.unique(classes_in_cur_train, return_counts=True)  # classi presenti nel batch e loro frequenze

        if conf['strategy'] in ['cwr+','ar1'] and batch > initial_batch:  # in the first batch done before to correctly update stats
            cwr.reset_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])  # reset weights to 0  (maximal head approach!)                 
            if 'cwr_nic_load_weight' in conf.keys() and conf['cwr_nic_load_weight']:
                cwr.load_weights_nic(solver.net, cwr_layers_Model[conf['model']], unique_y, cons_w) 
            # da abilitare (una delle due sotto) per simulare altre modalità head 
            # cwr.dynamic_head_expansion_cwr(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'], batch_y) 
            # cwr.reset_weights_single_head(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'], batch_y)  # reset weights to 0                   

        if conf['strategy'] in ['ar1'] and batch > initial_batch:
            syn.weight_stats(solver.net, batch, ewcData, conf['ewc_clip_to']) 
            solver.net.blobs['ewc'].data[...] = ewcData  # makes ewc info available to the network for successive training

        # conver label to float32
        batch_y = batch_y.astype(np.float32)
        assert(batch_x.shape[0] == batch_y.shape[0])
        rehe_y = rehe_y.astype(np.float32)
        
        # training
        avg_train_loss = 0
        avg_train_accuracy = 0
        avg_count = 0;
        
        if conf['strategy'] in ['syn','ar1']:
            syn.init_batch(solver.net, ewcData, synData)

        reharshal_size = conf["rehearsal_memory"] if batch > initial_batch else 0
        # proporzione pattern da batch e memoria esterna
        orig_in_minibatch = np.round(train_minibatch_size * batch_x.shape[0]/(batch_x.shape[0] + reharshal_size)).astype(np.int)
        reha_in_minibatch = train_minibatch_size - orig_in_minibatch

        print (' -> Current Batch: %d patterns, External Memory: %d patterns' % (batch_x.shape[0], reharshal_size))
        print (' ->   per minibatch (size %d): %d from current batch and %d from external memory' % (train_minibatch_size, orig_in_minibatch, reha_in_minibatch))

        # pad e shuffle dei due gruppi
        # original patterns
        batch_x, orig_iters_per_epoch = train_utils.pad_data_single(batch_x, orig_in_minibatch)
        batch_y, _ = train_utils.pad_data_single(batch_y, orig_in_minibatch)
        batch_t, _ = train_utils.pad_data_single(batch_t, orig_in_minibatch)
        batch_x, batch_y, batch_t = train_utils.shuffle_in_unison((batch_x, batch_y, batch_t), 0)  #shuffle
        # rehasal patterns
        reha_iters_per_epoch = 0
        if reharshal_size > 0:
            rehe_x, reha_iters_per_epoch = train_utils.pad_data_single(rehe_x, reha_in_minibatch)
            rehe_y, _ = train_utils.pad_data_single(rehe_y, reha_in_minibatch)
            rehe_t, _ = train_utils.pad_data_single(rehe_t, reha_in_minibatch)
            rehe_x, rehe_y, rehe_t = train_utils.shuffle_in_unison((rehe_x, rehe_y, rehe_t), 0)  #shuffle

        # potrebbero essere diversi tra loro per problemi di padding
        print (' ->   iterations per epoch (with padding): %d, %d (initial %d)' % (orig_iters_per_epoch, reha_iters_per_epoch, train_iterations_per_epoch[batch]))

        # the main solver loop (per batch)
        it = 0
        while it < train_iterations[batch]:

            # provide data to input layers (batch patterns)
            it_mod_orig = it % orig_iters_per_epoch
            orig_start = it_mod_orig * orig_in_minibatch
            orig_end = (it_mod_orig + 1) * orig_in_minibatch
            solver.net.blobs['data'].data[:orig_in_minibatch] = batch_x[orig_start:orig_end]
            solver.net.blobs['label'].data[:orig_in_minibatch] = batch_y[orig_start:orig_end]
            solver.net.blobs['target'].data[:orig_in_minibatch] = batch_t[orig_start:orig_end]
            
            # provide data to input layers (reharsal patterns)
            if reharshal_size > 0:
                it_mod_reha = it % reha_iters_per_epoch
                reha_start = it_mod_reha * reha_in_minibatch
                reha_end = (it_mod_reha + 1) * reha_in_minibatch
                solver.net.blobs['data'].data[orig_in_minibatch:] = rehe_x[reha_start:reha_end]
                solver.net.blobs['label'].data[orig_in_minibatch:] = rehe_y[reha_start:reha_end]
                solver.net.blobs['target'].data[orig_in_minibatch:] = rehe_t[reha_start:reha_end]

            if conf['strategy'] in ['ar1']:
                syn.pre_update(solver.net, ewcData, synData)

            # Explicit (step 1) - Attenzione appena creato il solver richiede sharing eplicito pesi con rete di Test (cerca solver.share_weights(solver.test_nets[0]))
            solver.net.clear_param_diffs()
            solver.net.forward()   # start=None, end=None
            if batch > 0 and conf['strategy'] in ['cwr+']:
                solver.net.backward(end = 'mid_fc7')  # retropropaga solo nell'output
            else: 
                if batch > 0 and 'rehearsal_stop_layer' in conf.keys():
                    solver.net.backward(end = conf['rehearsal_stop_layer'])  # retropropaga completo: start=None, end=None    
                else: solver.net.backward()  # retropropaga completo: start=None, end=None    
			# Aggiorna i pesi
            solver.apply_update()

            if conf['strategy'] == 'ar1':
                syn.post_update(solver.net, ewcData, synData, cwr_layers_Model[conf['model']])

            print ('+', end = '', flush=True)

            global_eval_iter +=1
            avg_count +=1

            # activation stats
            # if batch == 100 and it >= 72: # last epoch
            #    train_utils.stats_activations_update(solver.net, nonzero_activations)
            
            # visualize activations
            #if batch == 2 and it == 10:
            #    layer = 'conv5_4/dw'
            #    print("Activation shape at layer: ", layer, " = ", solver.net.blobs[layer].data.shape)
            #    visualization.ShowFeatureMaps(solver.net, 'conv5_4/dw', number = None)
    
            avg_train_loss += solver.net.blobs['loss'].data
            if report_train_accuracy:
                avg_train_accuracy += solver.net.blobs['accuracy'].data

            if global_eval_iter == eval_iters[eval_idx]:    # evaluation point ?
                if avg_count > 0:          
                    avg_train_loss/= avg_count
                    avg_train_accuracy /= avg_count
                train_loss[eval_idx] = avg_train_loss
                print ('\nIter {:>4}'.format(it+1), '({:>4})'.format(global_eval_iter), ': Train Loss = {:.5f}'.format(avg_train_loss), end='', flush = True)
                if report_train_accuracy:
                    train_acc[eval_idx] = avg_train_accuracy
                    print ('  Train Accuracy = {:.5f}%'.format(avg_train_accuracy*100), end='', flush = True)

                compute_confusion_matrix = True if (conf['confusion_matrix'] and it == train_iterations[batch]-1) else False   # last batch iter

                if conf['strategy'] in ['cwr+', 'ar1'] and it == train_iterations[batch]-1:
                    #y_freq = np.ones_like(unique_y)  # forza tutte le frequenze nel batch a 1   
                    cwr.consolidate_weights_cwr_plus(solver.net, cwr_layers_Model[conf['model']], unique_y, y_freq, class_updates, cons_w)    
                    class_updates[unique_y] += y_freq;
                    #if batch == 0:  
                    #    class_updates = class_updates * 10  # aumenta la stabilità del primo batch
                    print(class_updates)
                    cwr.load_weights(solver.net, cwr_layers_Model[conf['model']], conf['num_classes'], cons_w)   # load consolidated weights for testing
                    
                accuracy, _ , pred_y = train_utils.test_network_with_accuracy_layer(solver, test_x, test_y, test_iterat, test_minibatch_size, prediction_level_Model[conf['model']], return_prediction = compute_confusion_matrix)
                test_acc[eval_idx] = accuracy*100
                print ('  Test Accuracy = {:.5f}%'.format(accuracy*100))

                # BatchNorm Stats
                train_utils.print_bn_stats(solver.net)

                visualization.Plot_Incremental_Training_Update(eval_idx, eval_iters, train_loss, test_acc)  
            
                filelog.Train_Log_Update(conf['train_log_file'], eval_iters[eval_idx], accuracy, avg_train_loss, report_train_accuracy, avg_train_accuracy)
               
                avg_train_loss = 0
                avg_train_accuracy = 0
                avg_count = 0
                eval_idx+=1   # next eval
        
            it+=1  # next iter
        
        # current batch training concluded
        if conf['strategy'] in ['ar1']:
            syn.update_ewc_data(solver.net, ewcData, synData, batch, conf['ewc_clip_to'])
            if conf['save_ewc_histograms']:
                visualization.EwcHistograms(ewcData, 100, save_as = conf['exp_path'] + 'Syn/F_' + str(batch) + '.png')

        rehearsal.update_memory(batch_x, batch_y.astype(np.int), batch)  
                
        if compute_confusion_matrix:   # last iter of the batch
            cnf_matrix = confusion_matrix(test_y, pred_y)
            if batch ==0:
                prev_class_accuracies = np.zeros(conf['num_classes'])
            else:
                prev_class_accuracies = current_class_accuracies
            current_class_accuracies = np.diagonal(cnf_matrix) / cnf_matrix.sum(axis = 1)
            deltas = current_class_accuracies - prev_class_accuracies
            classes_in_batch = set(batch_y.astype(np.int))
            classes_non_in_batch = set(range(conf['num_classes']))-classes_in_batch
            mean_class_in_batch = np.mean(deltas[list(classes_in_batch)])
            std_class_in_batch = np.std(deltas[list(classes_in_batch)])
            mean_class_non_in_batch = np.mean(deltas[list(classes_non_in_batch)])
            std_class_non_in_batch = np.std(deltas[list(classes_non_in_batch)])
            print('InBatch -> mean =  %.2f%% std =  %.2f%%, OutBatch -> mean =  %.2f%% std =  %.2f%%' % (mean_class_in_batch*100, std_class_in_batch*100, mean_class_non_in_batch*100, std_class_non_in_batch*100))
            filelog.Train_LogDetails_Update(conf['train_log_file'], batch, mean_class_in_batch, std_class_in_batch, mean_class_non_in_batch, std_class_non_in_batch)
            visualization.plot_confusion_matrix(cnf_matrix, normalize = True, title='CM after batch: ' + str(batch), save_as = conf['exp_path'] + 'CM/CM_' + str(batch) + '.png')

        if conf['compute_param_stats']:
            train_utils.stats_compute_param_change_and_update_prev(solver.net, param_stats, batch, param_change)
            #if batch == 100:
            #    train_utils.stats_activations_normalize(solver.net, nonzero_activations, "D:/PRJ/CNN/Experiments/Core50/NIC/activ_stats_%s.txt" % batch)
            #    train_utils.stats_activations_initialize(solver.net)                

        # save model weights at the end of batch 0
        if batch == 0:
            solver.net.save(conf['tmp_weights_file'])
            print('Weights saved to: ', conf['tmp_weights_file'])
            del solver

    print('Training Time: %.2f sec' % (time.time() - start_train))

    if conf['compute_param_stats']:
        stats_normalization = True
        train_utils.stats_normalize(solver.net, param_stats, batch_count, param_change, stats_normalization)
        visualization.Plot3d_param_stats(solver.net, param_change, batch_count, stats_normalization)

    filelog.Train_Log_End(conf['train_log_file'])
    filelog.Train_LogDetails_End(conf['train_log_file']) 

    visualization.Plot_Incremental_Training_End(close = close_at_the_end)    


def main_Core50_multiRun(conf, runs):
    
    conf['confusion_matrix'] = True
    conf['save_ewc_histograms'] = False
    conf['compute_param_stats'] = False
    allfile = open(conf['train_log_file']+'All.txt', 'w')
    for r in range(runs):    
        run = 'run'+str(r)
        main_Core50(conf, run, close_at_the_end = True)
        runres = filelog.LoadAccuracyFromCurTrainingLog(conf['train_log_file'])
        for item in runres:
            allfile.write("%s " % item)
        allfile.write("\n")
        allfile.flush()
    allfile.close()
         
if __name__ == "__main__":
    
    # --- NC Scenario ------------------------------------------------------
    conf_NC = {
        'model': 'CaffeNet',  # 'CaffeNet', 'Nin', 'GoogleNet'
        'db_path': 'D:/DB/Core50/128/',   # location of patterns and filelists
        'class_labels': 'D:/DB/Core50/core50_labels.txt',  
        'exp_path': 'D:/PRJ/CNN/Experiments/Core50/NC/',   # location of snapshots, temp binary database, logfiles, etc.
        'solver_file_first_batch': 'D:/PRJ/CNN/Experiments/Core50/NC/NC_solver_X_first_batch.prototxt',    
        'solver_file': 'D:/PRJ/CNN/Experiments/Core50/NC/NC_solver_X.prototxt',
        'init_weights_file': 'D:/PRJ/CNN/CaffeModels/ImageNetPretrained/X.caffemodel', 
        'tmp_weights_file': 'D:/PRJ/CNN/Experiments/Core50/NC/X.caffemodel', 
        'train_filelists': 'D:/DB/Core50/batches_filelists/NC_inc/RUN_X/train_batch_XX_filelist.txt',
        'test_filelist': 'D:/DB/Core50/batches_filelists/NC_inc/RUN_X/test_filelist.txt',
        'train_log_file': 'D:/PRJ/CNN/Experiments/Core50/NC/trainLog', # 'Cur.txt' is appended to create the file. 'Pre.txt' is appended when searching an old file for (optional) comparison.
        'num_classes': 50, 
        'num_batches': 9,  # including first one (all -> 9)
        'initial_batch': 0, # valid values: (0, 1). 0 include initial tuning from ImageNet, 1 start from previously saved model after 0
        'num_epochs_first_batch': 2.0,  # training epochs for first batch (typically higher)
        'num_epochs': 2.0,   # training epochs for all the other batches
        'strategy': 'ewc', # 'naive','lwf','ewc','syn','cwr','cwr+'
        'lwf_weight': -1,  # if -1, computed according to pattern proportions in batches (-1 CaffeNet, 0.75 GoogleNet)
        'ewc_clip_to': 0.001,  # max value for fisher matrix elements 
        'ewc_w': 2,  # additional param that premultiply F
        'cwr_batch0_weight': 1.25,   # CwR: CaffeNetMonco = 1.5, CaffeNetFull 1.25, GoogleNet = 1.0 (CwR+: 1.0 e 1.25 a meno di non sottrarre la media in quel caso peso a 1)
        #'strategy': 'SST_A_D',
        'backend': 'GPU',
        'accuracy_max': 0.8,   # for plot        
        'test_interval_epochs': 1.0,  # evaluation (and graphical plot) every (fraction of) batch epochs
        'dynamic_head_expansion': False,
        'confusion_matrix': True,
        'save_ewc_histograms': True,
        'compute_param_stats': True,
        'verbose': False
    }
    ## --------------------------------------------------------------------

    conf_NIC = {
        'model': 'MobileNetV1',  # 'CaffeNet', 'Nin', 'GoogleNet'
        'db_path': 'D:/DB/Core50/128/',   # location of patterns and filelists
        'class_labels': 'D:/DB/Core50/core50_labels.txt',  
        #'exp_path': 'E:/Temp/Core50/NIC/NIC_v2/',   # location of snapshots, temp binary database, logfiles, etc.
        #'exp_path': 'E:/Temp/Core50/NIC/NIC_v2_196/',   # location of snapshots, temp binary database, logfiles, etc.
        'exp_path': 'E:/Temp/Core50/NIC/NIC_v2_391/',   # location of snapshots, temp binary database, logfiles, etc.
        'solver_file_first_batch': 'D:/PRJ/CNN/Experiments/Core50/NIC/NIC_solver_X_first_batch.prototxt',    
        'solver_file': 'D:/PRJ/CNN/Experiments/Core50/NIC/NIC_solver_X.prototxt',
        'init_weights_file': 'D:/PRJ/CNN/CaffeModels/ImageNetPretrained/X.caffemodel', 
        'tmp_weights_file': 'E:/Temp/Core50/NIC/NIC/X.caffemodel',         
        'train_filelists': 'D:/DB/Core50/batches_filelists/NIC_v2_inc/RUN_X/train_batch_XX_filelist.txt',
        'test_filelist': 'D:/DB/Core50/batches_filelists/NIC_v2_inc/RUN_X/test_filelist_20.txt',
        #'train_filelists': 'D:/DB/Core50/batches_filelists/NIC_v2_inc_196/RUN_X/train_batch_XX_filelist.txt',
        #'test_filelist': 'D:/DB/Core50/batches_filelists/NIC_v2_inc_196/RUN_X/test_filelist_20.txt',
        #'train_filelists': 'D:/DB/Core50/batches_filelists/NIC_v2_inc_391/RUN_X/train_batch_XX_filelist.txt',
        #'test_filelist': 'D:/DB/Core50/batches_filelists/NIC_v2_inc_391/RUN_X/test_filelist_20.txt',
        'train_log_file': 'D:/PRJ/CNN/Experiments/Core50/NIC/trainLog', # 'Cur.txt' is appended to create the file. 'Pre.txt' is appended when searching an old file for (optional) comparison.
        'num_classes': 50, 
        'num_batches': 79,  # including first one (all -> 79)
        #'num_batches': 196,  # including first one (all -> 196)
        #'num_batches': 391,  # including first one (all -> 391)
        'initial_batch': 0, # valid values: (0, 1). 0 include initial tuning from ImageNet, 1 start from previously saved model after 0
        'num_epochs_first_batch': 4.0,  # training epochs for first batch (typically higher)
        'num_epochs': 4.0,  # training epochs for all the other batches
        'strategy': 'ar1',  # 'naive','lwf','ewc'
        'lwf_weight': 0.1,  # Weight of previous training. If -1, computed according to pattern proportions in batches  (0.1?)
        'ewc_clip_to': 0.001,  # max value for fisher matrix elements
        'ewc_w': 1,  # additional param that premultiply F
        'cwr_batch0_weight': 2.0,   # CwR: CaffeNetMonco = 1.5, CaffeNetFull 1.25, GoogleNet = 1.0 (CwR+: 1.0 e 1.25 a meno di non sottrarre la media in quel caso peso a 1)
        'cwr_nic_load_weight': True,
        'cwr_lr_mult': 10,  # moltiplica per questo valore il LR nei livelli CWR
        'rehearsal': True,
        'rehearsal_memory': 1500,
        #'rehearsal_stop_layer': 'conv5_4/sep', # reharsal pattern @ prev_level (/dw)
        #'strategy': 'SST_A_D',
        'backend': 'GPU',
        'accuracy_max': 0.8,   # for plot        
        'test_interval_epochs': 4.0,  # evaluation (and graphical plot) every (fraction of) batch epochs
        'dynamic_head_expansion': False,
        'confusion_matrix': True,
        'save_ewc_histograms': True,
        'compute_param_stats': True,
        'verbose': False
    }    

    import nic_aggregate

    ## --------------------------------------------------------------------

    prediction_level_Model = {
        'CaffeNet': 'mid_fc8',
        'Nin': 'pool4',
        'GoogleNet': 'loss3_50/classifier',
        'MobileNetV1': 'mid_fc7'
    }

    cwr_layers_Model = {
        'CaffeNet': ['mid_fc8'], 
        'GoogleNet': ['loss1_50/classifier', 'loss2_50/classifier', 'loss3_50/classifier'],
        'MobileNetV1': ['mid_fc7'],
    }

    latent_reharshal_Model = {
        'MobileNetV1': 'pool6'
    }

    #conf = conf_Cumulative
    #conf = conf_NC
    #conf = conf_NI
    conf = conf_NIC
    #conf = nic_aggregate.conf_NIC2
    
    # Setting hardware
    if conf['backend'] == 'GPU':
       caffe.set_device(0)
       caffe.set_mode_gpu()
    else:
       caffe.set_mode_cpu()    

    assert conf['strategy'] in ['naive','cwr+','ar1'], "Strategy " + conf['strategy'] + " not supported!"
    assert conf['rehearsal'], "Rehearsal must be enabled!"

    # Single Run
    sys.exit(int(main_Core50(conf, 'run0') or 0))

    # Multi Run
    # runs = 10   # always 10 except for cumulative -> 5
    # sys.exit(int(main_Core50_multiRun(conf, runs) or 0))


# NOTA
# per aumentare epoche per certio batch modificare alla linea 84: train_iterations[batch] = train_iterations_per_epoch[batch] * conf['num_epochs']

#Creating EwC data for Optimal params and their Fisher info
#Layer conv1: Weight 34848, Bias 96
#Layer conv2: Weight 307200, Bias 256
#Layer conv3: Weight 884736, Bias 384
#Layer conv4: Weight 663552, Bias 384
#Layer conv5: Weight 442368, Bias 256
#Layer mid_fc6: Weight 18874368, Bias 2048
#Layer mid_fc7: Weight 4194304, Bias 2048
#Layer mid_fc8: Weight 102400, Bias 50
#Total size 25509298  

#Per MobileNetV1
# Total size 3280141

# NOTE per TUNING degli ALGORITMI
# 1 - impostare numero di epoche per primo batch 'num_epochs_first_batch' e successivi batch 'num_epochs' in CONF 
# 2 - tarare il learning rate 'base_lr' nel file del solver del primo batch
# 3 - tarare il learning rate 'base_lr' nel file del solver del batch successivi
# 4 - per tutti gli algoritmi TRANNE EwC, Syn, Ar1 commentare 'L2ewc' e 'momentum2' nel solver dei batch successivi
# 5.a -    per Naive non ci sono iperparametri 
# 5.b -    per LwF parametro 'lwf_weight' in CONF, ovvero peso dei batch successivi. In genere lasciare a -1 (automatico, in base a cardinalità punti nuovi e passati). 
# 5.c -    per Ewc attivare 'momentum2' e 'L2ewc' nel solver dei batch successivi, togliendo commento. 
#             'ewc_clip_to' in CONF: lasciare come impostato
#             'momentum2': maggior è questo valore, più i parametri sono bloccati. Attenzione perchè l'entità del blocco è data da 'base_lr'*'momentum2' per cui aumentando il base_lr bisognerebbe calare questo
# 5.d -    per Syn tarare come per Ewc, ma agire anche sul valore di C nel file syn.py (C amplifica/comprime i valori in F, un criterio è quello di far saturare similarmente a EwC una frazione ragionavole di parametro es. 1e4, confrontare istrogrammi di F)
# 5.e -    per CwR solo iperparametro 'cwr_batch0_weight'
# 5.f -    per CwR+ nessun iperparametro. Per abilitare in NIC la versione "L" booleano 'cwr_nic_load_weight' in CONF   
# 5.g -    per Ar1 tarare come per SyN e CWR+ (attenzione a "L")


