

def main_Core50(conf, run, close_at_the_end=False):


    for batch in range(initial_batch, batch_count):

        print('\nBATCH = {:2d} ----------------------------------------------------'.format(batch))

        if batch == 0:

            if conf['strategy'] in ['cwr+', 'ar1', 'ar1free']:

                cwr.zeros_cwr_layer_bias_lr(solver.net, cwr_layers_Model[conf['model']])
                class_updates = np.full(
                    conf['num_classes'], conf['initial_class_updates_value'], dtype=np.float32)
                cons_w = cwr.init_consolidated_weights(
                    solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])
                cwr.reset_weights(
                    solver.net, cwr_layers_Model[conf['model']], conf['num_classes'])

            if conf['strategy'] in ['ar1', 'ar1free']:
                ewcData, synData = syn.create_syn_data(solver.net)

            if conf['rehearsal_is_latent']:
                reha_data_size = solver.net.blobs[conf['rehearsal_layer']].data[0].size
                rehearsal.allocate_memory(conf['rehearsal_memory'], reha_data_size, 1)
            else:
                rehearsal.allocate_memory(conf['rehearsal_memory'], test_x[0].size, 1)

        elif batch == 1:

            # blocca livelli sotto
            if conf['strategy'] in ['cwr+']:
                cwr.zeros_non_cwr_layers_lr(
                    solver.net, cwr_layers_Model[conf['model']])

            if conf['strategy'] in ['cwr+', 'ar1', 'ar1free']:
                if 'cwr_lr_mult' in conf.keys() and conf['cwr_lr_mult'] != 1:
                    cwr.zeros_cwr_layer_bias_lr(
                        solver.net, cwr_layers_Model[conf['model']],
                        force_weights_lr_mult=conf['cwr_lr_mult'])
                else:
                    cwr.zeros_cwr_layer_bias_lr(
                        solver.net, cwr_layers_Model[conf['model']])

            cwr.set_brn_past_weight(solver.net, conf['brn_past_weight'])

        # Load patterns from Rehearsal Memory
        rehe_x, rehe_y = rehearsal.get_samples()
        rehe_t = train_utils.compute_one_hot_vectors(rehe_y, conf['num_classes'])

        if conf['strategy'] in ['cwr+', 'ar1', 'ar1free'] and batch > initial_batch:
            cwr.reset_weights(
                solver.net, cwr_layers_Model[conf['model']], conf['num_classes']
            )  # Reset weights of CWR layers to 0

            # Loads previously consolidated weights
            # This procedure, explained in Fine-Grained Continual Learning
            # (https://arxiv.org/pdf/1907.03799.pdf),
            # is necessary in the NIC scenario
            if 'cwr_nic_load_weight' in conf.keys() and conf['cwr_nic_load_weight']:
                cwr.load_weights_nic(
                    solver.net, cwr_layers_Model[conf['model']], unique_y, cons_w
                )

        if conf['strategy'] in ['ar1'] and batch > initial_batch:
            syn.weight_stats(solver.net, batch, ewcData, conf['ewc_clip_to'])
            solver.net.blobs['ewc'].data[...] = ewcData

        if conf['strategy'] in ['syn', 'ar1']:
            syn.init_batch(solver.net, ewcData, synData)

        reharshal_size = conf["rehearsal_memory"] if batch > initial_batch else 0
        orig_in_minibatch = np.round(train_minibatch_size * batch_x.shape[0] /
                            (batch_x.shape[0] + reharshal_size)).astype(np.int)
        reha_in_minibatch = train_minibatch_size - orig_in_minibatch

        if conf['rehearsal_is_latent']:
            req_shape = (batch_x.shape[0],) + solver.net.blobs[conf['rehearsal_layer']].data.shape[1:]
            latent_batch_x = np.zeros(req_shape, dtype=np.float32)

        # Padding and shuffling of rehasal patterns
        reha_iters_per_epoch = 0
        if reharshal_size > 0:
            rehe_x, reha_iters_per_epoch = train_utils.pad_data_single(rehe_x, reha_in_minibatch)
            rehe_y, _ = train_utils.pad_data_single(rehe_y, reha_in_minibatch)
            rehe_t, _ = train_utils.pad_data_single(rehe_t, reha_in_minibatch)
            rehe_x, rehe_y, rehe_t = train_utils.shuffle_in_unison((rehe_x, rehe_y, rehe_t), 0)  # shuffle

        # The main solver loop (per batch)
        it = 0
        while it < train_iterations[batch]:

            it_mod_orig = it % orig_iters_per_epoch
            orig_start = it_mod_orig * orig_in_minibatch
            orig_end = (it_mod_orig + 1) * orig_in_minibatch

            if conf['rehearsal_is_latent']:
                solver.net.blobs['data'].data[...] = batch_x[orig_start:orig_end]
            else:
                solver.net.blobs['data'].data[:orig_in_minibatch] = batch_x[orig_start:orig_end]

            # Provide data to input layers (new patterns)
            solver.net.blobs['label'].data[:orig_in_minibatch] = batch_y[orig_start:orig_end]
            solver.net.blobs['target'].data[:orig_in_minibatch] = batch_t[orig_start:orig_end]

            # Provide data to input layers (reharsal patterns)
            if reharshal_size > 0:
                it_mod_reha = it % reha_iters_per_epoch
                reha_start = it_mod_reha * reha_in_minibatch
                reha_end = (it_mod_reha + 1) * reha_in_minibatch

                if conf['rehearsal_is_latent']:
                    solver.net.blobs['data_reha'].data[...] = rehe_x[reha_start:reha_end]
                else:
                    solver.net.blobs['data'].data[orig_in_minibatch:] = rehe_x[reha_start:reha_end]

                solver.net.blobs['label'].data[orig_in_minibatch:] = rehe_y[reha_start:reha_end]
                solver.net.blobs['target'].data[orig_in_minibatch:] = rehe_t[reha_start:reha_end]

            if conf['strategy'] in ['ar1']:
                syn.pre_update(solver.net, ewcData, synData)

            # Explicit (net.step(1))
            solver.net.clear_param_diffs()
            solver.net.forward()  # start=None, end=None
            if batch > 0 and conf['strategy'] in ['cwr+', 'cwr']:
                solver.net.backward(end='mid_fc7')  # In CWR+ we stop the backward step at the CWR layer
            else:
                if batch > 0 and 'rehearsal_stop_layer' in conf.keys() and \
                        conf['rehearsal_stop_layer'] is not None:
                    # When using latent replay we stop the backward step at the latent rehearsal layer
                    solver.net.backward(end=conf['rehearsal_stop_layer'])
                else:
                    solver.net.backward()

            if conf['rehearsal_is_latent']:
                # Save latent features of new patterns (only during the first epoch)
                if batch > 0 and it < orig_iters_per_epoch:
                    latent_batch_x[orig_start:orig_end] = solver.net.blobs[conf['rehearsal_layer']].data

            # Weights update
            solver.apply_update()

            if conf['strategy'] == 'ar1':
                syn.post_update(solver.net, ewcData, synData, cwr_layers_Model[conf['model']])

            global_eval_iter += 1
            avg_count += 1

            avg_train_loss += solver.net.blobs['loss'].data
            if report_train_accuracy:
                avg_train_accuracy += solver.net.blobs['accuracy'].data

                # The following lines are executed only if this is the last iteration for the current batch
                if conf['strategy'] in ['cwr+', 'ar1', 'ar1free'] and it == train_iterations[batch] - 1:
                    cwr.consolidate_weights_cwr_plus(
                        solver.net, cwr_layers_Model[conf['model']], unique_y, y_freq, class_updates, cons_w)
                    class_updates[unique_y] += y_freq
                    cwr.load_weights(
                        solver.net, cwr_layers_Model[conf['model']],
                        conf['num_classes'], cons_w) # Load consolidated weights for testing

        # Current batch training concluded
        if conf['strategy'] in ['ar1']:
            syn.update_ewc_data(solver.net, ewcData, synData, batch, conf['ewc_clip_to'], c=conf['ewc_w'])
            if conf['save_ewc_histograms']:
                visualization.EwcHistograms(ewcData, 100, save_as=conf['exp_path'] + 'Syn/F_' + str(batch) + '.png')

        if conf['rehearsal_is_latent']:
            if batch == 0:
                reha_it = 0
                while reha_it < orig_iters_per_epoch:
                    orig_start = reha_it * orig_in_minibatch
                    orig_end = (reha_it + 1) * orig_in_minibatch
                    solver.net.blobs['data'].data[...] = batch_x[orig_start:orig_end]
                    solver.net.forward()
                    latent_batch_x[orig_start:orig_end] = solver.net.blobs[conf['rehearsal_layer']].data
                    reha_it+=1

            rehearsal.update_memory(latent_batch_x, batch_y.astype(np.int), batch)
        else:
            rehearsal.update_memory(batch_x, batch_y.astype(np.int), batch)