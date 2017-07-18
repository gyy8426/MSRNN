from utils import *
import copy
import numpy as np

class Predict(object):

    def build_sampler(self, layers, tparams, options, use_noise, trng, word2vec_Wemb = None):
        debug_print = []
        #debug_print.append( theano.printing.Print('input_p.shapa')(input_p.shape))
        # context: #annotations x dim
        ctx0 = T.matrix('ctx_sampler', dtype='float32')
        ctx_mask = T.vector('ctx_mask', dtype='float32')

        ctx_ = ctx0
        counts = ctx_mask.sum(-1)
        ctx_mean = ctx_.sum(0)/counts
        ctx_max = ctx_.max(0)
        ctx_ = ctx0.dimshuffle('x',0,1)
        ctx_input = ctx_mean
        # initial state/cell
        tu_init_state = [T.alloc(0., options['rnn_word_dim'])]
        tu_init_memory = [T.alloc(0., options['rnn_word_dim'])]
        mu_init_state = [T.alloc(0., options['rnn_cond_wv_dim'])]
        mu_init_memory = [T.alloc(0., options['rnn_cond_wv_dim'])]
        '''
        if options['smoothing'] :
            a_init_state = [T.alloc(0., options['latent_size_a'])]
            #a_init_memory = [T.alloc(0., options['latent_size_a'])]
        else :
            a_init_state = None
        '''
        z_init_state = [T.alloc(0., options['latent_size_z'])]
        mu_p_init = [T.alloc(0., options['latent_size_z'])]
        print 'Building f_init...',
        '''
        f_init = theano.function([ctx0, ctx_mask], [ctx0]+tu_init_state+tu_init_memory+
                         mu_init_state+mu_init_memory+
                         a_init_state+a_init_memory+
                         z_init_state+
                         mu_p_init, name='f_init',
                         on_unused_input='ignore',
                         profile=False)
        '''
        f_init = theano.function([ctx0, ctx_mask], [ctx0]+tu_init_state+tu_init_memory+
                                 mu_init_state+mu_init_memory+
                                 z_init_state+
                                 mu_p_init, name='f_init',
                                 on_unused_input='ignore',
                                 profile=False)
        print 'Done'

        x = T.vector('x_sampler', dtype='int64')

        tu_init_state = [T.matrix('tu_init_state', dtype='float32')]
        tu_init_memory = [T.matrix('tu_init_memory', dtype='float32')]
        mu_init_state = [T.matrix('mu_init_state', dtype='float32')]
        mu_init_memory = [T.matrix('mu_init_memory', dtype='float32')]
        '''
        if options['smoothing'] :
            a_init_state = [T.matrix('a_init_state', dtype='float32')]
            #a_init_memory = [T.matrix('a_init_memory', dtype='float32')]
        '''
        # if it's the first word, emb should be all zero 
        if options['word2vec']:
            emb = T.switch(x[:, None] < 0,
                       T.alloc(0., 1, word2vec_Wemb.shape[1]), word2vec_Wemb[x])    

        else :
            emb = T.switch(x[:, None] < 0,
                       T.alloc(0., 1, tparams['Wemb'].shape[1]), tparams['Wemb'][x])      
        # emb ff 
        
        emb_ff1 = layers.get_layer('ff')[1](tparams, emb,activ=options['nonlin_decoder'], 
                                                prefix="emb_ff1")
        #emb_ff2 = layers.get_layer('ff')[1](tparams, emb_ff1,activ=options['nonlin_decoder'],
        #                                        prefix='emb_ff2')
        emb_drop = layers.dropout_layer(emb_ff1, use_noise, trng)
        tu_gru = layers.get_layer('lstm')[1](options,tparams, emb, one_step=True,
                                              init_state=tu_init_state[0],
                                              init_memory=tu_init_memory[0],
                                              prefix='tu_rnn')
        #debug_print.append( theano.printing.Print('mu_init_state.shapa')(mu_init_state.shape))
        if options['att_fun'] == None:        
            mu_gru = layers.get_layer('lstm_cond')[1](options, tparams, tu_gru[0],
                                                   mask=None, context=ctx_input,
                                                   one_step=True,
                                                   init_state=mu_init_state[0],
                                                   init_memory=mu_init_memory[0],
                                                   trng=trng,
                                                   use_noise=use_noise,
                                                   prefix='mu_rnn')
        else :
            mu_gru = layers.get_layer('lstm_att')[1](options, tparams, tu_gru[0],
                                                   mask=None, context=ctx_,
                                                   one_step=True,
                                                   init_state=mu_init_state[0],
                                                   init_memory=mu_init_memory[0],
                                                   trng=trng,
                                                   use_noise=use_noise,
                                                   prefix='mu_rnn')
        tu_next_state = [tu_gru[0]]
        tu_next_memory = [tu_gru[1]]
        mu_next_state = [mu_gru[0]]
        mu_next_memory = [mu_gru[1]]
        proj_h = mu_gru[0]
        d_layer = proj_h
        if options['use_dropout']:
            d_drop_layer = layers.dropout_layer(d_layer, use_noise, trng)
        '''
        input_a_layer = T.concatenate([d_drop_layer, emb_drop], axis=1)
        if options['smoothing']:
            a_layer = layers.get_layer('gru_cond')[1](options, tparams, input_a_layer,one_step=True,
                                                 init_state=a_init_state[0],context=ctx_input,
                                                      prefix='a_rnn')
            
            #a_layer = layers.get_layer('lstm')[1](options, tparams, input_a_layer,one_step=True,
            #                         init_state=a_init_state[0],init_memory=a_init_memory[0]
            #                              prefix='a_rnn')
            
            #a_layer = a_layer[:, ::-1]
            a_next_state = [a_layer[0]]
            #a_next_memory = [a_layer[1]]
            input_a = a_layer[0]
        else:
            temp_a = layers.get_layer('ff')[1](options, tparams, input_a_layer,
                                                   prefix='a_layer_0')
            for i in range(options['flat_mlp_num'] - 1):
                temp_a = layers.get_layer('ff')[1](options, tparams, temp_a,
                                                       prefix='a_layer_' + str(i + 1))
            a_layer = temp_a
            input_a = a_layer
        #debug_print.append( theano.printing.Print('a_layer.shapa')(a_layer.shape))
        '''
        #################
        ###stochastic parts####
        #################

        # Define shared variables for quantities to be updated across batches (truncated BPTT)
        z_init = [T.matrix('z', dtype='float32')]
        mu_p_init = [T.matrix('mu_p_init', dtype='float32')]
        stochastic_layer = layers.stochastic_layer_onestep_noq(options,tparams,
                                                     input_p=d_drop_layer,#input_q=input_a,
                                                     z_init=z_init[0],mu_p_init=mu_p_init[0],
                                                     num_units=options['latent_size_z'],
                                                     unroll_scan=options['unroll_scan'],
                                                     use_mu_residual_q=options['use_mu_residual_q']
                                                     )

        z_layer = [stochastic_layer[0]]
        mean_prior_layer = [stochastic_layer[1]]
        log_var_prior_layer = stochastic_layer[2]
        '''
        mean_q_layer = stochastic_layer[3]
        log_var_q_layer = stochastic_layer[4]
        '''
        z_dropout_layer = layers.dropout_layer(z_layer[0], use_noise, trng)
        '''
        z_layer_shp = z_dropout_layer.shape
        z_layer_reshaped = z_dropout_layer.reshape([z_layer_shp[0]*z_layer_shp[1],
                                                    z_layer_shp[2]])
        d_layer_shp = d_drop_layer.shape
        d_layer_reshaped = d_drop_layer.reshape([d_layer_shp[0]*d_layer_shp[1],
                                                    d_layer_shp[2]])
        '''
        input_gen_ff = T.concatenate([d_drop_layer, z_dropout_layer], axis=1)
        '''
        gen_word_emb_ff = layers.get_layer('ff')[1](tparams, input_gen_ff, activ=options['nonlin_decoder'],
                                               prefix='gen_word_emb_ff')
        '''
        logit = layers.get_layer('ff')[1](tparams, input_gen_ff,
                                          prefix='ff_logit_zd', activ='linear')
        # logit_shp = logit.shape
        next_probs = T.nnet.softmax(logit)
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        # next word probability
        print 'building f_next...'
        '''
        f_next = theano.function([x, ctx0, ctx_mask]+
                         tu_init_state+tu_init_memory+
                         mu_init_state+mu_init_memory+
                         a_init_state+a_init_memory
                         z_init+
                         mu_p_init,
                         [next_probs, next_sample]+
                         tu_next_state+tu_next_memory+
                         mu_next_state+mu_next_memory+
                         a_next_state+a_next_memory+
                         z_layer+
                         mean_prior_layer,
                         name='f_next', profile=False,
                         on_unused_input='ignore')
        '''
        f_next = theano.function([x, ctx0, ctx_mask]+
                                 tu_init_state+tu_init_memory+
                                 mu_init_state+mu_init_memory+
                                 z_init+
                                 mu_p_init,
                                 [next_probs, next_sample]+
                                 tu_next_state+tu_next_memory+
                                 mu_next_state+mu_next_memory+
                                 z_layer+
                                 mean_prior_layer,
                                 name='f_next', profile=False,
                                 on_unused_input='ignore')
        print 'Done'
        return f_init, f_next

    def gen_sample(self, tparams, f_init, f_next, ctx0, ctx_mask,
                   trng=None, k=1, maxlen=30, stochastic=False):
        '''
        ctx0: (26,1024)
        ctx_mask: (26,)
        '''

        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling'

        sample = []
        z_res = []
        p_mean = []
        sample_score = []
        if stochastic:
            sample_score = 0

        live_k = 1
        dead_k = 0

        hyp_samples = [[]] * live_k
        hyp_z_res = [[]]* live_k
        hyp_p_meam= [[]]* live_k
        hyp_scores = np.zeros(live_k).astype('float32')
        # [(26,1024),(512,),(512,)]

        rval = f_init(ctx0, ctx_mask)
        ctx0 = rval[0]

        # next gru and stacked gru state and memory
        next_states = []
        next_memorys = []
        n_layers_rnn = 2
        n_rnn_return = 2
        for lidx in xrange(n_layers_rnn):
            next_states.append([])
            next_memorys.append([])
            next_states[lidx].append(rval[n_rnn_return*lidx+1])
            next_states[lidx][-1] = next_states[lidx][-1].reshape([live_k, next_states[lidx][-1].shape[0]])
            next_memorys[lidx].append(rval[n_rnn_return*lidx+2])
            next_memorys[lidx][-1] = next_memorys[lidx][-1].reshape([live_k, next_memorys[lidx][-1].shape[0]])
        #print "init gru state shape is ",len(next_states),',',len(next_states[0])

        '''
        next_a_state = []
        next_a_state.append([])
        next_a_state[0].append(rval[-3])
        next_a_state = []
        next_a_state.append([])
        next_a_state[0].append(rval[-4])
        next_a_memory = []
        next_a_memory.append([])
        next_a_memory[0].append(rval[-3])
        '''
        next_z = []
        next_z.append([])
        next_z[0].append(rval[-2])
        next_mu_p = []
        next_mu_p.append([])
        next_mu_p[0].append(rval[-1])
        #print "init next_mu_p shape is ",len(next_mu_p),',',len(next_mu_p[0]),','               
        next_w = -1 * np.ones((1,)).astype('int64')
        # next_state: [(1,512)]
        # next_memory: [(1,512)]
        for ii in xrange(maxlen):
            # return [(1, 50000), (1,), (1, 512), (1, 512)]
            # next_w: vector
            # ctx: matrix
            # ctx_mask: vector
            # next_state: [matrix]
            # next_memory: [matrix]
            #print "next_states ", len(next_states),',',len(next_states[1]),',',len(next_states[1][0]),',',len(next_states[1][0][0])            
            rval = f_next(*([next_w, ctx0, ctx_mask] +
                            next_states[0] + next_memorys[0] + 
                            next_states[1] + next_memorys[1] +
                            next_z +
                            next_mu_p))
            next_p = rval[0]
            next_w = rval[1] # already argmax sorted

            next_states = []
            next_memorys = []
            for lidx in xrange(n_layers_rnn):
                next_states.append([])
                next_memorys.append([])
                next_states[lidx].append(rval[n_rnn_return*lidx+2])
                next_memorys[lidx].append(rval[n_rnn_return*lidx+3])
            #print "gru state is ", len(next_states),',',len(next_states[0]),',',len(next_states[0][0])

            '''
            next_a_state = [rval[-3]]
            next_a_state = [rval[-4]]
            next_a_memory = [rval[-3]]
            '''
            next_z = [rval[-2]]
            next_mu_p = [rval[-1]]
            #print "init next_a shape is ",len(next_a),',',len(next_a[0]),','             
            #print "init next_mu_p shape is ",len(next_mu_p),',',len(next_mu_p[0]),','           
            if stochastic:
                sample.append(next_w[0]) # take the most likely one
                sample_score += next_p[0,next_w[0]]
                z_layer.append(next_z[0])
                if next_w[0] == 0:
                    break
            else:
                # the first run is (1,50000)
                cand_scores = hyp_scores[:,None] - np.log(next_p)
                cand_flat = cand_scores.flatten()
                ranks_flat = cand_flat.argsort()[:(k-dead_k)]

                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size # index of row
                word_indices = ranks_flat % voc_size # index of col
                costs = cand_flat[ranks_flat]

                new_hyp_samples = []
                new_hyp_scores = np.zeros(k-dead_k).astype('float32')
                new_hyp_z_res = []
                new_hyp_p_mean=[]
                new_hyp_states = []
                new_hyp_memories = []
                #new_hyp_a_state = []
                #new_hyp_a_state.append([])
                #new_hyp_a_memory = []
                #new_hyp_a_memory.append([])
                new_hyp_z = []
                new_hyp_z.append([])
                new_hyp_mu_p = []
                new_hyp_mu_p.append([])

                for lidx in xrange(n_layers_rnn):
                    new_hyp_states.append([])
                    new_hyp_memories.append([])
                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti]+[wi])
                    new_hyp_z_res.append(hyp_z_res[ti]+[next_z[0][ti]])
                    new_hyp_p_mean.append(hyp_p_meam[ti]+[next_mu_p[0][ti]])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    for lidx in np.arange(n_layers_rnn):
                        new_hyp_states[lidx].append(copy.copy(next_states[lidx][0][ti]))
                        new_hyp_memories[lidx].append(copy.copy(next_memorys[lidx][0][ti]))
                    #new_hyp_a_state[0].append( copy.copy(next_a_state[0][ti]))
                    #new_hyp_a_memory[0].append( copy.copy(next_a_memory[0][ti]))
                    new_hyp_z[0].append(copy.copy(next_z[0][ti]))
                    new_hyp_mu_p[0].append(copy.copy(next_mu_p[0][ti]))
                #print "init new_hyp_states shape is ",len(new_hyp_states),',',len(new_hyp_states[0]),','               
                #print "init new_hyp_mu_p shape is ",len(new_hyp_mu_p),',',len(new_hyp_mu_p[0]),','
                # check the finished samples
                new_live_k = 0
                hyp_samples = []
                hyp_z_res = []
                hyp_p_meam = []
                hyp_scores = []
                hyp_states = []
                hyp_a_state = []
                hyp_a_state.append([])
                hyp_a_memory = []
                hyp_a_memory.append([])
                hyp_z = []
                hyp_z.append([])
                hyp_mu_p = []
                hyp_mu_p.append([])
                hyp_memories = []
                for lidx in xrange(n_layers_rnn):
                    hyp_states.append([])
                    hyp_memories.append([])

                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        z_res.append(new_hyp_z_res[idx])
                        p_mean.append(new_hyp_p_mean[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_z_res.append(new_hyp_z_res[idx])
                        hyp_p_meam.append(new_hyp_p_mean[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        for lidx in xrange(n_layers_rnn):
                            hyp_states[lidx].append(new_hyp_states[lidx][idx])
                            hyp_memories[lidx].append(new_hyp_memories[lidx][idx])
                        #hyp_a_state[0].append(new_hyp_a_state[0][idx])
                        #hyp_a_memory[0].append(new_hyp_a_memory[0][idx])
                        hyp_z[0].append(new_hyp_z[0][idx])
                        hyp_mu_p[0].append(new_hyp_mu_p[0][idx])
                #print "init hyp_states shape is ",len(hyp_states),',',len(hyp_states[0]),','              
                #print "init hyp_mu_p shape is ",len(hyp_mu_p),',',len(hyp_mu_p[0]),','               
                hyp_scores = np.array(hyp_scores)
                live_k = new_live_k

                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break

                next_w = np.array([w[-1] for w in hyp_samples])
                next_states = []
                next_memorys = []
                for lidx in xrange(n_layers_rnn):
                    next_states.append([])
                    next_memorys.append([])
                    next_states[lidx].append(np.array(hyp_states[lidx]))
                    next_memorys[lidx].append(np.array(hyp_memories[lidx]))
                #next_a_state=hyp_a_state
                #next_a_memory=hyp_a_memory
                next_z = hyp_z
                #z_layer.append(next_z)
                next_mu_p = hyp_mu_p
                #print "init next_states shape is ",len(next_states),',',len(next_states[0]),',',len(next_states[0][0])                
                #print "init next_mu_p shape is ",len(next_mu_p),',',len(next_mu_p[0]),','
        if not stochastic:
            # dump every remaining one
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])
                    z_res.append([hyp_z[0][idx]])
                    p_mean.append([hyp_mu_p[0][idx]])
        '''        
        for i in np.arange(len(sample)):
            length = len(sample[i])
            sample_score[i] = sample_score[i]/(1.0 * length)
            '''
        return sample, sample_score, next_states, z_res,p_mean

    def sample_execute(self, engine, options, tparams, f_init, f_next, x, ctx, mask_ctx, trng):
        stochastic = False
        for jj in xrange(np.minimum(10, x.shape[1])):
            sample, score,  _,_,_ = self.gen_sample(tparams, f_init, f_next, ctx[jj], mask_ctx[jj],
                                                  trng=trng, k=5, maxlen=30, stochastic=stochastic)
            if not stochastic:
                best_one = np.argmin(score)
                sample = sample[best_one]
            else:
                sample = sample
            print 'Truth ', jj, ': ',
            for vv in x[:, jj]:
                if vv == 0:
                    break
                if vv in engine.ix_word:
                    print engine.ix_word[vv],
                else:
                    print 'UNK',
            print
            for kk, ss in enumerate([sample]):
                print 'Sample (', jj, ') ', ': ',
                for vv in ss:
                    if vv == 0:
                        break
                    if vv in engine.ix_word:
                        print engine.ix_word[vv],
                    else:
                        print 'UNK',
            print
