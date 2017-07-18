from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import numpy
import sys
import time
from layers import Layers
import data_engine_res
import data_engine_googlenet
import metrics
from optimizers import *
from predict import *
import theano.tensor as T 
from parmesan.distributions import log_bernoulli
from decay import *
import math
from sklearn.preprocessing import minmax_scale
import cPickle
data_engine =  data_engine_googlenet
def validate_options(options):
    if options['ctx2out']:
        warnings.warn('Feeding context to output directly seems to hurt.')
    if options['dim_word'] > options['rnn_cond_wv_dim']:
        warnings.warn('dim_word should only be as large as rnn_cond_wv_dim.')
    return options

c = - 0.5 * math.log(2 * math.pi)
def log_normal2(x, mean, log_var):
    return c - log_var / 2 - (x - mean) ** 2 / (2 * T.exp(log_var))

def kl_normal2_normal2(mean1, log_var1, mean2, log_var2):
    return 0.5 * log_var2 - 0.5 * log_var1 + (T.exp(log_var1) + (mean1 - mean2) ** 2) / (2 * T.exp(log_var2)) - 0.5
class Test():
    def __init__(self):
        self.rng_numpy, self.rng_theano = get_two_rngs()
        self.layers = Layers()
        self.predict = Predict()

    def load_params(self, path, params):
        # load params from disk
        pp = np.load(path)
        for kk, vv in params.iteritems():
            params[kk] = pp[kk]
    def init_params(self, options):
        # all parameters
        params = OrderedDict()
        # embedding
        params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

        ctx_dim = options['ctx_dim']
        print "ctx_dim is ", ctx_dim
        # emb ff 

        params = self.layers.get_layer('ff')[0]( params, nin = options['dim_word'], 
                                                nout=options['dim_word'],
                                                prefix='emb_ff1')
        '''
        params = self.layers.get_layer('ff')[0]( params, nin = options['dim_word'], 
                                                nout=options['dim_word'],
                                                prefix='emb_ff2')
        '''
        # decoder: gru
        params = self.layers.get_layer('lstm')[0](options,params, nin=options['dim_word'],
                                                  dim=options['rnn_word_dim'], prefix='tu_rnn')
        if options['att_fun'] == None :
            print "murnn using lstm cond" 
            params = self.layers.get_layer('lstm_cond')[0](options, params, nin=options['rnn_word_dim'],
                                                       dim=options['rnn_cond_wv_dim'], dimctx=ctx_dim,
                                                       prefix='mu_rnn')
        else :
            print "murnn using lstm att" 
            params = self.layers.get_layer('lstm_att')[0](options, params, nin=options['rnn_word_dim'],
                                                       dim=options['rnn_cond_wv_dim'], dimctx=ctx_dim,
                                                       prefix='mu_rnn')

        
        if options['smoothing'] is True :
            if options['a_layer_type'] == 'lstm' :
                a_rnn_indim=options['dim_word']+options['rnn_cond_wv_dim']
                params = self.layers.get_layer('lstm')[0](options, params, nin=a_rnn_indim,
                                                     dim=options['latent_size_a'], 
                                                     prefix='a_rnn')
            elif options['a_layer_type'] == 'gru' :
                a_rnn_indim=options['dim_word']+options['rnn_cond_wv_dim']
                params = self.layers.get_layer('gru')[0](options, params, nin=a_rnn_indim,
                                                     dim=options['latent_size_a'], 
                                                     prefix='a_rnn')            
            elif options['a_layer_type'] == 'lstm_cond' :
                a_rnn_indim = options['dim_word']
                a_rnn_ctxdim = options['rnn_cond_wv_dim']
                params = self.layers.get_layer('lstm_cond')[0](options, params, nin=a_rnn_indim,
                                                     dim=options['latent_size_a'], dimctx=a_rnn_ctxdim,
                                                     prefix='a_rnn')
            elif options['a_layer_type'] == 'gru_cond' :
                a_rnn_indim = options['dim_word']
                a_rnn_ctxdim = options['rnn_cond_wv_dim']
                params = self.layers.get_layer('gru_cond')[0](options, params, nin=a_rnn_indim,
                                                     dim=options['latent_size_a'], dimctx=a_rnn_ctxdim,
                                                     prefix='a_rnn')            
        else :
            params = self.layers.get_layer('ff')[0]( params, nin = a_rnn_indim, 
                                                        nout=options['latent_size_a'],
                                                        prefix='a_layer_0')
            for i in range(options['flat_mlp_num']-1) :
                params = self.layers.get_layer('ff')[0](params, nin=options['latent_size_a'],
                                                        nout=options['latent_size_a'],
                                                        prefix='a_layer_'+str(i+1))
        ###Init stochastic parts####
        params = self.layers.get_layer('stochastic')[0](options, params)
        '''
        params = self.layers.get_layer('ff')[0](params,nin=options['latent_size_z']+options['rnn_cond_wv_dim'],nout=options['dim_word'],
                                                prefix = 'gen_word_emb_ff')
        params = self.layers.get_layer('ff')[0](params,nin=options['dim_word'],nout=options['dim_word'],
                                                prefix = 'mean_gen_word_emb')
        params = self.layers.get_layer('ff')[0](params,nin=options['dim_word'],nout=options['dim_word'],
                                                prefix = 'var_gen_word_emb')
        '''
        # readout
        ff_logit_zd_nin = options['latent_size_z'] + options['dim_word']
        params = self.layers.get_layer('ff')[0](params, nin=ff_logit_zd_nin, nout=options['n_words'],
                                                prefix='ff_logit_zd')

        return params

    def build_model(self, tparams, options):
        debug_print = []
        #debug_print.append( theano.printing.Print('input_a_layer.shapa')(input_a_layer.shape)) 
        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
        # description string: #words x #samples
        x = tensor.matrix('x', dtype='int64')
        mask = tensor.matrix('mask', dtype='float32')
        # context: #samples x #annotations x dim
        ctx = tensor.tensor3('ctx', dtype='float32')
        mask_ctx = tensor.matrix('mask_ctx', dtype='float32')

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        # index into the word embedding matrix, shift it forward in time
        emb = tparams['Wemb'][x.flatten()].reshape(
                [n_timesteps, n_samples, options['dim_word']])
        emb_before = emb
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])
        emb = emb_shifted
        ctx_ = ctx
        counts = mask_ctx.sum(-1).dimshuffle(0,'x')
        ctx_mean = ctx_.sum(1)/counts
        ctx_max = ctx_.max(axis = 1)
        ctx_input = ctx_mean
        # emb ff 
        emb_before_ff1 = self.layers.get_layer('ff')[1](tparams, emb_before,activ=options['nonlin_decoder'], 
                                                prefix="emb_ff1")
        emb_before_drop = self.layers.dropout_layer(emb_before_ff1, use_noise, trng)
        emb_ff1 = self.layers.get_layer('ff')[1](tparams, emb,activ=options['nonlin_decoder'], 
                                                prefix="emb_ff1")
        #emb_ff2 = self.layers.get_layer('ff')[1](tparams, emb_ff1,activ=options['nonlin_decoder'],
        #                                       prefix='emb_ff2')
        emb_drop = self.layers.dropout_layer(emb_ff1, use_noise, trng)
        # decoder
        tu_rnn = self.layers.get_layer('lstm')[1](options,tparams, emb, mask=mask, prefix='tu_rnn')
        if options['att_fun'] == None  :
            print "murnn using lstm cond" 
            mu_rnn = self.layers.get_layer('lstm_cond')[1](options, tparams, tu_rnn[0],
                                                        mask=mask, context=ctx_input,
                                                        one_step=False,
                                                        trng=trng,
                                                        use_noise=use_noise,
                                                        prefix='mu_rnn')        
        else :
            mu_rnn = self.layers.get_layer('lstm_att')[1](options, tparams, tu_rnn[0],
                                                        mask=mask, context=ctx_,
                                                        one_step=False,
                                                        trng=trng,
                                                        use_noise=use_noise,
                                                        prefix='mu_rnn')
                                                                
        proj_h = mu_rnn[0]
        d_layer = proj_h
        if options['use_dropout']:
            d_drop_layer = self.layers.dropout_layer(d_layer, use_noise, trng)

        
        if options['smoothing'] :
            if options['a_layer_type'] == 'lstm' :
                input_a_layer=T.concatenate([d_drop_layer,emb_before],axis=2)
                input_a_layer = input_a_layer[::-1]
                a_layer = self.layers.get_layer('lstm')[1](options, tparams, input_a_layer,mask=mask,
                                                        prefix='a_rnn')
            elif options['a_layer_type'] == 'gru' :
                input_a_layer=T.concatenate([d_drop_layer,emb_before],axis=2)
                input_a_layer = input_a_layer[::-1]
                a_layer = self.layers.get_layer('gru')[1](options, tparams, input_a_layer,mask=mask,
                                                        prefix='a_rnn')
            elif options['a_layer_type'] == 'lstm_cond' :
                input_a_layer = emb_before[::-1]
                input_ctx_a_layer = d_drop_layer[::-1]
                a_layer = self.layers.get_layer('lstm_cond')[1](options, tparams, input_a_layer,mask=mask,context = input_ctx_a_layer,
                                                        prefix='a_rnn')
            elif options['a_layer_type'] == 'gru_cond' :
                input_a_layer = emb_before[::-1]
                input_ctx_a_layer = d_drop_layer[::-1]
                a_layer = self.layers.get_layer('gru_cond')[1](options, tparams, input_a_layer,mask=mask,context = input_ctx_a_layer,
                                                        prefix='a_rnn')
            a_layer=a_layer[0][::-1]
            input_a = a_layer
        else :
            temp_a=self.layers.get_layer('ff')[1]( tparams, input_a_layer,activ='linear',
                                                    prefix='a_layer_0')
            for i in range(options['flat_mlp_num']-1) :
                temp_a=self.layers.get_layer('ff')[1]( tparams, temp_a,activ='linear',
                                                    prefix='a_layer_'+str(i+1))
            a_layer = temp_a
            input_a = a_layer
        #debug_print.append( theano.printing.Print('\n a_layer info \n')(a_layer)) 
        #################
        ###stochastic parts####
        #################

        # Define shared variables for quantities to be updated across batches (truncated BPTT)
        #debug_print.append( theano.printing.Print('\n num sample  info \n')(n_samples)) 
        self.z_init_sh = theano.shared(np.zeros((options['batch_size'], options['latent_size_z']),                                    
                                                dtype=theano.config.floatX))        
        self.mean_prior_init_sh = theano.shared(np.zeros((options['batch_size'], options['latent_size_z']),
                                                         dtype=theano.config.floatX))
        self.log_var_prior_init_sh = theano.shared(np.zeros((options['batch_size'], options['latent_size_z']),
                                                   dtype=theano.config.floatX))
        #z_init = tensor.matrix('z', dtype='float32')
        #mu_p_init = tensor.matrix('mu_p_init',dtype='float32')
        #mask_input = mask_ctx
        stochastic_layer = self.layers.get_layer('stochastic')[1](options,tparams,
                                                     input_p=d_drop_layer,input_q=input_a,
                                                     z_init=self.z_init_sh,
                                                     mu_p_init=self.mean_prior_init_sh,
                                                     mask_input=mask,
                                                     num_units=options['latent_size_z'],
                                                     unroll_scan=options['unroll_scan'],
                                                     use_mu_residual_q=options['use_mu_residual_q']
                                                     )
                                                      
        z_layer = stochastic_layer[0]
        mean_prior_layer = stochastic_layer[1]
        log_var_prior_layer = stochastic_layer[2]
        mean_q_layer = stochastic_layer[3]
        log_var_q_layer = stochastic_layer[4]
        #debug_print.append( theano.printing.Print('z_layer.info : \n')(z_layer))  
        #debug_print.extend(stochastic_layer[5])
        z_dropout_layer = self.layers.dropout_layer(z_layer,use_noise,trng)
        '''
        z_layer_shp = z_dropout_layer.shape
        z_layer_reshaped = z_dropout_layer.reshape([z_layer_shp[0]*z_layer_shp[1],
                                                    z_layer_shp[2]])
        d_layer_shp = d_drop_layer.shape
        d_layer_reshaped = d_drop_layer.reshape([d_layer_shp[0]*d_layer_shp[1],
                                                    d_layer_shp[2]])
        '''
        input_gen_ff = T.concatenate([d_drop_layer,z_dropout_layer],axis=2)
        '''
        gen_word_emb_ff = self.layers.get_layer('ff')[1](tparams, input_gen_ff, activ=options['nonlin_decoder'],
                                               prefix='gen_word_emb_ff')
        ##mean and var ##                                       
        mean_gen_word_emb = self.layers.get_layer('ff')[1](tparams, gen_word_emb_ff, activ='linear',
                                               prefix='mean_gen_word_emb')
        mean_gen_word_emb = mean_gen_word_emb.dimshuffle(1,0,2)
        var_gen_word_emb = self.layers.get_layer('ff')[1](tparams, gen_word_emb_ff, activ='linear',
                                               prefix='var_gen_word_emb')
        var_gen_word_emb = var_gen_word_emb.dimshuffle(1,0,2)
        ################
        '''
        logit = self.layers.get_layer('ff')[1](tparams, input_gen_ff, activ='linear',
                                               prefix='ff_logit_zd')
                                               # compute word probabilities
        logit_shp = logit.shape
        #debug_print.append( theano.printing.Print('logit shape : \n')(logit_shp))  
        # (t*m, n_words)
        probs = tensor.nnet.softmax(logit.reshape([logit_shp[0]*logit_shp[1],
                                                   logit_shp[2]]))

        # cost
        x_flat = x.flatten() # (t*m,)
        cost = -tensor.log(probs[T.arange(x_flat.shape[0]), x_flat] + 1e-8)
        cost = cost.reshape([x.shape[0], x.shape[1]])
        cost = (cost * mask).sum(0)
        mask_sum = T.sum(mask, axis=0)
        #cost = cost/mask_sum
        probs_reshape = probs.reshape([n_timesteps,n_samples,-1])
        probs_reshape = probs_reshape.dimshuffle(1,0,2)
        #probs_reshape = tensor.log(probs_reshape)
        #probs_reshape = probs_reshape * mask.dimshuffle(1, 0, 'x')
        x_t_3 = tensor.tensor3('x_t_x', dtype='float32')  # n_sample * n_timesteps * n_words
        emb_reshape = emb.dimshuffle(1,0,2)
        #log_p_x_given_h = log_normal2(x=emb_reshape, mean=mean_gen_word_emb, log_var=var_gen_word_emb,) * mask.dimshuffle(1, 0,'x')
        log_p_x_given_h = log_bernoulli(x=x_t_3, p=probs_reshape, eps=options['tolerance_softmax']) * mask.dimshuffle(1, 0,'x')
        log_p_x_given_h = log_p_x_given_h.sum(axis=(1, 2)) #/ mask_sum
        log_p_x_given_h_tot = log_p_x_given_h.mean()
        mean_q_layer = mean_q_layer.dimshuffle(1,0,2)
        log_var_q_layer = log_var_q_layer.dimshuffle(1,0,2)       
        mean_prior_layer = mean_prior_layer.dimshuffle(1,0,2)
        log_var_prior_layer = log_var_prior_layer.dimshuffle(1,0,2)
        kl_divergence = kl_normal2_normal2(mean_q_layer, log_var_q_layer, mean_prior_layer, log_var_prior_layer)
        #debug_print.append( theano.printing.Print('kl_divergence.shapa')(kl_divergence.shape)) 
        kl_divergence_tmp = kl_divergence * mask.dimshuffle(1, 0, 'x')
        kl_divergence_tmp = kl_divergence_tmp.sum(axis=(1, 2)) #/ mask_sum
        kl_divergence_tot = T.mean(kl_divergence_tmp)
        temperature_KL = tensor.scalar('temperature_KL', dtype='float32')
        #lower_bound = -cost.mean() -  temperature_KL* kl_divergence_tot        
        lower_bound = log_p_x_given_h_tot -  temperature_KL * kl_divergence_tot
        lower_bound = -lower_bound
        #LB_beta = tensor.scalar('LB_beta',dtype='float32')
        if options['loss_fun'] == 'LB':
            loss = options['LB_beta_init']*(lower_bound) 
        elif options['loss_fun'] == 'cost_KL':
            loss = cost.mean() + temperature_KL* kl_divergence_tot
        elif options['loss_fun'] == 'cost_LB':
            loss = cost.mean() +  options['LB_beta_init']*lower_bound
        extra = [probs]

        return trng, use_noise, x,x_t_3,mask, ctx, mask_ctx,temperature_KL, kl_divergence_tot,self.z_init_sh,self.mean_prior_init_sh,cost,lower_bound,loss,debug_print, extra

    def pred_probs(self,options, whichset, f_log_probs, verbose=True):

        probs = []
        n_done = 0
        NLL = []
        L = []
        if whichset == 'train':
            tags = self.engine.train
            iterator = self.engine.kf_train
        elif whichset == 'valid':
            tags = self.engine.valid
            iterator = self.engine.kf_valid
        elif whichset == 'test':
            tags = self.engine.test
            iterator = self.engine.kf_test
        else:
            raise NotImplementedError()
        n_samples = np.sum([len(index) for index in iterator])
        for index in iterator:
            tag = [tags[i] for i in index]
            x, mask, ctx, ctx_mask = data_engine.prepare_data(
                self.engine, tag)
            self.reset_state(options,x.shape[1])    
            pred_probs = f_log_probs(x, mask, ctx, ctx_mask)
            L.append(mask.sum(0).tolist())
            NLL.append((-1 * pred_probs).tolist())
            probs.append(pred_probs.tolist())
            n_done += len(tag)
            if verbose:
                sys.stdout.write('\rComputing LL on %d/%d examples'%(
                             n_done, n_samples))
                sys.stdout.flush()
        print
        probs = flatten_list_of_list(probs)
        NLL = flatten_list_of_list(NLL)
        L = flatten_list_of_list(L)
        perp = 2**(np.sum(NLL) / np.sum(L) / np.log(2))
        return -1 * np.mean(probs), perp

    def reset_state(self,options, n_data_points):
        """
        Resets the hidden states to their default values.
        """
        self.z_init_sh.set_value(
            np.zeros((n_data_points, options['latent_size_z']), dtype=theano.config.floatX))
        self.mean_prior_init_sh.set_value(
            np.zeros((n_data_points, options['latent_size_z']), dtype=theano.config.floatX))
        self.log_var_prior_init_sh.set_value(
            np.zeros((n_data_points, options['latent_size_z']), dtype=theano.config.floatX))
    def sent2t3(self,x,options) :
        x = np.transpose(x)
        x_t_3 = np.zeros((x.shape[0], x.shape[1],options['n_words']),
                                                         dtype=theano.config.floatX)
        for i in range(x.shape[0]) :
            for j in range(x.shape[1]) :
                x_t_3[i,j,x[i,j]]=1.0
            
        return x_t_3 

    def _seqs2words(self,caps):
        capsw = []
        sentences = []
        for cc in caps:
            ww = []
            for w in cc:
                if w == 0:
                    break
                ww.append(self.engine.ix_word[1]
                          if w > len(self.engine.ix_word) else self.engine.ix_word[w])
            
            sentences.append(ww)
            capsw.append(' '.join(ww))
        return capsw,sentences

    def train(self):
        self.rng_numpy, self.rng_theano = get_two_rngs()
        test_dir = '/home/guoyu/results/MSR-VTT/dict_lstm_lstmcond_lstmcondrev_stochastic_cost_001KL_res_meanpooling_usext_scale0_1_res_nopad_z256_KL_sum_1/save_dir/'
        model_options = np.load(test_dir+"/model_config.pkl")
        model_options = model_options['attention']
        # reloading
        model_options['word2vec'] = False
        exp_dir = '/home/guoyu/results/youtube/test/experiment/data/SM_RNN_GNet/model_best_b4_'
        for i in model_options:
            print i,":",model_options[i]
        model_saved = test_dir+'/model_best_test_meteor.npz'#model_best_test_blue.npz#model_best_test_meteor.npz
        assert os.path.isfile(model_saved)
        self.engine = data_engine.Movie2Caption('attention', model_options['dataset'],
                                           model_options['video_feature'],
                                           model_options['batch_size'], model_options['valid_batch_size'],
                                           model_options['maxlen'], model_options['n_words'],
                                           model_options['K'], model_options['OutOf'],model_options)
        gts_test = OrderedDict()
        for vidID in self.engine.test_ids:
            gts_test[vidID] = self.engine.CAP[vidID]
        '''
        cPickle.dump(gts_test,open(exp_dir+'test_captions.npy','w'))
        '''
        model_options['ctx_dim'] = self.engine.ctx_dim  
        model_options['att_fun'] = None
        print "Reloading model params..."
        params = self.init_params(model_options)
        params = load_params(model_saved, params)
      
        tparams = init_tparams(params)
        print tparams.keys

        trng = RandomStreams(1234)
        use_noise = theano.shared(np.float32(0.))
        print 'buliding sampler'
        f_init, f_next = self.predict.build_sampler(self.layers, tparams, model_options, use_noise, trng)
        # before any regularizer

        for count_i in np.arange(5):
            stochastic = False
            vidID = self.engine#np.load("vid.npy")#vid1082,
            tags = self.engine.test
            iterator = self.engine.kf_test
            ctxs, ctx_masks = self.engine.prepare_data_for_blue('test')
            z_outs = []
            p_means = []
            samples = []
            for i, ctx, ctx_mask in zip(range(len(ctxs)), ctxs, ctx_masks):
                print 'sampling %d/%d'%(i,len(ctxs))
                sample, score,  _,z_out,p_mean = self.predict.gen_sample(None, f_init, f_next,
                                                       ctx, ctx_mask,None,
                                                       5, maxlen=30)
                
                sidx = np.argmin(score)
                sample = sample[sidx]
                #print _seqs2words([sample])[0]
                samples.append(sample)
                z_outs.append(z_out[sidx])
                p_means.append(p_mean[sidx])
            samples_word,sentences = self._seqs2words(samples)  
            '''
            cPickle.dump(sentences,open(exp_dir+'test_sentences.npy','w'))
            cPickle.dump(z_outs,open(exp_dir+'test_z_outs.npy','w'))
            cPickle.dump(p_means,open(exp_dir+'test_p_means.npy','w'))
            '''
            with open(exp_dir+'test_samples_'+str(count_i+1)+'.txt', 'w') as f:
                print >>f, '\n'.join(samples_word)
                    # Choose learning rate decay schedule
            


if __name__ == '__main__':
    model = Test()
    model.train()
    

