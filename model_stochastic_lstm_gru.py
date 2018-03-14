from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import sys
import time
from layers import Layers
import data_engine_res
import data_engine_googlenet
import data_engine_Gnet_c3d
import metrics
from optimizers import *
from predict import *
import theano.tensor as T 
from parmesan.distributions import log_bernoulli
from decay import *
import math
#from sklearn.preprocessing import minmax_scale
data_engine =  data_engine_res
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
class Attention(object):
    def __init__(self, channel=None):
        self.rng_numpy, self.rng_theano = get_two_rngs()
        self.layers = Layers()
        self.predict = Predict()
        self.channel = channel

    def load_params(self, path, params):
        # load params from disk
        pp = np.load(path)
        for kk, vv in params.iteritems():
            if kk not in pp:
                raise Warning('%s is not in the archive'%kk)
            params[kk] = pp[kk]

        return params
    def init_params(self, options):
        # all parameters
        params = OrderedDict()
        # embedding
        if not options['word2vec']:
            params['Wemb'] = norm_weight(options['n_words'], options['dim_word'])

        ctx_dim = options['ctx_dim']
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
            if options['a_layer_type'] == 'lstm':
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
        ff_logit_zd_nin = options['latent_size_z'] + options['rnn_cond_wv_dim']
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
        if options['word2vec']:
            emb = self.engine.Wemb[x.flatten()].reshape(
                [n_timesteps, n_samples, options['dim_word']])        
        else :
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
        cost = cost#/mask_sum
        probs_reshape = probs.reshape([n_timesteps,n_samples,-1])
        probs_reshape = probs_reshape.dimshuffle(1,0,2)
        #probs_reshape = tensor.log(probs_reshape)
        #probs_reshape = probs_reshape * mask.dimshuffle(1, 0, 'x')
        x_t_3 = tensor.tensor3('x_t_x', dtype='float32')  # n_sample * n_timesteps * n_words
        emb_reshape = emb.dimshuffle(1,0,2)
        #log_p_x_given_h = log_normal2(x=emb_reshape, mean=mean_gen_word_emb, log_var=var_gen_word_emb,) * mask.dimshuffle(1, 0,'x')
        log_p_x_given_h = log_bernoulli(x=x_t_3, p=probs_reshape, eps=options['tolerance_softmax']) * mask.dimshuffle(1, 0,'x')
        log_p_x_given_h = log_p_x_given_h.sum(axis=(1, 2))#/mask_sum
        log_p_x_given_h_tot = log_p_x_given_h.mean()
        mean_q_layer = mean_q_layer.dimshuffle(1,0,2)
        log_var_q_layer = log_var_q_layer.dimshuffle(1,0,2)       
        mean_prior_layer = mean_prior_layer.dimshuffle(1,0,2)
        log_var_prior_layer = log_var_prior_layer.dimshuffle(1,0,2)
        kl_divergence = kl_normal2_normal2(mean_q_layer, log_var_q_layer, mean_prior_layer, log_var_prior_layer)
        #debug_print.append( theano.printing.Print('kl_divergence.shapa')(kl_divergence.shape)) 
        kl_divergence_tmp = kl_divergence * mask.dimshuffle(1, 0, 'x')
        kl_divergence_tmp = kl_divergence_tmp.sum(axis=(1, 2))#/mask_sum 
        kl_divergence_tot = T.mean(kl_divergence_tmp)
        temperature_KL = tensor.scalar('temperature_KL', dtype='float32')
        #lower_bound = -cost.mean() -  temperature_KL* kl_divergence_tot        
        lower_bound = log_p_x_given_h_tot -  temperature_KL * kl_divergence_tot
        lower_bound = -lower_bound
        #LB_beta = tensor.scalar('LB_beta',dtype='float32')
        if options['loss_fun'] == 'LB':
            loss = options['LB_beta_init']*(lower_bound) 
        elif options['loss_fun'] == 'cost_KL':
            loss = cost.mean() + temperature_KL * kl_divergence_tot
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

    def train(self,
                random_seed=1234,
                reload_=False,
                verbose=True,
                debug=True,
                save_model_dir='',
                from_dir=None,
                # dataset
                dataset='youtube2text',
                video_feature='googlenet',
                word2vec = False,
                K=10,
                OutOf=240,
                # network
                dim_word=256, # word vector dimensionality
                ctx_dim=-1, # context vector dimensionality, auto set
                rnn_word_dim=512,rnn_cond_wv_dim=512,n_layers_out=1,n_layers_init=1,
                encoder='none',
                encoder_dim=100,prev2out=False,ctx2out=False,selector=False,n_words=100000,
                maxlen=100, # maximum length of the description
                use_dropout=False,isGlobal=False,
                att_fun='None',
                ##stochastic_part##
                a_layer_type = 'lstm',
                use_mu_residual_q=True,
                flat_mlp_num=1,
                unroll_scan=False,
                smoothing=True,
                latent_size_a=512,
                latent_size_z=128,
                num_hidden_mlp=256,
                nonlin_decoder = 'clipped_very_leaky_rectify',
                cons=-8.0,
                tolerance_softmax = 1e-8,
                temperature_KL = 1.0,
                LB_beta_init = 1.0,
                # training
                stochastic_scale = 0.01 ,
                patience=10,max_epochs=150,
                decay_c=0.,alpha_c=0.,alpha_entropy_r=0.,
                lrate=0.01,optimizer='adadelta',clip_c=2.,
                # learning rate set 
                decay_type='exponential', decay=1.2,
                scale_decay=1.0, no_decay_epochs=20, 
                # temp_KL set 
                tempKL_type='linear', tempKL_start=0.8, tempKL_epochs=20, tempKL_decay=1.02,
                loss_fun='KL_cost',
                # minibatch
                batch_size = 64,
                valid_batch_size = 64,
                dispFreq=10,
                validFreq=10,
                saveFreq=10, # save the parameters after every saveFreq updates
                sampleFreq=2, # generate some samples after every sampleFreq updates
                # metric
                metric='blue'
                ):
        self.rng_numpy, self.rng_theano = get_two_rngs()

        model_options = locals().copy()

        if 'self' in model_options:
            del model_options['self']
        model_options = validate_options(model_options)
        with open('%smodel_options.pkl'%save_model_dir, 'wb') as f:
            pkl.dump(model_options, f)
        print 'Loading data'
        self.engine = data_engine.Movie2Caption('attention', dataset,
                                           video_feature,
                                           batch_size, valid_batch_size,
                                           maxlen, n_words,
                                           K, OutOf, model_options)
        model_options['ctx_dim'] = self.engine.ctx_dim
        model_options['n_words'] = self.engine.n_words
        self.engine.Wemb = None
        if model_options['word2vec']:
            self.engine.Wemb = theano.shared(self.engine.Wemb, name='Wemb')
        print "batch size is",model_options['batch_size']
        '''
        ####word2vector#### 
        self.engine.Wemb = minmax_scale(self.engine.Wemb, feature_range=(0, 1))
        new_Wemb = np.zeros(shape=(model_options['n_words'],model_options['dim_word']),dtype='float32')
        new_Wemb[np.arange(new_Wemb.shape[0])]=self.engine.Wemb[1]
        new_Wemb[:self.engine.Wemb.shape[0],:self.engine.Wemb.shape[1]]=self.engine.Wemb
        self.engine.Wemb = new_Wemb
        self.engine.Wemb = theano.shared(self.engine.Wemb, name='Wemb')
        '''
        print 'init params'

        t0 = time.time()
        for i in model_options:
            print i ,":",model_options[i]
        params = self.init_params(model_options)

        # reloading
        if reload_:
            model_saved = from_dir+'/model_best_so_far.npz'
            assert os.path.isfile(model_saved)
            print "Reloading model params..."
            params = load_params(model_saved, params)

        tparams = init_tparams(params)
        if verbose:
            print tparams.keys

        trng, use_noise, x, x_t_3, mask, ctx, mask_ctx,temperature_KL,kl_divergence_tot,\
        z_init,mu_p_init,cost, lower_bound,loss,debug_print,extra = \
            self.build_model(tparams, model_options)

        print 'buliding sampler'
        f_init, f_next = self.predict.build_sampler(self.layers, tparams, model_options, use_noise, trng, self.engine.Wemb)
        # before any regularizer
        print 'building f_log_probs'
        f_log_probs = theano.function([x, mask, ctx, mask_ctx], -cost,
                                      profile=False, on_unused_input='ignore')
        # debug  printing 
        f_debug_printing = theano.function([x, mask, ctx, mask_ctx, x_t_3,temperature_KL], debug_print,
                                      profile=False, on_unused_input='ignore')
        cost = cost.mean()
        if decay_c > 0.:
            decay_c = theano.shared(np.float32(decay_c), name='decay_c')
            weight_decay = 0.
            for kk, vv in tparams.iteritems():
                weight_decay += (vv ** 2).sum()
            weight_decay *= decay_c
            loss += weight_decay

        print 'compute grad'
        grads = tensor.grad(loss, wrt=itemlist(tparams))
        if clip_c > 0.:
            g2 = 0.
            for g in grads:
                g2 += (g**2).sum()
            new_grads = []
            for g in grads:
                new_grads.append(tensor.switch(g2 > (clip_c**2),
                                               g / tensor.sqrt(g2) * clip_c,
                                               g))
            grads = new_grads

        lr = tensor.scalar(name='lr')
        print 'build train fns'
        f_grad_shared, f_update = eval(optimizer)(lr, tparams, grads,
                                                  [x, mask, ctx, mask_ctx, x_t_3,temperature_KL], cost,
                                                  extra + grads + [loss]+[lower_bound]+[kl_divergence_tot])

        print 'compilation took %.4f sec'%(time.time()-t0)
        print 'Optimization'

        history_errs = []
        # reload history
        if reload_:
            print 'loading history error...'
            history_errs = np.load(
                from_dir+'model_best_so_far.npz')['history_errs'].tolist()

        bad_counter = 0

        processes = None
        queue = None
        rqueue = None
        shared_params = None

        uidx = 0
        uidx_best_blue = 0
        uidx_best_valid_err = 0
        estop = False
        best_p = unzip(tparams)
        best_blue_valid = 0
        best_valid_err = 999
        temp_KL = 1.0
        if reload_ :
            uidx = 16850
            # Choose learning rate decay schedule
        if model_options['decay_type'].lower() == 'power':
            decay_learning_rate = PowerDecaySchedule(model_options['decay'], model_options['scale_decay'], model_options['max_epochs'],
                                                     model_options['no_decay_epochs'])
        elif model_options['decay_type'].lower() == 'exponential':
            decay_learning_rate = ExponentialDecaySchedule(model_options['decay'], model_options['max_epochs'],
                                                           model_options['no_decay_epochs'])
        else:
            raise ValueError('Invalid decay_type \'' + model_options['decay_type'] + '\'')
            # Choose temperature schedule for the KL term
        # We change the KL divergence slightly after every batch, e.g. with temperature linearly increasing from 0.2 to 1
        n_batches_train = len(self.engine.train) // model_options['batch_size']
        max_decay_iters_KL = np.inf
        max_num_iters_KL = model_options['max_epochs'] * n_batches_train
        no_decay_iters_KL = max_num_iters_KL - model_options['tempKL_epochs'] * n_batches_train
        y_range_KL = (float(model_options['tempKL_start']), 1.0)
        reverse_KL = True
        if model_options['tempKL_type'].lower() == 'power':
            temperature_KL = PowerDecaySchedule(model_options['tempKL_decay'], scale_decay=1.0, max_num_epochs=max_num_iters_KL,
                                                no_decay_epochs=no_decay_iters_KL, max_decay_epochs=max_decay_iters_KL,
                                                reverse=reverse_KL, y_range=y_range_KL)
        elif model_options['tempKL_type'].lower() == 'exponential':
            temperature_KL = ExponentialDecaySchedule(model_options['tempKL_decay'], max_num_epochs=max_num_iters_KL,
                                                      no_decay_epochs=no_decay_iters_KL,
                                                      max_decay_epochs=max_decay_iters_KL, reverse=reverse_KL,
                                                      y_range=y_range_KL)
        elif model_options['tempKL_type'].lower() == 'linear':
            # in this case settings.tempKL_decay is useless as we are also passing y_range_KL
            temperature_KL = LinearDecaySchedule(model_options['tempKL_decay'], max_num_epochs=max_num_iters_KL,
                                                 no_decay_epochs=no_decay_iters_KL, max_decay_epochs=max_decay_iters_KL,
                                                 reverse=reverse_KL, y_range=y_range_KL)
        else:
            raise ValueError('Invalid tempKL_type \'' + model_options['tempKL_type'] + '\'')
        for eidx in xrange(model_options['max_epochs']):
            n_samples = 0
            train_costs = []
            train_losss = []
            train_LBs = []
            grads_record = []
            print 'Epoch ', eidx
            lrate = model_options['lrate'] * decay_learning_rate.get_decay(eidx)
            if (lrate >= model_options['lrate']):
                lrate = model_options['lrate']
            for idx in self.engine.kf_train:
                tags = [self.engine.train[index] for index in idx]
                n_samples += len(tags)
                uidx += 1
                use_noise.set_value(1.)
                temp_KL = np.asarray(temperature_KL.get_decay(uidx),
                     dtype=theano.config.floatX)
                if reload_ :
                    temp_KL = 1.0
                pd_start = time.time()
                x, mask, ctx, ctx_mask = data_engine.prepare_data(
                    self.engine, tags)
                x_t_3 = self.sent2t3(x,model_options)
                pd_duration = time.time() - pd_start
                if x is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    continue
                self.reset_state(model_options,x.shape[1])
                ud_start = time.time()
                #print "x shape ",x.shape
                #debugprint = f_debug_printing(x, mask, ctx, ctx_mask, x_t_3,temp_KL)
                '''
                flag_grad = True 
                while flag_grad :
                    try : 
                        rvals = f_grad_shared(x, mask, ctx, ctx_mask, x_t_3,temp_KL)
                        flag_grad = False
                    except :
                        flag_grad = True
                '''
                rvals = f_grad_shared(x, mask, ctx, ctx_mask, x_t_3,temp_KL)
                cost = rvals[0]
                probs = rvals[1]
                grads = rvals[2:-3]
                loss = rvals[-3]
                lower_bound = rvals[-2]
                kl_divergence_tot = rvals[-1]
                grads, NaN_keys = grad_nan_report(grads, tparams)
                if len(grads_record) >= 5:
                    del grads_record[0]
                grads_record.append(grads)
                if NaN_keys != []:
                    print 'grads contain NaN'
                    import pdb; pdb.set_trace()
                if np.isnan(loss) or np.isinf(loss):
                    print 'NaN detected in loss'
                    import pdb; pdb.set_trace()
                # update params
                flag_update = True 
                while flag_update :
                    try :
                        f_update(lrate)
                        flag_update = False
                    except :
                        flag_update = True
                        
                ud_duration = time.time() - ud_start

                if eidx == 0:
                    train_error = cost
                    train_LB = lower_bound
                    train_loss = loss
                else:
                    train_error = train_error * 0.95 + cost * 0.05
                    train_loss = train_loss * 0.95 + loss * 0.05
                    train_LB = train_LB * 0.95 + lower_bound * 0.05
                train_costs.append(cost)                
                train_losss.append(train_loss)
                train_LBs.append(train_LB)
                if np.mod(uidx, dispFreq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Train cost mean so far', cost, 'Train kl_divergence_tot mean so far', kl_divergence_tot,\
                        ' lower_bound mean so far is ',lower_bound, 'loss mean so far is ',loss,\
                      '\n fetching data time spent (sec)', np.round(pd_duration,3), \
                      ' update time spent (sec)', np.round(ud_duration,3) ,\
                      ' \n lrate is ',lrate,' temp_KL is ', temp_KL

                if np.mod(uidx, saveFreq) == 0:
                    pass

                if np.mod(uidx, sampleFreq) == 0:
                    use_noise.set_value(0.)
                    print '------------- sampling from train ----------'
                    self.predict.sample_execute(self.engine, model_options, tparams,
                                                f_init, f_next, x, ctx, ctx_mask, trng)

                    print '------------- sampling from valid ----------'
                    idx = self.engine.kf_valid[np.random.randint(1, len(self.engine.kf_valid) - 1)]
                    tags = [self.engine.valid[index] for index in idx]
                    x_s, mask_s, ctx_s, mask_ctx_s = data_engine.prepare_data(self.engine, tags)
                    self.predict.sample_execute(self.engine, model_options, tparams,
                                                f_init, f_next, x_s, ctx_s, mask_ctx_s, trng)
                    # end of sample

                if validFreq != -1 and np.mod(uidx, validFreq) == 0:
                    t0_valid = time.time()
                    
                    current_params = unzip(tparams)
                    np.savez(save_model_dir+'model_current.npz',
                             history_errs=history_errs, **current_params)

                    use_noise.set_value(0.)
                    train_err = -1
                    train_perp = -1
                    valid_err = -1
                    valid_perp = -1
                    test_err = -1
                    test_perp = -1
                    if not debug:
                        # first compute train cost
                        if 0:
                            print 'computing cost on trainset'
                            train_err, train_perp = self.pred_probs(
                                    model_options,
                                    'train', f_log_probs,
                                    verbose=model_options['verbose'])
                        else:
                            train_err = 0.
                            train_perp = 0.
                        if 1:
                            print 'validating...'
                            valid_err, valid_perp = self.pred_probs(
                                model_options,
                                'valid', f_log_probs,
                                verbose=model_options['verbose'],
                                )
                        else:
                            valid_err = 0.
                            valid_perp = 0.
                        if 1:
                            print 'testing...'
                            test_err, test_perp = self.pred_probs(
                                model_options,
                                'test', f_log_probs,
                                verbose=model_options['verbose']
                                )
                        else:
                            test_err = 0.
                            test_perp = 0.

                    mean_ranking = 0
                    blue_t0 = time.time()
                    scores, processes, queue, rqueue, shared_params = \
                        metrics.compute_score(model_type='attention',
                                              model_archive=current_params,
                                              options=model_options,
                                              engine=self.engine,
                                              save_dir=save_model_dir,
                                              beam=5, n_process=5,
                                              whichset='both',
                                              on_cpu=False,
                                              processes=processes, queue=queue, rqueue=rqueue,
                                              shared_params=shared_params, metric=metric,
                                              one_time=False,
                                              f_init=f_init, f_next=f_next, model=self.predict
                                              )

                    valid_B1 = scores['valid']['Bleu_1']
                    valid_B2 = scores['valid']['Bleu_2']
                    valid_B3 = scores['valid']['Bleu_3']
                    valid_B4 = scores['valid']['Bleu_4']
                    valid_Rouge = scores['valid']['ROUGE_L']
                    valid_Cider = scores['valid']['CIDEr']
                    valid_meteor = scores['valid']['METEOR']
                    test_B1 = scores['test']['Bleu_1']
                    test_B2 = scores['test']['Bleu_2']
                    test_B3 = scores['test']['Bleu_3']
                    test_B4 = scores['test']['Bleu_4']
                    test_Rouge = scores['test']['ROUGE_L']
                    test_Cider = scores['test']['CIDEr']
                    test_meteor = scores['test']['METEOR']
                    print 'computing meteor/blue score used %.4f sec, '\
                          'blue score: %.1f, meteor score: %.1f'%(
                    time.time()-blue_t0, valid_B4, valid_meteor)
                    history_errs.append([eidx, uidx, train_err, train_perp,
                                         valid_perp, test_perp,
                                         valid_err, test_err,
                                         valid_B1, valid_B2, valid_B3,
                                         valid_B4, valid_meteor, valid_Rouge, valid_Cider,
                                         test_B1, test_B2, test_B3,
                                         test_B4, test_meteor, test_Rouge, test_Cider])
                    np.savetxt(save_model_dir+'train_valid_test.txt',
                                  history_errs, fmt='%.3f')
                    print 'save validation results to %s'%save_model_dir
                    # save best model according to the best blue or meteor
                    if len(history_errs) > 1 and valid_B4 > np.array(history_errs)[:-1,11].max():
                        print 'Saving to %s...'%save_model_dir,
                        best_p_valid_B4 = unzip(tparams)
                        np.savez(
                            save_model_dir+'model_best_valid_blue.npz',
                            history_errs=history_errs, **best_p_valid_B4)
                    if len(history_errs) > 1 and valid_meteor > np.array(history_errs)[:-1,12].max():
                        print 'Saving to %s...'%save_model_dir,
                        best_p_valid_meteor = unzip(tparams)
                        np.savez(
                            save_model_dir+'best_best_valid_meteor.npz',
                            history_errs=history_errs, **best_p_valid_meteor)
                    if len(history_errs) > 1 and valid_err < np.array(history_errs)[:-1,6].min():
                        best_p = unzip(tparams)
                        bad_counter = 0
                        best_valid_err = valid_err
                        uidx_best_valid_err = uidx

                        print 'Saving to %s...'%save_model_dir,
                        np.savez(
                            save_model_dir+'model_best_so_far.npz',
                            history_errs=history_errs, **best_p)
                        with open('%smodel_options.pkl'%save_model_dir, 'wb') as f:
                            pkl.dump(model_options, f)
                        print 'Done'
                    elif len(history_errs) > 1 and valid_err >= np.array(history_errs)[:-1,6].min():
                        bad_counter += 1
                        print 'history best ',np.array(history_errs)[:,6].min()
                        print 'bad_counter ',bad_counter
                        print 'patience ',patience
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                    if self.channel:
                        self.channel.save()

                    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err, \
                          'best valid err so far',best_valid_err
                    print 'valid took %.2f sec'%(time.time() - t0_valid)
                    # end of validatioin
                if debug:
                    break
            if estop:
                break
            if debug:
                break

            # end for loop over minibatches
            print 'This epoch has seen %d samples, train cost %.2f'%(
                n_samples, np.mean(train_costs))
        # end for loop over epochs
        print 'Optimization ended.'
        if best_p is not None:
            zipp(best_p, tparams)

        print 'stopped at epoch %d, minibatch %d, '\
              'curent Train %.2f, current Valid %.2f, current Test %.2f '%(
               eidx, uidx, np.mean(train_err), np.mean(valid_err), np.mean(test_err))
        params = copy.copy(best_p)
        np.savez(save_model_dir+'model_best.npz',
                 train_err=train_err,
                 valid_err=valid_err, test_err=test_err, history_errs=history_errs,
                 **params)

        if history_errs != []:
            history = np.asarray(history_errs)
            best_valid_idx = history[:,6].argmin()
            np.savetxt(save_model_dir+'train_valid_test.txt', history, fmt='%.4f')
            print 'final best exp ', history[best_valid_idx]

        return train_err, valid_err, test_err


def train_from_scratch(state, channel):
    t0 = time.time()
    print 'training an attention model'
    model = Attention(channel)
    model.train(**state.attention)
    print 'training time in total %.4f sec'%(time.time()-t0)

