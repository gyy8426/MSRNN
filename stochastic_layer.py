import theano
import theano.tensor as T
from lasagne.utils import unroll_scan
from lasagne.layers import MergeLayer, helper, get_output
from lasagne.random import get_rng
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils import norm_weight, _p, ortho_weight, tanh, linear, rectifier, get_two_rngs
from normal_layer import normal_layer
from rnn_layer import gru
from rnn_layer import gru_cond
from rnn_layer import  lstm
from rnn_layer import lstm_cond
from normal_layer import normal_layer
import numpy as np
rng_numpy, rng_theano = get_two_rngs()
layers = {
    'ff': ('normal_layer.param_init_fflayer', 'normal_layer.fflayer'),
    'lstm': ('lstm.param_init_lstm', 'lstm.lstm_layer'),
    'lstm_cond': ('lstm_cond.param_init_lstm_cond', 'lstm_cond.lstm_cond_layer'),
    'gru': ('gru.param_init_gru', 'gru.gru_layer'),
    'gru_cond': ('gru_cond.param_init_gru_cond', 'gru_cond.gru_cond_layer'),
    }
rng_numpy, rng_theano = get_two_rngs()
gradient_steps=-1
def get_layer(name):
    """
    Part of the reason the init is very slow is because,
    the layer's constructor is called even when it isn't needed
    """
    fns = layers[name]
    return eval(fns[0]), eval(fns[1])

def param_init_stochastic( options, params):
    # Define MLPs to be used in StochsticRecurrentLayer
    mlp_prior_input_dim = options['rnn_cond_wv_dim'] + options['latent_size_z']
    params = get_layer('ff')[0](params,nin=mlp_prior_input_dim,nout=options['num_hidden_mlp'],
                                            prefix='mean_prior_dense1',scale = options['stochastic_scale'])
    params = get_layer('ff')[0](params,nin=options['num_hidden_mlp'],nout=options['latent_size_z'],
                                            prefix='mean_prior_dense2',scale = options['stochastic_scale'])
    params = get_layer('ff')[0](params,nin=mlp_prior_input_dim,nout=options['num_hidden_mlp'],
                                            prefix='log_var_prior_dense1',scale = options['stochastic_scale'])
    params = get_layer('ff')[0](params,nin=options['num_hidden_mlp'],nout=options['latent_size_z'],
                                            prefix='log_var_prior_dense2',scale = options['stochastic_scale'])
    mlp_q_input_dim = options['latent_size_a'] + options['latent_size_z']
    params = get_layer('ff')[0](params,nin=mlp_q_input_dim,nout=options['num_hidden_mlp'],
                                            prefix='mean_q_dense1',scale = options['stochastic_scale'])
    params = get_layer('ff')[0](params,nin=options['num_hidden_mlp'],nout=options['latent_size_z'],
                                            prefix='mean_q_dense2',scale = options['stochastic_scale'])
    params = get_layer('ff')[0](params,nin=mlp_q_input_dim,nout=options['num_hidden_mlp'],
                                            prefix='log_var_q_dense1',scale = options['stochastic_scale'])
    params = get_layer('ff')[0](params,nin=options['num_hidden_mlp'],nout=options['latent_size_z'],
                                            prefix='log_var_q_dense2',scale = options['stochastic_scale'])
    return params

def stochastic_layer(options,tparams,
                            input_p,input_q,
                            z_init,mu_p_init,
                            num_units,unroll_scan,
                            use_mu_residual_q,only_return_final=False,
                            mask_input=None,
                            backwards=False,
                            name='stochastic_layer') :
    debug_print = []
    if options['cons'] == 0 :
        cons=0
    elif options['cons'] < 0 :
        cons=10 ** options['cons']
    else :
        raise  ValueError()
    #debug_print.append( theano.printing.Print('input_p.shapa')(input_p.shape))
    #debug_print.append( theano.printing.Print('input_q.shapa')(input_q.shape))    
    mask = mask_input
    seq_len, num_batch, _ = input_p.shape
    if z_init is None :
        z_init = T.alloc(0., num_batch, options['latent_size_z'])
    if mu_p_init is None :
        mu_p_init = T.alloc(0., num_batch, options['latent_size_z'])
    # Create single recurrent computation step function
    # input__n is the n'th vector of the input

    #debug_print.append( theano.printing.Print('z_init.shapa')(z_init.shape))
    #debug_print.append( theano.printing.Print('mu_p_init.shapa')(mu_p_init.shape))
    stochastic_rs = RandomStreams(get_rng().randint(1, 2147462579))
    def log_sum_exp(a, b):
        return T.log(T.exp(a) + T.exp(b))

    def step(noise_n, input_p_n, input_q_n,
             z_previous,
             mu_p_previous, logvar_p_previous,
             mu_q_previous, logvar_q_previous, *args):
        ####about p ####
        input_p = T.concatenate([input_p_n, z_previous], axis=1)
        
        mu_p_1 = get_layer('ff')[1](tparams, input_p, activ=options['nonlin_decoder'],
                                            prefix='mean_prior_dense1')
        mu_p = get_layer('ff')[1](tparams,mu_p_1,activ='linear',
                                            prefix='mean_prior_dense2')
        logvar_p_1 = get_layer('ff')[1](tparams, input_p, activ=options['nonlin_decoder'],
                                            prefix='log_var_prior_dense1')
        logvar_p = get_layer('ff')[1](tparams,logvar_p_1,activ='linear',
                                            prefix='log_var_prior_dense2')
        logvar_p = T.log(T.exp(logvar_p)+cons)

        ####about q ####
        input_q_n = T.concatenate([input_q_n,z_previous],axis=1)
        mu_q_1 = get_layer('ff')[1](tparams, input_q_n, activ=options['nonlin_decoder'],
                                            prefix='mean_q_dense1')
        mu_q = get_layer('ff')[1](tparams,mu_q_1,activ='linear',
                                            prefix='mean_q_dense2')
        if use_mu_residual_q :
            print "Using residuals for mean_q"
            mu_q += mu_p
        logvar_q_1 = get_layer('ff')[1](tparams, input_q_n, activ=options['nonlin_decoder'],
                                            prefix='log_var_q_dense1')
        logvar_q = get_layer('ff')[1](tparams,logvar_q_1,activ='linear',
                                            prefix='log_var_q_dense2')
        #z_n_print = theano.printing.Print('\n logvar_q info \n)')(logvar_q)
        logvar_q = T.log(T.exp(logvar_q)+cons)

        z_n = mu_q + T.exp(0.5*logvar_q)*noise_n
        return z_n, mu_p, logvar_p, mu_q, logvar_q

    def step_masked(noise_n, input_p_n, input_q_n, mask_n,
             z_previous,
             mu_p_previous, logvar_p_previous,
             mu_q_previous, logvar_q_previous, *args):

        z_n, mu_p, logvar_p, mu_q, logvar_q = step(
            noise_n, input_p_n, input_q_n,
            z_previous, mu_p_previous, logvar_p_previous,
            mu_q_previous, logvar_q_previous, *args)

        z_n = T.switch(mask_n, z_n, z_previous)
        mu_p = T.switch(mask_n, mu_p, mu_p_previous)
        logvar_p = T.switch(mask_n, logvar_p, logvar_p_previous)
        mu_q = T.switch(mask_n, mu_q, mu_q_previous)
        logvar_q = T.switch(mask_n, logvar_q, logvar_q_previous)

        return z_n, mu_p, logvar_p, mu_q, logvar_q
    eps = stochastic_rs.normal(
        size=(seq_len, num_batch, num_units), avg=0.0, std=1.0)

    logvar_init = T.zeros((num_batch,num_units))

    if mask is not None :
        mask = mask.dimshuffle(0, 1, 'x')
        sequences = [eps, input_p, input_q, mask]
        step_fun = step_masked
        #debug_print.append( theano.printing.Print('mask.shapa')(mask.shape))
    else:
        sequences = [eps, input_p, input_q]
        step_fun = step

    if unroll_scan:
        # Retrieve the dimensionality of the incoming layer
        input_shape = eps.shape[0]
        # Explicitly unroll the recurrence instead of using scan
        scan_out = unroll_scan(
            fn=step_fun,
            sequences=sequences,
            outputs_info=[z_init, mu_p_init, logvar_init, mu_p_init, logvar_init],
            go_backwards=backwards,
            n_steps=input_shape)
    else:
        # Scan op iterates over first dimension of input and repeatedly
        # applies the step function
        scan_out = theano.scan(
            fn=step_fun,
            sequences=sequences,
            go_backwards=backwards,
            outputs_info=[z_init, mu_p_init, logvar_init, mu_p_init, logvar_init],
            truncate_gradient=gradient_steps,
            n_steps=seq_len,
            )[0]

    z, mu_p, logvar_p, mu_q, logvar_q = scan_out
    #debug_print.append(z_n_print)
    # When it is requested that we only return the final sequence step,
    # we need to slice it out immediately after scan is applied
    if only_return_final:
        assert False
    '''
    else:
        # dimshuffle back to (n_batch, n_time_steps, n_features))
        z = z.dimshuffle(1, 0, 2)
        mu_p = mu_p.dimshuffle(1, 0, 2)
        logvar_p = logvar_p.dimshuffle(1, 0, 2)
        mu_q = mu_q.dimshuffle(1, 0, 2)
        logvar_q = logvar_q.dimshuffle(1, 0, 2)
    '''
        # if scan is backward reverse the output
    if backwards:
        z = z[:, ::-1]
        mu_p = mu_p[:, ::-1]
        logvar_p = logvar_p[:, ::-1]
        mu_q = mu_q[:, ::-1]
        logvar_q = logvar_q[:, ::-1]
    out_put_res = []
    out_put_res.append(z)
    out_put_res.append(mu_p)
    out_put_res.append(logvar_p)
    out_put_res.append(mu_q)
    out_put_res.append(logvar_q)
    out_put_res.append(debug_print)
    return out_put_res

def stochastic_layer_onestep_q(options,tparams,
                            input_p,input_q,
                            z_init,mu_p_init,
                            num_units,unroll_scan,
                            use_mu_residual_q,only_return_final=False,
                            backwards=False,
                            name='stochastic_layer') :
    if options['cons'] == 0 :
        cons=0
    elif options['cons'] < 0 :
        cons=10**options['cons']
    else :
        raise  ValueError()
    debug_print = []
    seq_len, _ = input_p.shape
    stochastic_rs = RandomStreams(get_rng().randint(1, 2147462579))
    # Create single recurrent computation step function
    # input__n is the n'th vector of the input
    def log_sum_exp(a, b):
        return T.log(T.exp(a) + T.exp(b))

    def step(noise_n, input_p_n, input_q_n,
             z_previous,
             mu_p_previous=None, logvar_p_previous=None,
             mu_q_previous=None, logvar_q_previous=None, *args):

        ####about p ####
        input_p = T.concatenate([input_p_n, z_previous], axis=1)
        mu_p_1 = get_layer('ff')[1](tparams, input_p, activ=options['nonlin_decoder'],
                                            prefix='mean_prior_dense1')
        mu_p = get_layer('ff')[1](tparams,mu_p_1,activ='linear',
                                            prefix='mean_prior_dense2')
        logvar_p_1 = get_layer('ff')[1](tparams, input_p, activ=options['nonlin_decoder'],
                                            prefix='log_var_prior_dense1')
        logvar_p = get_layer('ff')[1](tparams,logvar_p_1,activ='linear',
                                            prefix='log_var_prior_dense2')
        logvar_p = T.log(T.exp(logvar_p)+cons)

        ####about q ####
        input_q_n = T.concatenate([input_q_n,z_previous],axis=1)
        mu_q_1 = get_layer('ff')[1](tparams, input_q_n, activ=options['nonlin_decoder'],
                                            prefix='mean_q_dense1')
        mu_q = get_layer('ff')[1](tparams,mu_q_1,activ='linear',
                                            prefix='mean_q_dense2')
        if use_mu_residual_q :
            print "Using residuals for mean_q"
            mu_q += mu_p
        logvar_q_1 = get_layer('ff')[1](tparams, input_q_n, activ=options['nonlin_decoder'],
                                            prefix='log_var_q_dense1')
        logvar_q = get_layer('ff')[1](tparams,logvar_q_1,activ='linear',
                                            prefix='log_var_q_dense2')

        logvar_q = T.log(T.exp(logvar_q)+cons)

        z_n = mu_q + T.exp(0.5*logvar_q)*noise_n

        return z_n, mu_p, logvar_p, mu_q, logvar_q

    def step_masked(noise_n, input_p_n, input_q_n, mask_n,
             z_previous,
             mu_p_previous, logvar_p_previous,
             mu_q_previous, logvar_q_previous, *args):

        z_n, mu_p, logvar_p, mu_q, logvar_q = step(
            noise_n, input_p_n, input_q_n,
            z_previous, mu_p_previous, logvar_p_previous,
            mu_q_previous, logvar_q_previous, *args)

        z_n = T.switch(mask_n, z_n, z_previous)
        mu_p = T.switch(mask_n, mu_p, mu_p_previous)
        logvar_p = T.switch(mask_n, logvar_p, logvar_p_previous)
        mu_q = T.switch(mask_n, mu_q, mu_q_previous)
        logvar_q = T.switch(mask_n, logvar_q, logvar_q_previous)

        return z_n, mu_p, logvar_p, mu_q, logvar_q
    eps = stochastic_rs.normal(
        size=( 1,num_units), avg=0.0, std=1.0)

    logvar_init = T.zeros((num_units))
    z, mu_p, logvar_p, mu_q, logvar_q = step(eps, input_p, input_q,z_init)

    # When it is requested that we only return the final sequence step,
    # we need to slice it out immediately after scan is applied
    if only_return_final:
        assert False
    '''
    else:
        # dimshuffle back to (n_batch, n_time_steps, n_features))
        z = z.dimshuffle(1, 0, 2)
        mu_p = mu_p.dimshuffle(1, 0, 2)
        logvar_p = logvar_p.dimshuffle(1, 0, 2)
        mu_q = mu_q.dimshuffle(1, 0, 2)
        logvar_q = logvar_q.dimshuffle(1, 0, 2)
    '''
        # if scan is backward reverse the output
    if backwards:
        z = z[:, ::-1]
        mu_p = mu_p[:, ::-1]
        logvar_p = logvar_p[:, ::-1]
        mu_q = mu_q[:, ::-1]
        logvar_q = logvar_q[:, ::-1]
    out_put_res = []
    out_put_res.append(z)
    out_put_res.append(mu_p)
    out_put_res.append(logvar_p)
    out_put_res.append(mu_q)
    out_put_res.append(logvar_q)
    out_put_res.append(debug_print)
    return out_put_res

def stochastic_layer_onestep_noq(options,tparams,
                            input_p,
                            z_init,mu_p_init,
                            num_units,unroll_scan,
                            use_mu_residual_q,only_return_final=False,
                            backwards=False,
                            name='stochastic_layer') :
    if options['cons'] == 0 :
        cons=0
    elif options['cons'] < 0 :
        cons=10**options['cons']
    else :
        raise  ValueError()
    debug_print = []
    seq_len, _ = input_p.shape
    stochastic_rs = RandomStreams(get_rng().randint(1, 2147462579))
    # Create single recurrent computation step function
    # input__n is the n'th vector of the input
    def log_sum_exp(a, b):
        return T.log(T.exp(a) + T.exp(b))

    def step(noise_n, input_p_n,
             z_previous,
             mu_p_previous=None, logvar_p_previous=None,
             mu_q_previous=None, logvar_q_previous=None, *args):

        ####about p ####
        input_p = T.concatenate([input_p_n, z_previous], axis=1)
        mu_p_1 = get_layer('ff')[1](tparams, input_p, activ=options['nonlin_decoder'],
                                            prefix='mean_prior_dense1')
        mu_p = get_layer('ff')[1](tparams,mu_p_1,activ='linear',
                                            prefix='mean_prior_dense2')
        logvar_p_1 = get_layer('ff')[1](tparams, input_p, activ=options['nonlin_decoder'],
                                            prefix='log_var_prior_dense1')
        logvar_p = get_layer('ff')[1](tparams,logvar_p_1,activ='linear',
                                            prefix='log_var_prior_dense2')
        logvar_p = T.log(T.exp(logvar_p)+cons)

        
        '''
        ####about q ####
        input_q_n = T.concatenate([input_q_n,z_previous],axis=1)
        mu_q_1 = get_layer('ff')[1](tparams, input_q_n, activ=options['nonlin_decoder'],
                                            prefix='mean_q_dense1')
        mu_q = get_layer('ff')[1](tparams,mu_q_1,activ='linear',
                                            prefix='mean_q_dense2')
        if use_mu_residual_q :
            print "Using residuals for mean_q"
            mu_q += mu_p
        logvar_q_1 = get_layer('ff')[1](tparams, input_q_n, activ=options['nonlin_decoder'],
                                            prefix='log_var_q_dense1')
        logvar_q = get_layer('ff')[1](tparams,logvar_q_1,activ='linear',
                                            prefix='log_var_q_dense2')

        logvar_q = T.log(T.exp(logvar_q)+cons)
        '''
        z_n = mu_p + T.exp(0.5*logvar_p)*noise_n

        return z_n, mu_p, logvar_p #, mu_q, logvar_q

    def step_masked(noise_n, input_p_n, input_q_n, mask_n,
             z_previous,
             mu_p_previous, logvar_p_previous,
             mu_q_previous, logvar_q_previous, *args):

        z_n, mu_p, logvar_p, mu_q, logvar_q = step(
            noise_n, input_p_n, input_q_n,
            z_previous, mu_p_previous, logvar_p_previous,
            mu_q_previous, logvar_q_previous, *args)

        z_n = T.switch(mask_n, z_n, z_previous)
        mu_p = T.switch(mask_n, mu_p, mu_p_previous)
        logvar_p = T.switch(mask_n, logvar_p, logvar_p_previous)
        '''
        mu_q = T.switch(mask_n, mu_q, mu_q_previous)
        logvar_q = T.switch(mask_n, logvar_q, logvar_q_previous)
        '''
        return z_n, mu_p, logvar_p #, mu_q, logvar_q
    eps = stochastic_rs.normal(
        size=( 1,num_units), avg=0.0, std=1.0)

    logvar_init = T.zeros((num_units))
    z, mu_p, logvar_p = step(eps, input_p,z_init)

    # When it is requested that we only return the final sequence step,
    # we need to slice it out immediately after scan is applied
    if only_return_final:
        assert False
    '''
    else:
        # dimshuffle back to (n_batch, n_time_steps, n_features))
        z = z.dimshuffle(1, 0, 2)
        mu_p = mu_p.dimshuffle(1, 0, 2)
        logvar_p = logvar_p.dimshuffle(1, 0, 2)
        mu_q = mu_q.dimshuffle(1, 0, 2)
        logvar_q = logvar_q.dimshuffle(1, 0, 2)
    '''
        # if scan is backward reverse the output
    if backwards:
        z = z[:, ::-1]
        mu_p = mu_p[:, ::-1]
        logvar_p = logvar_p[:, ::-1]
        '''
        mu_q = mu_q[:, ::-1]
        logvar_q = logvar_q[:, ::-1]
        '''
    out_put_res = []
    out_put_res.append(z)
    out_put_res.append(mu_p)
    out_put_res.append(logvar_p)
    '''
    out_put_res.append(mu_q)
    out_put_res.append(logvar_q)
    '''
    out_put_res.append(debug_print)
    return out_put_res


            