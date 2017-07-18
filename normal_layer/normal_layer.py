import numpy as np
import theano
import theano.tensor as T
from utils import norm_weight, _p, ortho_weight, tanh, linear, rectifier, get_two_rngs, sigmoid

def dropout_layer( state_before, use_noise, trng):

    proj = T.switch(use_noise,
                         state_before *
                         trng.binomial(state_before.shape, p=0.5, n=1, dtype=state_before.dtype),
                         state_before * 0.5)
    return proj
def clipped_very_leaky_rectify(x):
    range_nonlin=3.0
    return T.clip(theano.tensor.nnet.relu(x, 1. / 3), -range_nonlin, range_nonlin)
def fflayer(tparams, state_below, activ='lambda x: T.tanh(x)', prefix='ff', **kwargs):
    return eval(activ)(T.dot(state_below, tparams[_p(prefix,'W')])+
                       tparams[_p(prefix,'b')])

def param_init_fflayer( params, nin, nout, scale = 0.01,prefix=None):
    assert prefix is not None
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01)
    params[_p(prefix, 'b')] = np.zeros((nout,)).astype('float32')
    return params