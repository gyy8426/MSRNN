import numpy as np
import theano
import theano.tensor as T
from utils import norm_weight, _p, ortho_weight, tanh, linear, rectifier, get_two_rngs

def param_init_gru_cond( options, params,
                            nin=None, dim=None, dimctx=None,
                            prefix='gru_cond' ):
    if nin == None:
        nin = options[prefix+'hid_dim']
    if dim == None:
        dim = options[prefix+'hid_dim']
    if dimctx == None:
        dimctx = options['video_feature_dim']
    # params of h_
    W = np.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)], axis=1)
    params[_p(prefix, 'W')] = W
    # ctx to LSTM
    V = np.concatenate([norm_weight(dimctx,dim),
                           norm_weight(dimctx,dim),
                           norm_weight(dimctx,dim)], axis=1)
    params[_p(prefix, 'V')] = V
    # LSTM to LSTM
    U = np.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U
    params[_p(prefix, 'b')] = np.zeros((3 * dim,)).astype('float32')
    return params


def gru_cond_layer( options, tparams, state_below,
                    mask=None, context=None, one_step=False,
                    init_state=None,
                    prefix='gru', **kwargs):
    # state_below (t, m, dim_word), or (m, dim_word) in sampling
    # mask (t, m)
    # context (m, dim_ctx), or (dim_word) in sampling
    # init_memory, init_state (m, dim)
    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    n_steps = state_below.shape[0]
    dim = tparams[_p(prefix, 'U')].shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        if init_state is None:
            init_state = T.alloc(0., n_samples, dim)
    else:
        n_samples = 1
        if init_state is None:
            init_state = T.alloc(0., dim)

    if mask is None:
        mask = T.alloc(1., state_below.shape[0], 1)

    U = tparams[_p(prefix, 'U')]
    b = tparams[_p(prefix, 'b')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(m_, x_, h_,U):
        in_r = _slice(x_,0,dim)+T.dot(h_, _slice(U,0,dim))
        r = T.nnet.sigmoid(in_r)
        in_z = _slice(x_,1,dim)+T.dot(h_,_slice(U,1,dim))
        z = T.nnet.sigmoid(in_z)
        in_mh =_slice(x_,2,dim)+T.dot(r * h_,_slice(U,2,dim))
        mh = T.tanh(in_mh)
        h=z * h_+(1.0-z)*mh

        if m_.ndim == 0:
            # when using this for minibatchsize=1
            h = m_ * h + (1. - m_) * h_
        else:
            h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h , mh

    # projected inputs
    state_below = T.dot(state_below, tparams[_p(prefix, 'W')]) + \
                  T.dot(context, tparams[_p(prefix,'V')]) + b

    if one_step:
        rval = _step(mask, state_below, init_state,U)
    else:
        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=[init_state,None],
                                    non_sequences=[U],
                                    name=_p(prefix,'_layers'),
                                    n_steps=n_steps,
                                    profile=False
                                    )
    return rval