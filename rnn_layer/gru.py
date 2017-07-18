import numpy as np
import theano
import theano.tensor as T
from utils import norm_weight, _p, ortho_weight, tanh, linear, rectifier, get_two_rngs
def param_init_gru( options, params, nin=None, dim=None, prefix='gru'):
    assert prefix is not None
    if nin == None:
        nin = options['dim_proj']
    if dim == None:
        dim = options['dim_proj']
    # Stack the weight matricies for faster dot prods
    W = np.concatenate([norm_weight(nin, dim),
                        norm_weight(nin, dim),
                        norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    U = np.concatenate([ortho_weight(dim),
                        ortho_weight(dim),
                        ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U
    params[_p(prefix, 'b')] = np.zeros((3 * dim,)).astype('float32')

    return params

    # This function implements the lstm fprop


def gru_layer( options, tparams, state_below, mask=None, init_state=None,
              one_step=False, prefix='gru', **kwargs):
    if one_step:
        assert init_state, 'previous state must be provided'
    nsteps = state_below.shape[0]
    dim = tparams[_p(prefix, 'U')].shape[0]
    U = tparams[_p(prefix, 'U')]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
        if init_state is None:
            init_state = T.alloc(0., n_samples, dim)
    else:
        n_samples = 1
        if init_state is None:
            init_state = T.alloc(0., dim)

    if mask == None:
        mask = T.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        elif _x.ndim == 2:
            return _x[:, n * dim:(n + 1) * dim]
        return _x[n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, U):

        in_r = _slice(x_, 0, dim) + T.dot(h_, _slice(U, 0, dim))
        r = T.nnet.sigmoid(in_r)
        in_z = _slice(x_, 1, dim) + T.dot(h_, _slice(U, 1, dim))
        z = T.nnet.sigmoid(in_z)
        in_mh = _slice(x_, 2, dim) + T.dot(r * h_, _slice(U, 2, dim))
        mh = T.tanh(in_mh)
        h = z * h_ + (1.0 - z) * mh

        if m_.ndim == 0:
            # when using this for minibatchsize=1
            h = m_ * h + (1. - m_) * h_
        else:
            h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h, r, z, mh

    # W*x+b
    state_below = T.dot(
        state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    if one_step:
        rval = _step(mask, state_below, init_state, U)
    else:
        # b = tparams[_p(prefix, 'b')]
        rval, updates = theano.scan(
            _step,
            sequences=[mask, state_below],
            outputs_info=[init_state, None, None, None],
            non_sequences=[U],
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            strict=True,
            profile=False)
    return rval