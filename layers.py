import numpy as np
import theano
import theano.tensor as T
from utils import norm_weight, _p, ortho_weight, tanh, linear, rectifier, get_two_rngs
from rnn_layer import gru,gru_cond,lstm,lstm_cond,lstm_att
from normal_layer import normal_layer
import  stochastic_layer
class Layers(object):

    def __init__(self):
        # layers: 'name': ('parameter initializer', 'feedforward')
        self.layers = {
            'ff': ('normal_layer.param_init_fflayer', 'normal_layer.fflayer'),
            'lstm': ('lstm.param_init_lstm', 'lstm.lstm_layer'),
            'lstm_cond': ('lstm_cond.param_init_lstm_cond', 'lstm_cond.lstm_cond_layer'),
            'lstm_att': ('lstm_att.param_init_lstm_att', 'lstm_att.lstm_att_layer'),
            'gru': ('gru.param_init_gru', 'gru.gru_layer'),
            'gru_cond': ('gru_cond.param_init_gru_cond', 'gru_cond.gru_cond_layer'),
            'stochastic':('stochastic_layer.param_init_stochastic', 'stochastic_layer.stochastic_layer'),            
            }
        self.rng_numpy, self.rng_theano = get_two_rngs()
        self.dropout_layer=normal_layer.dropout_layer
        self.stochastic_layer_onestep_q=stochastic_layer.stochastic_layer_onestep_q
        self.stochastic_layer_onestep_noq=stochastic_layer.stochastic_layer_onestep_noq
        return


    def get_layer(self, name):
        """
        Part of the reason the init is very slow is because,
        the layer's constructor is called even when it isn't needed
        """
        fns = self.layers[name]
        return eval(fns[0]), eval(fns[1])






