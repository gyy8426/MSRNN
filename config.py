from jobman import DD

RAB_DATASET_BASE_PATH = '/data4/guozhao/predatas/MSVD/'
#RAB_C3D_FEATURE_BASE_PATH = '/mnt/disk3/guozhao/features/MSVD/C3D/'
#/mnt/disk1/guozhao/predatas/yaoli/youtube2text_iccv15/
#/mnt/disk3/guozhao/predatas/MSVD/
RAB_FEATURE_BASE_PATH = '/data4/guozhao/features/MSVD/ResNet_152/' 
#/mnt/disk3/guozhao/features/MSVD/inception-v3/ 
#/mnt/disk3/guozhao/features/MSVD/ResNet_152/
RAB_EXP_PATH = '/home/guoyuyu/results/youtube/lstm_lstmcond_lstmcond_srnn_soft_nouseresq/'
'''
RAB_DATASET_BASE_PATH = '/mnt/disk3/guozhao/predatas/MSR-VTT/'
#RAB_C3D_FEATURE_BASE_PATH = '/mnt/disk3/guozhao/features/MSR-VTT/C3D/'
RAB_FEATURE_BASE_PATH = '/mnt/disk3/guozhao/features/MSR-VTT/ResNet_152/'
RAB_EXP_PATH = '/home/guoyu/results/MSR-VTT/word2vec_lstm_lstmcond_lstmcondrev_stochastic_cost_001KL_res_meanpooling_usext_scale0_1_resc3d_nopad_z256_KL_sum_vtt/'
'''
config = DD({
    'model': 'attention',
    'random_seed': 1234,
    # ERASE everything under save_model_path
    'erase_history': False,
    'attention': DD({
        'reload_': False,
        'verbose': True,
        'debug': False,
        'save_model_dir': RAB_EXP_PATH + 'save_dir/',
        'from_dir': RAB_EXP_PATH + 'from_dir/',
        # datasetre
        'dataset': 'youtube2text',#msr-vtt #youtube2text
        'video_feature': 'googlenet',#Gnet_c3d
        'K':28, # 26 when compare
        'OutOf':None,
        # network
        'word2vec':False, 
        'dim_word':512,#468, # 474 #300 #512
        'rnn_word_dim': 512,
        'rnn_cond_wv_dim': 512,
        'ctx_dim':-1,# auto set
        'n_layers_out':1, # for predicting next word
        'n_layers_init':0,
        'encoder_dim': 1024,#300,
        'prev2out':True,
        'ctx2out':True,
        'selector':True,
        'n_words':20000,
        'maxlen':30, # max length of the descprition
        'use_dropout':True,
        'isGlobal': True,
        'att_fun': None,
        ##stochastic_part##
        'a_layer_type' : 'lstm_cond',
        'use_mu_residual_q' : False,
        'flat_mlp_num' : 1,
        'unroll_scan' : False,
        'smoothing' : True,
        'latent_size_a' : 512,
        'latent_size_z' :256,
        'num_hidden_mlp' : 256,
        'nonlin_decoder' : 'clipped_very_leaky_rectify',
        'cons' : -8.0,
        'tolerance_softmax' :1e-8,
        'loss_fun':'cost_KL',
        ## training
        'stochastic_scale':0.01,
        'patience':20,
        'max_epochs':500,
        'decay_c':1e-4,
        'alpha_entropy_r': 0.,
        'alpha_c':0.70602,
        'lrate':0.0001,
        'optimizer':'adadelta',
        'clip_c': 10.,
        # learning rate set 
        'decay_type':'exponential', 'decay':1.2,
        'scale_decay':1.0, 'no_decay_epochs':20, 
        # temp_KL set 
        'tempKL_type':'linear', 'tempKL_start':0.01, 'tempKL_epochs':20, 'tempKL_decay':1.02,
        # minibatches
        'batch_size': 64, # for trees use 25
        'valid_batch_size':200,
        'dispFreq':10,
        'validFreq':1000,
        'saveFreq':-1, # this is disabled, now use sampleFreq instead
        'sampleFreq':100,
        'LB_beta_init':1.0,
        # blue, meteor, or both
        'metric': 'everything', # set to perplexity on DVS
        }),
    })
