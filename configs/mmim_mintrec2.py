class Param():
    
    def __init__(self, args):

        self.common_param = self._get_common_parameters(args)
        self.hyper_param = self._get_hyper_parameters(args)

    def _get_common_parameters(self, args):
        """
            padding_mode (str): The mode for sequence padding ('zero' or 'normal').
            padding_loc (str): The location for sequence padding ('start' or 'end'). 
            eval_monitor (str): The monitor for evaluation ('loss' or metrics, e.g., 'f1', 'acc', 'precision', 'recall').  
            need_aligned: (bool): Whether to perform data alignment between different modalities.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patience (int): Patient steps for Early Stop.
        """
        common_parameters = {
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': False,
            'eval_monitor': ['f1'],
            'train_batch_size': 16,
            'eval_batch_size': 8,
            'test_batch_size': 8,
            'wait_patience': 8,
            'num_train_epochs': 100,
        }
        return common_parameters
    
    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            beta_shift (float): The coefficient for nonverbal displacement to create the multimodal vector.
            dropout_prob (float): The embedding dropout probability.
            warmup_proportion (float): The warmup ratio for learning rate.
            lr (float): The learning rate of backbone.
            aligned_method (str): The method for aligning different modalities. ('ctc', 'conv1d', 'avg_pool')
            weight_decay (float): The coefficient for L2 regularization. 
        """
        hyper_parameters = {
            'add_va': False,
            'cpc_activation': 'Tanh',
            'mmilb_mid_activation': 'ReLU',
            'mmilb_last_activation': 'Tanh',
            'optim': 'Adam',
            'contrast': True,
            'bidirectional': True,
            'grad_clip': 1.0,
            'lr_main': [], #org 1e-4,9e-5
            'weight_decay_main': [5e-5],
            'lr_bert': [1e-6],
            'weight_decay_bert': 8e-5,
            'lr_mmilb': 0.001,
            'weight_decay_mmilb': 0.0001,
            'alpha': 0.1,
            'dropout_a': 0.1,
            'dropout_v': 0.1,
            'dropout_prj': [],
            'n_layer': 1,
            'cpc_layers': 1,
            'd_vh': 32,
            'd_ah': 32,
            'd_vout': 16,
            'd_aout': 16,
            'd_prjh': 512,
            'scale': 20,
            'beta':[0.5,0.4,0.6],
            'label_len': 4,
            
            
            # enhance
            'enhance_dst_feature_dims': 768,
            'enhance_feat_fusion_interact_num_heads': 8,
            'enhance_feat_fusion_interact_layers': [1],
            'enhance_feat_fusion_attn_dropout':[0.1],
            'enhance_feat_fusion_relu_dropout': [],
            'enhance_feat_fusion_embed_dropout': [0.1],
            'enhance_feat_fusion_res_dropout': [0.1],
            'enhance_feat_fusion_attn_mask': False,
            'enhance_main_dim': 800,
            'enhance_aux_dim': 768,
            'enhance_dropout': [],             
            
            
        }
        return hyper_parameters