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
            'feats_processing_type': 'padding',
            'padding_mode': 'zero',
            'padding_loc': 'end',
            'need_aligned': True,
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
            warmup_proportion (float): The warmup ratio for learning rate.
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation. 
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """
        hyper_parameters = {
            'warmup_proportion': 0.1,
            'grad_clip': [-1.0],
            'lr': [2e-5],
            'weight_decay': 0.1,
            'mag_aligned_method': ['ctc'],  # alter ['ctc']
            # parameters of similarity-based modality alignment
            'aligned_method': ['sim'],
            'shared_dim': [256],
            'eps': 1e-9,
            # parameters of NT-Xent
            'loss': 'SupCon',
            'temperature': [0.5], 
            # parameters of multimodal fusion
            'beta_shift': [], 
            'dropout_prob': [], 
            # parameters of modality-aware prompting
            'use_ctx': True,
            'prompt_len': 3, 
            'nheads': [8], 
            'n_levels': [5], 
            'attn_dropout': [0.1], 
            'relu_dropout': 0.0, 
            'embed_dropout': [0.2], 
            'res_dropout': 0.1,
            'attn_mask': True,
            'label_len': 4,
            

            # enhance_llm
            'enhance_dst_feature_dims': 768,
            'enhance_feat_fusion_interact_num_heads': 8,
            'enhance_feat_fusion_interact_layers': [1],
            'enhance_feat_fusion_attn_dropout':[],
            'enhance_feat_fusion_relu_dropout': [0.1],
            'enhance_feat_fusion_embed_dropout': [],
            'enhance_feat_fusion_res_dropout': [0.1],
            'enhance_feat_fusion_attn_mask': False,
            'enhance_classifier_dropout': [0.1],
            'enhance_transformer_dropout': [],
            'num_transformer_layers_per_fusion': 1,
     
            
            
        }
        return hyper_parameters 