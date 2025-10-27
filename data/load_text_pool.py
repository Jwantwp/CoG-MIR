import pickle
import os
import numpy as np


def load_feats_text_pool(data_args, feats_path):

    if not os.path.exists(feats_path):
        raise Exception('Error: The directory of video features is empty.')  

    with open(feats_path, 'rb') as f:
        feats = pickle.load(f)
    
    train_feats=[]
    for train_index in data_args['train_data_index']:
        train_index_name=f"{train_index}"
        feat= feats[train_index_name]
        # print("========",feat.shape)
        # if feat.nidm == 3:
        feat = np.squeeze(feat,axis=0)
        train_feats.append(feat)
        
    
    dev_feats=[]
    for dev_index in data_args['dev_data_index']:
        dev_index_name=f"{dev_index}"
        feat= feats[dev_index_name]
        # if feat.nidm == 3:
        feat = np.squeeze(feat,axis=0)
        dev_feats.append(feat)
        
        
    test_feats=[]
    for test_index in data_args['test_data_index']:
        test_index_name=f"{test_index}"
        feat= feats[test_index_name]
        # if feat.nidm == 3:
        feat = np.squeeze(feat,axis=0)
        test_feats.append(feat)

    outputs = {
        'train': train_feats,
        'dev': dev_feats,
        'test': test_feats
    }
    return outputs





def get_text_feats_pool(data_args, feats_paths):


    """
    {
        'train': {'video_text': [feat1, feat2, ...],'audio_text': [feat1, feat2, ...]},
        'dev': {'video_text': [feat1, feat2, ...],'audio_text': [feat1, feat2, ...]},
        'test': {'video_text': [feat1, feat2, ...],'audio_text': [feat1, feat2, ...]}
    }
    
    """

    item={}
    
    speaker_text_feat_pool = load_feats_text_pool(data_args, feats_paths['speak_text_pool_path'])
    # audio_text_feat = load_feats_text(data_args, feats_paths['audio_text_path'])
    
    
    item['train']={'speaker_text':speaker_text_feat_pool['train']}
    item['dev']={'speaker_text':speaker_text_feat_pool['dev']}
    item['test']={'speaker_text':speaker_text_feat_pool['test']}
    
    return item







