import pickle
import os
import numpy as np


def load_feats_text(data_args, feats_path):

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





def get_text_feats(data_args, feats_paths):


    """
    {
        'train': {'video_text': [feat1, feat2, ...],'audio_text': [feat1, feat2, ...]},
        'dev': {'video_text': [feat1, feat2, ...],'audio_text': [feat1, feat2, ...]},
        'test': {'video_text': [feat1, feat2, ...],'audio_text': [feat1, feat2, ...]}
    }
    
    """

    item={}
    
    video_text_feat = load_feats_text(data_args, feats_paths['video_text_path'])
    audio_text_feat = load_feats_text(data_args, feats_paths['audio_text_path'])
    
    
    item['train']={'video_text':video_text_feat['train'],'audio_text':audio_text_feat['train']}
    item['dev']={'video_text':video_text_feat['dev'],'audio_text':audio_text_feat['dev']}
    item['test']={'video_text':video_text_feat['test'],'audio_text':audio_text_feat['test']}
    
    return item






























# def get_text_feats(data_args, feats_paths):


#     item={}
#     # feats = load_feats_text(data_args, feats_path)
    
    
    
#     # train_feats=feats['train']
#     # dev_feats=feats['dev']
#     # test_feats=feats['test']
    
    
#     speakrt_text_feat = load_feats_text(data_args, feats_paths['speakrt_text_path'])
#     video_text_au_feat = load_feats_text(data_args, feats_paths['video_text_au_path'])
#     video_text_head_feat = load_feats_text(data_args, feats_paths['video_text_head_path'])
#     video_text_hand_feat = load_feats_text(data_args, feats_paths['video_text_hand_path'])
#     video_text_body_feat = load_feats_text(data_args, feats_paths['video_text_body_path'])
    
    
    
#     audio_text_sex_feat = load_feats_text(data_args, feats_paths['audio_text_sex_path'])
#     audio_text_mean_pitch_feat = load_feats_text(data_args, feats_paths['audio_text_mean_pitch_path'])
#     audio_text_mean_intensity_feat = load_feats_text(data_args, feats_paths['audio_text_mean_intensity_path'])
#     audio_text_ratio_speech_feat = load_feats_text(data_args, feats_paths['audio_text_ratio_speech_path'])
#     audio_text_tone_feat = load_feats_text(data_args, feats_paths['audio_text_tone_path'])
#     audio_text_speed_rate_feat = load_feats_text(data_args, feats_paths['audio_text_speed_rate_path'])
#     audio_text_pitch_change_feat = load_feats_text(data_args, feats_paths['audio_text_pitch_change_path'])
#     audio_text_intensity_change_feat = load_feats_text(data_args, feats_paths['audio_text_intensity_change_path'])
    
    
#     item['train']={'speakrt_text':speakrt_text_feat['train'],'video_text_au':video_text_au_feat['train'],'video_text_head':video_text_head_feat['train'],'video_text_hand':video_text_hand_feat['train'],'video_text_body':video_text_body_feat['train'],
#                    'audio_text_sex':audio_text_sex_feat['train'],'audio_text_mean_pitch':audio_text_mean_pitch_feat['train'],'audio_text_mean_intensity':audio_text_mean_intensity_feat['train'],'audio_text_ratio_speech':audio_text_ratio_speech_feat['train'],
#                    'audio_text_tone':audio_text_tone_feat['train'],'audio_text_speed_rate':audio_text_speed_rate_feat['train'],'audio_text_pitch_change':audio_text_pitch_change_feat['train'],'audio_text_intensity_change':audio_text_intensity_change_feat['train']
#                    }
    
#     item['dev']={'speakrt_text':speakrt_text_feat['dev'],'video_text_au':video_text_au_feat['dev'],'video_text_head':video_text_head_feat['dev'],'video_text_hand':video_text_hand_feat['dev'],'video_text_body':video_text_body_feat['dev'],
#                     'audio_text_sex':audio_text_sex_feat['dev'],'audio_text_mean_pitch':audio_text_mean_pitch_feat['dev'],'audio_text_mean_intensity':audio_text_mean_intensity_feat['dev'],'audio_text_ratio_speech':audio_text_ratio_speech_feat['dev'],
#                     'audio_text_tone':audio_text_tone_feat['dev'],'audio_text_speed_rate':audio_text_speed_rate_feat['dev'],'audio_text_pitch_change':audio_text_pitch_change_feat['dev'],'audio_text_intensity_change':audio_text_intensity_change_feat['dev']
#     }
#     item['test']={'speakrt_text':speakrt_text_feat['test'],'video_text_au':video_text_au_feat['test'],'video_text_head':video_text_head_feat['test'],'video_text_hand':video_text_hand_feat['test'],'video_text_body':video_text_body_feat['test'],
#                    'audio_text_sex':audio_text_sex_feat['test'],'audio_text_mean_pitch':audio_text_mean_pitch_feat['test'],'audio_text_mean_intensity':audio_text_mean_intensity_feat['test'],'audio_text_ratio_speech':audio_text_ratio_speech_feat['test'],
#                    'audio_text_tone':audio_text_tone_feat['test'],'audio_text_speed_rate':audio_text_speed_rate_feat['test'],'audio_text_pitch_change':audio_text_pitch_change_feat['test'],'audio_text_intensity_change':audio_text_intensity_change_feat['test']
#                }
    
#     return item






# if __name__ == '__main__':
    
    
#     pass