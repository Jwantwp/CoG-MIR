from torch.utils.data import Dataset
import torch
import numpy as np

__all__ = ['MMDataset', 'AuGDataset','MMDataset_GOBI','MMDataset_WUMO']

class MMDataset(Dataset):
        
    def __init__(self, label_ids, text_data, video_data, audio_data, other_data, enhance_llm_data):
        
        self.label_ids = label_ids
        self.text_data = text_data
        self.video_data = video_data
        self.audio_data = audio_data
        
        self.size = len(self.text_data)

        self.other_data = other_data
        self.enhance_llm_data=enhance_llm_data
        
        if self.other_data is not None:
            for key in other_data.keys():
                setattr(self, key, other_data[key])  
    
    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': torch.tensor(self.label_ids[index]), 
            'text_feats': torch.tensor(self.text_data[index]),
            'video_feats': torch.tensor(np.array(self.video_data['feats'][index])),
            'video_lengths': torch.tensor(np.array(self.video_data['lengths'][index])),
            'audio_feats': torch.tensor(np.array(self.audio_data['feats'][index])),
            'audio_lengths': torch.tensor(np.array(self.audio_data['lengths'][index])),
            'enhance_llm_feats':torch.tensor(self.enhance_llm_data[index]),
        }

        if self.other_data is not None:    
            for key in self.other_data.keys():
                sample[key] = torch.tensor(getattr(self, key)[index])
        
        return sample




class MMDataset_WUMO(Dataset):
    def __init__(self, label_ids, text_speaker_data, video_text_data, audio_text_data, text_video_data, text_audio_data,video_data, audio_data):
        
        self.label_ids = label_ids
        self.text_speaker_data=text_speaker_data
        self.video_text_data =video_text_data
        self.audio_text_data=audio_text_data
        self.text_video_data=text_video_data
        self.text_audio_data=text_audio_data

        self.video_data = video_data
        self.audio_data = audio_data
        self.size = len(self.text_speaker_data)


    
    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': torch.tensor(self.label_ids[index]), 
            'text_speaker_feats':torch.tensor(self.text_speaker_data[index]),
            'video_text_feats':torch.tensor(self.video_text_data[index]),
            'audio_text_feats':torch.tensor(self.audio_text_data[index]),
            'text_video_feats':torch.tensor(self.text_video_data[index]),
            'text_audio_feats':torch.tensor(self.text_audio_data[index]),
            'video_feats': torch.tensor(np.array(self.video_data['feats'][index])),
            'audio_feats': torch.tensor(np.array(self.audio_data['feats'][index])),
        }
        
        
        return sample




class MMDataset_GOBI(Dataset):
    def __init__(self, label_ids, text_speaker_data, text_speaker_pool_data,video_text_data,audio_text_data,video_data, audio_data):

        
        self.label_ids = label_ids
        self.text_speaker_data=text_speaker_data

        self.text_speaker_pool_data = text_speaker_pool_data
        

        self.video_text_data =video_text_data
        self.audio_text_data=audio_text_data       

        
        self.video_data = video_data
        self.audio_data = audio_data
        self.size = len(self.text_speaker_data)


    
    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': torch.tensor(self.label_ids[index]), 
            
            'text_speaker_feats':torch.tensor(self.text_speaker_data[index]),
        
            'text_speaker_pool_feats':torch.tensor(self.text_speaker_pool_data[index]),
    
            'video_text_feats':torch.tensor(self.video_text_data[index]),
            'audio_text_feats':torch.tensor(self.audio_text_data[index]),            
            'video_feats': torch.tensor(np.array(self.video_data['feats'][index])),
            'audio_feats': torch.tensor(np.array(self.audio_data['feats'][index])),

        }

        
        return sample







class AuGDataset(Dataset):
        
    def __init__(self, label_ids, text_feats):
        
        self.label_ids = torch.tensor(label_ids)
        self.text_feats = torch.tensor(text_feats)
        self.size = len(self.text_feats)

    def __len__(self):
        return self.size

    def __getitem__(self, index):

        sample = {
            'label_ids': self.label_ids[index], 
            'text_feats': self.text_feats[index],
        } 
        return sample