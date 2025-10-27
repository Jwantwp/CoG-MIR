import os
import logging
import csv
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from .mm_pre import MMDataset, AuGDataset,MMDataset_GOBI
# from .text_pre import TextDataset
# from .video_pre import VideoDataset
# from .audio_pre import AudioDataset
from .__init__ import benchmarks
from .text_pre_new import get_t_data
from .load_video_audio import get_video_feats,get_audio_feats
from .load_enhance import get_enhance_feats

__all__ = ['DataManager']


class DataManager:
    
    def __init__(self, args, logger_name = 'Multimodal Intent Recognition'):
        
        self.logger = logging.getLogger(logger_name)
        self.data_path = os.path.join(args.data_path, args.dataset)
        self.benchmarks = benchmarks[args.dataset]
        self.label_list = self.benchmarks["intent_labels"]
        self.logger.info('Lists of intent labels are: %s', str(self.label_list))
        args.num_labels = len(self.label_list)
        args.video_seq_len, args.audio_seq_len = self.benchmarks['max_seq_lengths']['video'], self.benchmarks['max_seq_lengths']['audio']


        text_max_lengths=self.benchmarks['max_seq_lengths']
        # args.text_seq_len, args.video_text_seq_len, args.audio_text_seq_len, args.text_video_seq_len, args.text_audio_seq_len = \
        #     text_max_lengths['text']['text_speaker'], text_max_lengths['text']['video_text'], text_max_lengths['text']['audio_text'], text_max_lengths['text']['text_video'], text_max_lengths['text']['text_audio']


        if args.method == 'gobi':
            
            args.text_seq_len = text_max_lengths['text']['text_speaker']
        else:
            args.text_seq_len = text_max_lengths['text']['base_text']
        

        
        self.text_all_data_path={}
        # self.text_all_data_path['video_text_path']=os.path.join(self.data_path,args.video_text_path)
        # self.text_all_data_path['audio_text_path']=os.path.join(self.data_path,args.audio_text_path)
        # self.text_all_data_path['speak_text_pool_path']=os.path.join(self.data_path,args.text_pool_path)
        
        
        self.enhance_feats_path=os.path.join(self.data_path,args.enhance_llm_feats_path)
        
        
        self.train_data_index, self.train_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'train.tsv'), args)
        self.dev_data_index, self.dev_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'dev.tsv'), args)
        self.test_data_index, self.test_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'test.tsv'), args)
        args.num_train_examples = len(self.train_data_index)
        
        self.index={
            'train_data_index':self.train_data_index,
            'dev_data_index':self.dev_data_index,
            'test_data_index':self.test_data_index
        }


        args.text_all_data_path=self.text_all_data_path

        args.text_feat_dim, args.video_feat_dim, args.audio_feat_dim=self.benchmarks['feat_dims']['text'],self.benchmarks['feat_dims']['video'],self.benchmarks['feat_dims']['audio'],



        
        
        if args.aug:
            self.aug_data_index, self.aug_label_ids = self._get_indexes_annotations(os.path.join(self.data_path, 'augment_train.tsv'), args)
            
        self.unimodal_feats = self._get_unimodal_feats(args, self.data_path,self.index)


        self.mm_data = self._get_multimodal_data(args)
        self.mm_dataloader = self._get_dataloader(args, self.mm_data)        
        
        

    def _get_unimodal_feats(self, args, data_path, index):
        
        video_feats_path=os.path.join(self.data_path,args.video_data_path,args.video_feats_path)
        audio_feats_path=os.path.join(self.data_path,args.audio_data_path,args.audio_feats_path)  
        

        text_feats = get_t_data(args,data_path)
        video_feats = get_video_feats(index,video_feats_path,args.video_seq_len)
        audio_feats = get_audio_feats(index,audio_feats_path,args.video_seq_len)
        enhance_llm_feats=get_enhance_feats(index,self.enhance_feats_path)

        return {
            'text': text_feats,
            'video': video_feats,
            'audio': audio_feats,
            'enhance_llm':enhance_llm_feats
        }
        
        
        
            
    def _get_multimodal_data(self, args):

        text_data = self.unimodal_feats['text']['text_feats']
        video_data = self.unimodal_feats['video']
        audio_data = self.unimodal_feats['audio']
        enhance_llm_data = self.unimodal_feats['enhance_llm']
        
        
        text = self.unimodal_feats['text']
        
        other_data = self._get_other_data(text)

        mm_train_data = MMDataset(self.train_label_ids, text_data['train'], video_data['train'], audio_data['train'], other_data['train'], enhance_llm_data['train'])
        mm_dev_data = MMDataset(self.dev_label_ids, text_data['dev'], video_data['dev'], audio_data['dev'], other_data['dev'], enhance_llm_data['dev'])
        mm_test_data = MMDataset(self.test_label_ids, text_data['test'], video_data['test'], audio_data['test'], other_data['test'], enhance_llm_data['test'])

        
        if args.aug:
            mm_aug_data = AuGDataset(self.aug_label_ids, text_data['aug'])
            return {
                'train': mm_train_data,
                'aug': mm_aug_data,
                'dev': mm_dev_data,
                'test': mm_test_data
            }
        
        return {
            'train': mm_train_data,
            'dev': mm_dev_data,
            'test': mm_test_data
        }

    def _get_dataloader(self, args, data):
        
        self.logger.info('Generate Dataloader Begin...')

        train_dataloader = DataLoader(data['train'], shuffle=True, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
        dev_dataloader = DataLoader(data['dev'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        test_dataloader = DataLoader(data['test'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        
        self.logger.info('Generate Dataloader Finished...')

        if args.aug:
            aug_dataloader = DataLoader(data['aug'], shuffle=True, batch_size = args.aug_batch_size)
            return {
                'train': train_dataloader,
                'aug': aug_dataloader,
                'dev': dev_dataloader,
                'test': test_dataloader
            }
                    
        return {
            'train': train_dataloader,
            'dev': dev_dataloader,
            'test': test_dataloader
        }
        
    def _get_attrs(self):

        attrs = {}
        for name, value in vars(self).items():
            attrs[name] = value

        return attrs
    
    def _get_other_data(self, inputs):
        other_data = {}
        other_data['train'] = {}
        other_data['dev'] = {}
        other_data['test'] = {}

        for key in inputs.keys():
            if key not in ['text_feats']:
                if 'train' in inputs[key]:
                    other_data['train'][key] = inputs[key]['train']
                if 'dev' in inputs[key]:
                    other_data['dev'][key] = inputs[key]['dev']
                if 'test' in inputs[key]:
                    other_data['test'][key] = inputs[key]['test']

        return other_data
            
    def _get_indexes_annotations(self, read_file_path, args):


        # args可以用作为未来的二分类添加做铺垫
        label_map = {}
        for i, label in enumerate(self.label_list):
            label_map[label] = i
    
        with open(read_file_path, 'r') as f:
            csv_reader = csv.reader(f, delimiter='\t')
            next(csv_reader)
            labels_ids = []
            names = []
            for line in csv_reader:
                label=line[2].strip()
                # if data_mode == 'multi-class':
                label_id = label_map[label]
                
                labels_ids.append(label_id)
                names.append(line[0])
            
        return names,labels_ids 


    def _get_multimodal_data_gobi(self,args):

        text_data = self.unimodal_feats['text']
        video_data = self.unimodal_feats['video']
        audio_data = self.unimodal_feats['audio']
        

        mm_train_data = MMDataset_GOBI(self.train_label_ids, text_data['text_feats']['train']['speaker_text'],text_data['text_feats_pool']['train']['speaker_text'],text_data['video_audio_text_feats']['train']['video_text'],text_data['video_audio_text_feats']['train']['audio_text'],video_data['train'], audio_data['train'])

        mm_dev_data = MMDataset_GOBI(self.dev_label_ids, text_data['text_feats']['dev']['speaker_text'],text_data['text_feats_pool']['dev']['speaker_text'],text_data['video_audio_text_feats']['dev']['video_text'],text_data['video_audio_text_feats']['dev']['audio_text'],video_data['dev'], audio_data['dev'])
                                     
        mm_test_data = MMDataset_GOBI(self.test_label_ids, text_data['text_feats']['test']['speaker_text'],text_data['text_feats_pool']['test']['speaker_text'],text_data['video_audio_text_feats']['test']['video_text'],text_data['video_audio_text_feats']['test']['audio_text'],video_data['test'],audio_data['test'])
        
        
        return {
            'train': mm_train_data,
            'dev': mm_dev_data,
            'test': mm_test_data
        }
        
        

    def _get_dataloader_gobi(self, args, data):
        
        self.logger.info('Generate Dataloader Begin...')

        train_dataloader = DataLoader(data['train'], shuffle=True, batch_size = args.train_batch_size, num_workers = args.num_workers, pin_memory = True)
        dev_dataloader = DataLoader(data['dev'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        test_dataloader = DataLoader(data['test'], batch_size = args.eval_batch_size, num_workers = args.num_workers, pin_memory = True)
        
        self.logger.info('Generate Dataloader Finished...')
                    
        return {
            'train': train_dataloader,
            'dev': dev_dataloader,
            'test': test_dataloader
        }


class TextDataset(Dataset):
    def __init__(self, label_ids, text_feats):

        self.label_ids = torch.tensor(label_ids)
        self.text_feats = torch.tensor(text_feats)
        self.size = len(self.text_feats)
    def __len__(self):
        return self.size
    def __getitem__(self, index):

        sample = {
            'text_feats': self.text_feats[index],
            'label_ids': self.label_ids[index], 
        } 
        return sample
