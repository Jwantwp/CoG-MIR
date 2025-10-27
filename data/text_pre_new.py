import os
import csv
import sys
import torch
from transformers import BertTokenizer, RobertaTokenizer, T5Tokenizer
from torch.utils.data import Dataset
import numpy as np



class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        
        
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

# 从tsv文件中读取name，text，label，只读出这几个值
def read_tsv(args,input_file,mode):
    if args.method=='gobi':
        data_path= os.path.join(input_file, mode)
        
        with open(data_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            examples_speaker_texts = []
            
            for row in reader:
                speaker_text=row[4]
                
                example_speaker_text= InputExample(guid=row[0], text_a=speaker_text, text_b=None, label=row[2])
                
                examples_speaker_texts.append(example_speaker_text)

            all_examples = [examples_speaker_texts]           
            return all_examples

    else:
        data_path= os.path.join(input_file, mode)
        with open(data_path, "r") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            examples = []
            for row in reader:
                example= InputExample(guid=row[0], text_a=row[1], text_b=None, label=row[2])
                examples.append(example)
            return examples


def get_examples(args,data_dir, mode):
    
    if mode == 'train':
        return read_tsv(args,data_dir, f"{mode}.tsv")
    elif mode == 'dev':
        return read_tsv(args,data_dir, f"{mode}.tsv")
    elif mode == 'test':
        return read_tsv(args,data_dir, f"{mode}.tsv")
    # elif mode == 'all':
    #     return self._create_examples(self._read_tsv(os.path.join(data_dir, "all.tsv")), "all")




def get_examples_my_method(args,all_examples,tokenizer):  
            
        features_speaker_text = convert_examples_to_features(all_examples[0], args.text_seq_len, tokenizer)
        features_speaker_text_ids = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features_speaker_text]


        all_feats={}
        all_feats['speaker_text']=features_speaker_text_ids
        

        return all_feats


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)  # For dialogue context
        else:
            tokens_b.pop()
def tcl_map_convert_examples_to_features(args, examples, tokenizer):
    
    
    """Loads a data file into a list of `InputBatch`s."""
    # label_len = data_args['bm']['label_len']
    max_seq_length = args.text_seq_len
    label_len = args.label_len  # 4
    prompt_len = args.prompt_len  # 3
    
    
    
    features = []
    cons_features = []
    condition_idx = []
    prefix = ['MASK'] * prompt_len
    # prefix = ['MASK'] * data_args['prompt_len']



    max_cons_seq_length = max_seq_length + len(prefix) + label_len
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)
        # if args.dataset in ['MIntRec']:
        condition = tokenizer.tokenize(example.label)
        
        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # construct augmented sample pair
        cons_tokens = ["[CLS]"] + tokens_a + prefix + condition + (label_len - len(condition)) * ["MASK"] + ["[SEP]"]
        tokens = ["[CLS]"] + tokens_a + prefix + label_len * ["[MASK]"] + ["[SEP]"]

        segment_ids = [0] * len(tokens)
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        cons_inputs_ids = tokenizer.convert_tokens_to_ids(cons_tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_cons_seq_length - len(input_ids))
        input_ids += padding
        cons_inputs_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_cons_seq_length
        assert len(cons_inputs_ids) == max_cons_seq_length
        assert len(input_mask) == max_cons_seq_length
        assert len(segment_ids) == max_cons_seq_length
        # record the position of prompt
        condition_idx.append(1 + len(tokens_a) + len(prefix))


        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
        
        cons_features.append(
            InputFeatures(input_ids=cons_inputs_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    return features, cons_features, condition_idx, max_cons_seq_length


def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""
        
    features = []
    for (ex_index, example) in enumerate(examples):
        # print("+++++++++",example.text_a)
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    return features

def get_backbone_feats(args,examples):
    
    # tokenizer = BertTokenizer.from_pretrained(args.text_pretrained_model, do_lower_case=True)
    tokenizer = BertTokenizer.from_pretrained(args.bert_base_uncased_path, do_lower_case=True)  
    

    if args.method == 'tcl_map':
        features, cons_features, condition_idx, args.max_cons_seq_length = tcl_map_convert_examples_to_features(args, examples,tokenizer)
        features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]
        cons_features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in cons_features]
        return features_list, cons_features_list, condition_idx
    # 适应于所有基线方法
    else:
        features = convert_examples_to_features(examples, args.text_seq_len, tokenizer)
        features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]
        return features_list    




def get_data(args, data_path):
    # data_path = data_args['text_data_path']
    
    
    if args.method == 'gobi':
        
        # 这个train_examples应该是一个所有的列表
        tokenizer = BertTokenizer.from_pretrained(args.bert_base_uncased_path, do_lower_case=True)
        all_train_examples = get_examples(args,data_path, 'train')
        assert len(all_train_examples) == 1, "Expected 10 types of text examples for my_method"
        
        train_all_feats = get_examples_my_method(args, all_train_examples, tokenizer)
        
        
        all_dev_examples = get_examples(args,data_path, 'dev')
        assert len(all_dev_examples) == 1, "Expected 10 types of text examples for my_method"
        
        dev_all_feats = get_examples_my_method(args, all_dev_examples, tokenizer)
        
        all_test_examples = get_examples(args,data_path, 'test')
        assert len(all_test_examples) == 1, "Expected 10 types of text examples for my_method"
        
        test_all_feats = get_examples_my_method(args, all_test_examples, tokenizer)
        
        text_feats = {
            'train': train_all_feats,
            'dev': dev_all_feats,
            'test': test_all_feats
        }
        
        format_output={'text_feats': text_feats}
        
        

    elif args.method == 'tcl_map':
        train_examples = get_examples(args,data_path, 'train') 
        train_feats, train_cons_text_feats, train_condition_idx = get_backbone_feats(args, train_examples)        
        
        dev_examples = get_examples(args,data_path, 'dev')
        dev_feats, dev_cons_text_feats, dev_condition_idx = get_backbone_feats(args, dev_examples)

        test_examples = get_examples(args,data_path, 'test')
        test_feats, test_cons_text_feats, test_condition_idx = get_backbone_feats(args,test_examples)

        # outputs = {
        #     'train': train_feats,
        #     'train_cons_text_feats': train_cons_text_feats,
        #     'train_condition_idx': train_condition_idx,
        #     'dev': dev_feats,
        #     'dev_cons_text_feats': dev_cons_text_feats,
        #     'dev_condition_idx': dev_condition_idx,
        #     'test': test_feats,
        #     'test_cons_text_feats': test_cons_text_feats,
        #     'test_condition_idx': test_condition_idx,
        # }


        cons_text_feats = {'train': train_cons_text_feats, 'dev': dev_cons_text_feats, 'test': test_cons_text_feats}
        condition_idx = {'train': train_condition_idx, 'dev': dev_condition_idx, 'test': test_condition_idx}
        text_feats={'train':train_feats,'dev':dev_feats,'test':test_feats}
        
        format_output={'text_feats':text_feats,'cons_text_feats':cons_text_feats,'condition_idx':condition_idx}
        
    else:
        train_examples = get_examples(args,data_path, 'train') 
        train_feats = get_backbone_feats(args,  train_examples)

        dev_examples = get_examples(args,data_path, 'dev')
        dev_feats = get_backbone_feats(args,  dev_examples)

        test_examples = get_examples(args,data_path, 'test')
        test_feats = get_backbone_feats(args, test_examples)

        text_feats = {
            'train': train_feats,
            'dev': dev_feats,
            'test': test_feats
        }
        
        format_output={'text_feats': text_feats}
        
        
    return format_output





def get_t_data(args, data_path):
    
    
    
    """
    {
        'text_feats':{
            'train': train_all_feats,
            'dev': dev_all_feats,
            'test': test_all_feats
        }
    }
    """
    
    if args.text_backbone.startswith('bert'):
        t_data = get_data(args, data_path)
    else:
        raise Exception('Error: inputs are not supported text backbones.')

    return t_data








# class TextDataset(Dataset):
    
#     def __init__(self, label_ids, text_feats):
        
#         self.label_ids = torch.tensor(label_ids)
#         self.text_feats = torch.tensor(text_feats)
#         self.size = len(self.text_feats)

#     def __len__(self):
#         return self.size

#     def __getitem__(self, index):

#         sample = {
#             'text_feats': self.text_feats[index],
#             'label_ids': self.label_ids[index], 
#         } 
#         return sample
    
    

