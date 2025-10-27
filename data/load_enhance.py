import pickle
import os


def load_feats_enhance_infer(data_args, feats_path):

    with open(feats_path, 'rb') as f:
        feats = pickle.load(f)
    train_feats=[]
    for train_index in data_args['train_data_index']:
        train_index_name=train_index
        train_feats.append(feats[train_index_name])
    dev_feats=[]
    for dev_index in data_args['dev_data_index']:
        dev_index_name=dev_index
        dev_feats.append(feats[dev_index_name])
    test_feats=[]
    for test_index in data_args['test_data_index']:
        test_index_name=test_index
        test_feats.append(feats[test_index_name])

    outputs = {
        'train': train_feats,
        'dev': dev_feats,
        'test': test_feats
    }
    return outputs
    


def get_enhance_feats(data_args, feats_path, max_seq_len=128):
    if not os.path.exists(feats_path):
        raise Exception('Error: The directory of video features is empty.')    
    item={}
    feats = load_feats_enhance_infer(data_args, feats_path)
    train_feats=feats['train']
    dev_feats=feats['dev']
    test_feats=feats['test']
    
    
    train_length=[max_seq_len]*len(train_feats)
    dev_length=[max_seq_len]*len(dev_feats)
    test_length=[max_seq_len]*len(test_feats)
    
    
    
    # item['train']={'feats':train_feats,'lengths':train_length}
    # item['dev']={'feats':dev_feats,'lengths':dev_length}
    # item['test']={'feats':test_feats,'lengths':test_length}
    # item['train']={'feats':train_feats,'lengths':train_length}
    # item['dev']={'feats':dev_feats,'lengths':dev_length}
    # item['test']={'feats':test_feats,'lengths':test_length}
    item['train']=train_feats
    item['dev']=dev_feats
    item['test']=test_feats
    return item


