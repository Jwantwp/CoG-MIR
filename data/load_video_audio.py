import pickle
import os







def load_feats_video(data_args, feats_path):

    with open(feats_path, 'rb') as f:
        feats = pickle.load(f)

    # print("======================",feats_path)
    train_feats=[]
    for train_index in data_args['train_data_index']:
        # train_index_name=train_index.replace("MIntRec_", "")
        train_index_name=f"{train_index}.mp4"
        # print("+++++++++++++",train_index_name)
        # print(len(feats[train_index_name]))
        train_feats.append(feats[train_index_name])
        
    
    dev_feats=[]
    for dev_index in data_args['dev_data_index']:
        # dev_index_name=f"{dev_index}".replace("MIntRec_", "")
        dev_index_name=f"{dev_index}.mp4"
        dev_feats.append(feats[dev_index_name])
        
        
    test_feats=[]
    for test_index in data_args['test_data_index']:
        # test_index_name=f"{test_index}".replace("MIntRec_", "")
        test_index_name=f"{test_index}.mp4"
        test_feats.append(feats[test_index_name])

    outputs = {
        'train': train_feats,
        'dev': dev_feats,
        'test': test_feats
    }
    return outputs
        


def get_video_feats(data_args, feats_path, max_seq_len):
    if not os.path.exists(feats_path):
        raise Exception('Error: The directory of video features is empty.')    
    item={}
    feats = load_feats_video(data_args, feats_path)
    train_feats=feats['train']
    dev_feats=feats['dev']
    test_feats=feats['test']
    
    train_length=[max_seq_len]*len(train_feats)
    dev_length=[max_seq_len]*len(dev_feats)
    test_length=[max_seq_len]*len(test_feats)
    
    item['train']={'feats':train_feats,'lengths':train_length}
    item['dev']={'feats':dev_feats,'lengths':dev_length}
    item['test']={'feats':test_feats,'lengths':test_length}
    

    # return feats
    return item

    # 我提取的音频和视频特征都已经达到了最大长度,所以不用对其进行填充,这里直接注释掉,但代码保留,以防未来提取特征方式的改变
    # data = padding_feats(feats, max_seq_len)
    # return data



def load_feats_audio(data_args, feats_path):
    with open(feats_path, 'rb') as f:
        feats = pickle.load(f)

    train_feats=[]
    for train_index in data_args['train_data_index']:
        # train_index_name=f"{train_index}".replace("MIntRec_", "")
        train_index_name=f"{train_index}.wav"
        train_feats.append(feats[train_index_name])
    dev_feats=[]
    for dev_index in data_args['dev_data_index']:
        # dev_index_name=f"{dev_index}".replace("MIntRec_", "")
        dev_index_name=f"{dev_index}.wav"
        dev_feats.append(feats[dev_index_name])
    test_feats=[]
    for test_index in data_args['test_data_index']:
        # test_index_name=f"{test_index}".replace("MIntRec_", "")
        test_index_name=f"{test_index}.wav"
        test_feats.append(feats[test_index_name])
        


    outputs = {
        'train': train_feats,
        'dev': dev_feats,
        'test': test_feats
    }
    return outputs




def get_audio_feats(data_args, feats_path, max_seq_len):
    if not os.path.exists(feats_path):
        raise Exception('Error: The directory of audio features is empty.')    

    feats = load_feats_audio(data_args, feats_path)
    # return feats
    item={}
    train_feats=feats['train']
    dev_feats=feats['dev']
    test_feats=feats['test']
    
    train_length=[max_seq_len]*len(train_feats)
    dev_length=[max_seq_len]*len(dev_feats)
    test_length=[max_seq_len]*len(test_feats)
    
    item['train']={'feats':train_feats,'lengths':train_length}
    item['dev']={'feats':dev_feats,'lengths':dev_length}
    item['test']={'feats':test_feats,'lengths':test_length}    
    
    return item
    # data = padding_feats(feats, max_seq_len)
    
    # 我提取的音频和视频特征都已经达到了最大长度,所以不用对其进行填充,这里直接注释掉,但代码保留,以防未来提取特征方式的改变
    # data = padding_feats(feats, max_seq_len)
    # return data