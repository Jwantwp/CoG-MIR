#!/usr/bin bash

for dataset in 'MIntRec'
do
    for seed in 0 1 2 3 4 5 6 7 8 9
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'mag_bert' \
        --method 'mag_bert' \
        --data_mode 'multi-class' \
        --train \
        --save_results \
        --seed $seed \
        --gpu_id '3' \
        --data_path '/home/sharing/disk1/disk1/wangpeiwu/wangpeiwu/dlib-19.13_e/Intent_Data/Test' \
        --video_feats_path 'MIntRec_video_pool.pkl' \
        --audio_feats_path 'MIntRec_wavlm_feats_pool.pkl' \
        --text_backbone 'bert-base-uncased' \
        --bert_base_uncased_path '/home/sharing/disk1/disk1/pretrained_embedding/bert/uncased_L-12_H-768_A-12' \
        --config_file_name 'mag_bert_mintrec' \
        --results_file_name 'results_mag_bert_mintrec.csv'
    done
done