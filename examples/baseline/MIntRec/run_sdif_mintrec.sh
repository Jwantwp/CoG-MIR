#!/usr/bin bash

for dataset in 'MIntRec'
do
    for seed in 0
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'sdif' \
        --method 'sdif' \
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
        --config_file_name 'sdif_mintrec' \
        --results_file_name 'results_sdif_mintrec.csv'
    done
done