#!/usr/bin bash

for dataset in 'MIntRec'
do
    for seed in 0 1 2 3 4
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'mmim' \
        --method 'mmim' \
        --data_mode 'multi-class' \
        --train \
        --save_results \
        --seed $seed \
        --gpu_id '3' \
        --data_path '' \
        --video_feats_path 'MIntRec_video_pool.pkl' \
        --audio_feats_path 'MIntRec_wavlm_feats_pool.pkl' \
        --enhance_llm_feats_path 'pro_MIntRec_reason_all.pkl' \
        --text_backbone 'bert-base-uncased' \
        --bert_base_uncased_path 'bert/uncased_L-12_H-768_A-12' \
        --config_file_name 'mmim_mintrec' \
        --results_file_name 'results_mmim_mintrec_new_10_5.csv'
    done
done