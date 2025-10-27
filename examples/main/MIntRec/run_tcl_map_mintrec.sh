#!/usr/bin/bash

for seed in 0 1 2 3 4
do
    for method in 'tcl_map' 
    do
        for text_backbone in 'bert-base-uncased'
        do
            python run.py \
            --dataset 'MIntRec' \
            --logger_name ${method} \
            --method ${method} \
            --tune \
            --train \
            --save_results \
            --seed $seed \
            --gpu_id '2' \
            --text_backbone $text_backbone \
            --data_path '' \
            --video_feats_path 'MIntRec_video_pool.pkl' \
            --audio_feats_path 'MIntRec_wavlm_feats_pool.pkl' \
            --enhance_llm_feats_path 'pro_MIntRec_reason_all.pkl' \
            --bert_base_uncased_path 'bert/uncased_L-12_H-768_A-12' \
            --config_file_name "tcl_map_mintrec" \
            --results_file_name "results_tcl_map_mintrec_main.csv"
        done
    done
done