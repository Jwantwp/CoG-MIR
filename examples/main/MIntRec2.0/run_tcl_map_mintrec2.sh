#!/usr/bin/bash

for seed in 0 1 2 3 4
do
    for method in 'tcl_map' 
    do
        for text_backbone in 'bert-base-uncased'
        do
            python run.py \
            --dataset 'MIntRec2.0' \
            --logger_name ${method} \
            --method ${method} \
            --tune \
            --train \
            --save_results \
            --seed $seed \
            --gpu_id '2' \
            --text_backbone $text_backbone \
            --data_path '' \
            --video_feats_path 'MIntRec2.0_video_pool.pkl' \
            --audio_feats_path 'MIntRec2.0_wavlm_feats_pool.pkl' \
            --enhance_llm_feats_path 'pro_MIntRec2.0_reason_all.pkl' \
            --bert_base_uncased_path 'bert/uncased_L-12_H-768_A-12' \
            --config_file_name "tcl_map_mintrec2" \
            --results_file_name "results_tcl_map_mintrec2.0_main.csv"
        done
    done
done