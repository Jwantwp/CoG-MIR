#!/usr/bin bash

for dataset in 'MIntRec2.0'
do
    for seed in 4 3
    do
        python run.py \
        --dataset $dataset \
        --logger_name 'mmim' \
        --method 'mmim' \
        --data_mode 'multi-class' \
        --train \
        --save_results \
        --seed $seed \
        --gpu_id '0' \
        --data_path '' \
        --video_feats_path 'MIntRec2.0_video_pool.pkl' \
        --audio_feats_path 'MIntRec2.0_wavlm_feats_pool.pkl' \
        --enhance_llm_feats_path 'pro_MIntRec2.0_reason_all.pkl' \
        --text_backbone 'bert-base-uncased' \
        --bert_base_uncased_path 'bert/uncased_L-12_H-768_A-12' \
        --config_file_name 'mmim_mintrec2' \
        --results_file_name 'results_mmim_mintrec2.0_main_seed_4.csv'
    done
done