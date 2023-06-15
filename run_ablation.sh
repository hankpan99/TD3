#!/bin/bash

for seed in 0 1 2 3
do
    for env in Walker2d-v3 Ant-v3
    do
        for ablation in delay noise clip noise_clip delay_clip delay_noise delay_noise_clip all
        do 
            python main.py \
            --policy "TD3" \
            --env $env \
            --seed $seed \
            --ablation $ablation
        done
    done
done