#!/bin/bash

# List of datasets to process
DATASETS="gsm8k math_qa aqua_rat commonsenseqa trivia_qa strategy-qa"

# Common parameters
BASE_MODEL="mistralai/Mistral-7B-v0.1"
EPOCHS=10
BATCH_SIZE=3
WARM_UP=0.1
LEARNING_RATE=5e-06
ACUMULATE=4
GRAD_NORM=1.0
WEIGHT_DECAY=0.005

for DATA in $DATASETS
do
    TRAIN_PATH="data/${DATA}/${DATA}_train.txt"
    VAL_PATH="data/${DATA}/${DATA}_valid.txt"

    SAVE_M="/data2/joonwon/reasoning-verified/mistral-original/${DATA}/bf16_lr${LEARNING_RATE}_total_epochs${EPOCHS}"
    SAVE_D="/data2/joonwon/reasoning-verified/mistral-original/${DATA}/data-original_lr${LEARNING_RATE}_total_epochs${EPOCHS}"

    # Create save directory and log file
    mkdir -p $SAVE_M
    mkdir -p $SAVE_D

    # Run the training script
    CUDA_VISIBLE_DEVICES=1 python src/train.py \
        --train_path $TRAIN_PATH \
        --val_path $VAL_PATH \
        --save_model $SAVE_M \
        --save_data $SAVE_D \
        --base_model $BASE_MODEL \
        --epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --warmup_ratio $WARM_UP \
        --hidden_improve $HIDDEN_IMPROVE \
        --hidden_interval $HIDDEN_INTERVAL \
        --improved_ratio $IMPROVED_RATIO \
        --lr $LEARNING_RATE \
        --weight_decay $WEIGHT_DECAY \
        --accumulate $ACUMULATE \
        --max_grad_norm $GRAD_NORM \
        --bf16 \
        --train_orig \
        > ${SAVE_M}/log.train_orig 2>&1

    # Clear CUDA memory
    python -c "import torch; torch.cuda.empty_cache()"

    # Optional: wait for a short time to ensure memory is cleared
    sleep 10
done