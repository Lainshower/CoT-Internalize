#!/bin/bash

# Set the number of GPUs you want to use
NUM_GPUS=3

# Set your parameters
TRAIN_PATH="data/gsm8k/train_orig.txt"
VAL_PATH="data/gsm8k/valid_orig.txt"
export SAVE_M="mistral/gsm8k_entropy"
export SAVE_D="mistral/gsm8k-training-data"
BASE_MODEL="mistralai/Mistral-7B-v0.1"
EPOCHS=1
BATCH_SIZE=3
WARM_UP=0.2 #0.2
LEARNING_RATE=5e-5
ACUMULATE=1
GRAD_NORM=1.0

# Create save directory and log file
mkdir -p $SAVE_M
mkdir -p $SAVE_D

# Run the training script
torchrun --nproc_per_node=$NUM_GPUS \
    src/train.py \
    --train_path $TRAIN_PATH \
    --val_path $VAL_PATH \
    --save_model $SAVE_M \
    --save_data $SAVE_D \
    --base_model $BASE_MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --warmup_ratio $WARM_UP \
    --lr $LEARNING_RATE \
    --accumulate $ACUMULATE \
    --max_grad_norm $GRAD_NORM \
    > ${SAVE_M}/log.train_orig 2>&1&