#!/bin/bash

# Set the number of GPUs you want to use
NUM_GPUS=3

# Set your parameters
TRAIN_PATH="data/gsm8k/train_orig.txt"
VAL_PATH="data/gsm8k/valid_orig.txt"
export SAVE="train_mistral/gsm8k_orig-test"
BASE_MODEL="mistralai/Mistral-7B-v0.1"
EPOCHS=10
BATCH_SIZE=4
LEARNING_RATE=5e-5
ACUMULATE=2
GRAD_NORM=1.0

# Create save directory and log file
mkdir -p $SAVE

# Run the training script
torchrun --nproc_per_node=$NUM_GPUS \
    src/train_mistral.py \
    --train_path $TRAIN_PATH \
    --val_path $VAL_PATH \
    --save_model $SAVE \
    --base_model $BASE_MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LEARNING_RATE \
    --accumulate $ACUMULATE \
    --max_grad_norm $GRAD_NORM \
    --bf16 \
    > ${SAVE}/log.train_orig 2>&1