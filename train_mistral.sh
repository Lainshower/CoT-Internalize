#!/bin/bash

# Set the number of GPUs you want to use
# NUM_GPUS=1

# Set your parameters
TRAIN_PATH="data/gsm8k/train_orig.txt"
VAL_PATH="data/gsm8k/valid_orig.txt"
export SAVE_M="mistral/gsm8k_original_bf16"
export SAVE_D="mistral/gsm8k-training-data-original"
BASE_MODEL="mistralai/Mistral-7B-v0.1"
EPOCHS=7
BATCH_SIZE=3
WARM_UP=0.2 #0.2
LEARNING_RATE=5e-05
ACUMULATE=4
GRAD_NORM=1.0
WEIGHT_DECAY=0.005
LOSS_SCALE=0.7

# Create save directory and log file
mkdir -p $SAVE_M
mkdir -p $SAVE_D

# Run the training script
# torchrun --nproc_per_node=$NUM_GPUS --master_port=29700 \
#    
python src/train.py \
    --train_path $TRAIN_PATH \
    --val_path $VAL_PATH \
    --save_model $SAVE_M \
    --save_data $SAVE_D \
    --base_model $BASE_MODEL \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --warmup_ratio $WARM_UP \
    --loss_scale $LOSS_SCALE \
    --lr $LEARNING_RATE \
    --weight_decay $WEIGHT_DECAY \
    --accumulate $ACUMULATE \
    --max_grad_norm $GRAD_NORM \
    --bf16 \
    > ${SAVE_M}/log.train_orig 2>&1&
