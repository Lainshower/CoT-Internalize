#!/bin/sh

DATA='gsm8k'
TRAIN_PATH="data/${DATA}/${DATA}_train.txt"
VAL_PATH="data/${DATA}/${DATA}_valid.txt"
BASE_MODEL="mistralai/Mistral-7B-v0.1"
EPOCHS=5
BATCH_SIZE=3
LEARNING_RATE=5e-06
ACUMULATE=4
GRAD_NORM=1.0
WEIGHT_DECAY=0.005
UPDATE=1

# Loop over WARM_UP values
for WARM_UP in 0.1; do
  # Loop over VERBOSITY values
  for VERBOSITY in 0.0; do
    # Loop over use_in_batch_negative values (0: disabled, 1: enabled)
    for USE_IN_BATCH_NEGATIVE in 1 0; do # for USE_IN_BATCH_NEGATIVE in 0 1; do
      # Loop over use_hard_negative values (0: disabled, 1: enabled)
      for USE_HARD_NEGATIVE in 0; do # for USE_HARD_NEGATIVE in 0 1; do

        # Set option flags and tags based on activation
        if [ $USE_IN_BATCH_NEGATIVE -eq 1 ]; then
          IN_BATCH_NEGATIVE_OPTION="--use_in_batch_negative"
          IN_BATCH_NEGATIVE_TAG="in-batch-neg"
        else
          IN_BATCH_NEGATIVE_OPTION=""
          IN_BATCH_NEGATIVE_TAG="no-in-batch-neg"
        fi

        if [ $USE_HARD_NEGATIVE -eq 1 ]; then
          HARD_NEGATIVE_OPTION="--use_hard_negative"
          HARD_NEGATIVE_TAG="hard-neg"
        else
          HARD_NEGATIVE_OPTION=""
          HARD_NEGATIVE_TAG="no-hard-neg"
        fi

        if [ $UPDATE -eq 1 ]; then
          UPDATE_OPITON="--update"
          UPDATE_TAG="update-data"
        else
            UPDATE_OPITON=""
            UPDATE_TAG="non-update-data"
        fi

        # Construct SAVE_M and SAVE_D paths including the new tags
        SAVE_M="/data2/joonwon/reasoning-verified/mistral-direct-v9/${DATA}/bf16_lr${LEARNING_RATE}_total_epochs${EPOCHS}_${UPDATE_TAG}_warm${WARM_UP}_verbos${VERBOSITY}_${IN_BATCH_NEGATIVE_TAG}_${HARD_NEGATIVE_TAG}"
        SAVE_D="/data2/joonwon/reasoning-verified/mistral-direct-v9/${DATA}/bf16_lr${LEARNING_RATE}_total_epochs${EPOCHS}_${UPDATE_TAG}_warm${WARM_UP}_verbos${VERBOSITY}_${IN_BATCH_NEGATIVE_TAG}_${HARD_NEGATIVE_TAG}"

        mkdir -p $SAVE_M
        mkdir -p $SAVE_D

        # Run the training script with the appropriate options
        CUDA_VISIBLE_DEVICES=2 python src/train_direct_v9.py \
            --train_path $TRAIN_PATH \
            --val_path $VAL_PATH \
            --save_model $SAVE_M \
            --save_data $SAVE_D \
            --base_model $BASE_MODEL \
            --epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --warmup_ratio $WARM_UP \
            $IN_BATCH_NEGATIVE_OPTION \
            $HARD_NEGATIVE_OPTION \
            --verbosity_threshold $VERBOSITY \
            --lr $LEARNING_RATE \
            --weight_decay $WEIGHT_DECAY \
            --accumulate $ACUMULATE \
            --max_grad_norm $GRAD_NORM \
            --bf16 \
            $UPDATE_OPITON \
            > "${SAVE_M}/log.train" 2>&1

      done
    done
  done
done
