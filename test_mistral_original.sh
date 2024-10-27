#!/bin/bash

# Set your parameters
DATASETS=('gsm8k' 'commonsenseqa'  'math_qa' 'trivia_qa' 'strategy-qa' 'aqua_rat')
BASE_DATA_PATH="data"
BASE_MODEL_PATH="/data2/joonwon/reasoning-verified/mistral-original"
BATCH_SIZE=1
MAX_NEW_TOKENS=400

# Specify the epochs you want to run
EPOCHS=(0 1 2 3 4 5 6 7)

# Loop through each dataset
for DATA in "${DATASETS[@]}"; do
    VAL_PATH="${BASE_DATA_PATH}/${DATA}/${DATA}_test.txt"
    MODEL="${BASE_MODEL_PATH}/${DATA}/bf16_lr5e-06_total_epochs10"
    BASE_CHECK_POINTS="${MODEL}/best_model/epoch_"

    # Loop through each epoch
    for EPOCH in "${EPOCHS[@]}"; do
        CHECK_POINTS="${BASE_CHECK_POINTS}${EPOCH}/"
        SAVE="${MODEL}/best_model/epoch_${EPOCH}/test-output"

        # Create save directory and log file
        mkdir -p $SAVE

        # Run the evaluate script
        CUDA_VISIBLE_DEVICES=3 python src/evaluate.py \
            --val_path $VAL_PATH \
            --checkpoint_path $CHECK_POINTS \
            --save_path $SAVE \
            --batch_size $BATCH_SIZE \
            --max_new_tokens $MAX_NEW_TOKENS \
            > ${SAVE}/log.test 2>&1 

        # Optional: Add a small delay between runs if needed
        sleep 5
    done
done
