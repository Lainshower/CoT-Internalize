#!/bin/bash

# Set your parameters
DATASETS=('gsm8k')
BASE_DATA_PATH="data"
BASE_MODEL_PATH="/data2/joonwon/reasoning-verified/mistral-direct-v9"  # 환경에 맞게 조정
BATCH_SIZE=1
MAX_NEW_TOKENS=400
TOTAL_EPOCHS=5
GPU_ID=2 # 사용할 GPU ID

# Specify the epochs you want to run
EPOCHS=(0 1 2 3 4)

# Specify WARM_UP and VERBOSITY as arrays
WARM_UPS=(0.1)  # 필요에 따라 값 추가
VERBOSITIES=(0.0)  # 필요에 따라 값 추가

# Error handling function
handle_error() {
    echo "Error occurred in script at line: $1"
    exit 1
}

trap 'handle_error $LINENO' ERR

# Loop through each dataset
for DATA in "${DATASETS[@]}"; do
    VAL_PATH="${BASE_DATA_PATH}/${DATA}/${DATA}_test.txt"
    for WARM_UP in "${WARM_UPS[@]}"; do
        for VERBOSITY in "${VERBOSITIES[@]}"; do
            for USE_IN_BATCH_NEGATIVE in 0 1; do
                for USE_HARD_NEGATIVE in 0; do
                    # Set option flags and tags based on activation
                    IN_BATCH_NEGATIVE_TAG=$([ $USE_IN_BATCH_NEGATIVE -eq 1 ] && echo "in-batch-neg" || echo "no-in-batch-neg")
                    HARD_NEGATIVE_TAG=$([ $USE_HARD_NEGATIVE -eq 1 ] && echo "hard-neg" || echo "no-hard-neg")

                    MODEL="${BASE_MODEL_PATH}/${DATA}/bf16_lr5e-06_total_epochs${TOTAL_EPOCHS}_update-data_warm${WARM_UP}_verbos${VERBOSITY}_${IN_BATCH_NEGATIVE_TAG}_${HARD_NEGATIVE_TAG}"
                    BASE_CHECK_POINTS="${MODEL}/best_model/epoch_"

                    # Loop through each epoch
                    for EPOCH in "${EPOCHS[@]}"; do
                        CHECK_POINTS="${BASE_CHECK_POINTS}${EPOCH}/"
                        SAVE="${MODEL}/best_model/epoch_${EPOCH}/test-output"

                        # Create save directory
                        mkdir -p "$SAVE"

                        # Run the evaluate script
                        echo "Running evaluation for ${DATA}, epoch ${EPOCH}, warm-up ${WARM_UP}, verbosity ${VERBOSITY}, ${IN_BATCH_NEGATIVE_TAG}, ${HARD_NEGATIVE_TAG}"
                        CUDA_VISIBLE_DEVICES=$GPU_ID python src/evaluate.py \
                            --val_path "$VAL_PATH" \
                            --checkpoint_path "$CHECK_POINTS" \
                            --save_path "$SAVE" \
                            --batch_size "$BATCH_SIZE" \
                            --max_new_tokens "$MAX_NEW_TOKENS" \
                            > "${SAVE}/log.test" 2>&1 || handle_error $LINENO

                        echo "Evaluation complete. Output saved to ${SAVE}/log.test"
                        
                        # Optional: Add a small delay between runs if needed
                        sleep 5
                    done
                done
            done
        done
    done
done

echo "All evaluations completed successfully."