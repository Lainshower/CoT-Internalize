# #!/bin/bash

# # Set your parameters
# VAL_PATH="data/gsm8k/test_orig.txt"
# CHECK_POINTS="mistral/gsm8k_original_1gpu/best_model/epoch_0/"
# export SAVE="mistral/gsm8k_original_1gpu/best_model/epoch_0/test-output"
# BATCH_SIZE=1
# MAX_NEW_TOKENS=400

# # Create save directory and log file
# mkdir -p $SAVE

# # Run the evaluate script
# python src/evaluate.py \
#     --val_path $VAL_PATH \
#     --checkpoint_path $CHECK_POINTS \
#     --save_path $SAVE \
#     --batch_size $BATCH_SIZE \
#     --max_new_tokens $MAX_NEW_TOKENS \
#     > ${SAVE}/log.test 2>&1 &

#!/bin/bash

# Set your parameters
VAL_PATH="data/gsm8k/test_orig.txt"
BASE_CHECK_POINTS="mistral/gsm8k_original_1gpu/best_model/epoch_"
BATCH_SIZE=1
MAX_NEW_TOKENS=400

# Specify the epochs you want to run
EPOCHS=(1 3 4 5)

for EPOCH in "${EPOCHS[@]}"; do
    CHECK_POINTS="${BASE_CHECK_POINTS}${EPOCH}/"
    SAVE="mistral/gsm8k_original_1gpu/best_model/epoch_${EPOCH}/test-output"

    # Create save directory and log file
    mkdir -p $SAVE

    # Run the evaluate script
    python src/evaluate.py \
        --val_path $VAL_PATH \
        --checkpoint_path $CHECK_POINTS \
        --save_path $SAVE \
        --batch_size $BATCH_SIZE \
        --max_new_tokens $MAX_NEW_TOKENS \
        > ${SAVE}/log.test 2>&1 &

    # Optional: Add a small delay between runs if needed
    sleep 5
done