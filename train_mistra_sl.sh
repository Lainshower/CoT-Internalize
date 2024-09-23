#!/bin/sh

#SBATCH -J MISTRAL_REASONING
#SBATCH -o /home/kaoara/CoT-Internalize/mistral_original.log
#SBATCH -t 3-00:00:00

#SBATCH --qos=hpgpu
#SBATCH -p A100-80GB 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=20

echo "Activate conda"
source /home/kaoara/anaconda3/bin/activate cot

# Set the number of GPUs you want to use
NUM_GPUS=3

# Set your parameters
TRAIN_PATH="data/gsm8k/train_orig.txt"
VAL_PATH="data/gsm8k/valid_orig.txt"
export SAVE_M="mistral/gsm8k_original_small_lr"
export SAVE_D="mistral/gsm8k-small_lr_training-data-original"
BASE_MODEL="mistralai/Mistral-7B-v0.1"
EPOCHS=10
BATCH_SIZE=4
WARM_UP=0.2
LEARNING_RATE=2e-06
ACCUMULATE=1
GRAD_NORM=1.0
LOSS_SCALE=0.7

# Create save directories
mkdir -p $SAVE_M
mkdir -p $SAVE_D

# Run the training script
torchrun --nproc_per_node=$NUM_GPUS --master_port=29700 \
    src/train.py \
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
    --accumulate $ACCUMULATE \
    --max_grad_norm $GRAD_NORM \
    > ${SAVE_M}/log.train_orig 2>&1

