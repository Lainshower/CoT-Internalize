import inspect
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, FullStateDictConfig, StateDictType
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast

import argparse
import os
import tqdm
import logging
import time
import math
import wandb
import json
import random
from transformers import AdamW, get_linear_schedule_with_warmup

from mistral import Mistral 
from configuration import MistralConfig
from utils import get_sep_position, tensorize_batch, split_rationale, compute_entropy_improvement
from data import CoTDataset, CoTDataCollator, extract_answer, extract_answer_w_prefix, extract_cot_w_prefix

from torch.nn.utils.rnn import pad_sequence
import torch.distributed as dist

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING)

def print_memory_usage(step):
    print(f"Step: {step}")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def print_model_memory(model, name):
    total_params = sum(p.numel() for p in model.parameters())
    total_mem = total_params * 4 / (1024 ** 3)  # Assuming float32, convert to GB
    print(f"{name} parameters memory: {total_mem:.2f} GB")
 
def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["OMP_NUM_THREADS"] = "32"
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    if rank == 0:
        print(f"--> Running with torch dist debug set to detail")

# Initialize the process group
def setup_distributed(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        print(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()

def save_model(local_rank, model, tokenizer, model_dir, current_epoch):
    print('Saving model to', model_dir)
    outpath = os.path.join(model_dir, f"epoch_{current_epoch}")
    os.makedirs(outpath, exist_ok=True)

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_policy):
        state_dict = model.module.state_dict()

    if local_rank == 0:
        print(f"SAVING MODEL")
        model.save_pretrained(outpath, state_dict=state_dict)
        tokenizer.save_pretrained(outpath)

def check_entropy(model, input_sequence, label_sequence):
    entropies = model.compute_entropy(input_sequence, label_sequence, interval=1).cross_entropies[0]
    return compute_entropy_improvement(entropies)

def process_rationales(model, question, rationales, single_label, max_removes_ratio):
    """
    Process rationales based on entropy improvement, limiting discards based on training progress.
    Ensures consistency between input and label removals.
    """
    if not rationales:
        return question, single_label[:, :question.size(1)], []
    
    current_input = question
    current_label = single_label[:, :question.size(1)]
    removed = 0
    i = 0
    removed_indices = []
    label_start_idx = question.size(1)

    max_removes = round(max_removes_ratio * len(rationales))
    while i < len(rationales) and removed < max_removes:
        rationale_size = rationales[i].size(1)

        # Prepend rationale sentence to the input
        test_input = torch.cat([current_input, rationales[i]], dim=1)
        
        # Prepend corresponding rationale index to the test
        test_label = torch.cat([current_label, single_label[:, label_start_idx:label_start_idx + rationale_size]], dim=1)

        if check_entropy(model, test_input, test_label):
            current_input = test_input
            current_label = test_label
        else:
            removed_indices.append(i)
            discarded += 1

        i += 1
        label_start_idx += rationale_size

    # Add remaining rationales if any
    if i < len(rationales):
        remaining_input = torch.cat(rationales[i:], dim=1)
        remaining_label = single_label[:, label_start_idx:]
        current_input = torch.cat([current_input, remaining_input], dim=1)
        current_label = torch.cat([current_label, remaining_label], dim=1)

    return current_input, current_label, removed_indices

def process_batch_with_entropy(model, input_ids, labels, tokenizer, max_removes_ratio):
    """
    Process the batch with step-by-step entropy-based CoT removal.
    """
    batch_size = input_ids.size(0)
    rationale_prefix_len = len(tokenizer.tokenize("Answer:"))
    new_input_ids = []
    new_labels = []

    for i in range(batch_size):
        single_input = input_ids[i].unsqueeze(0)
        single_label = labels[i].unsqueeze(0)

        question_end = get_sep_position(single_input, tokenizer.eos_token_id, skip=0).item()
        cot_end = get_sep_position(single_input, tokenizer.eos_token_id, skip=1).item()
        answer_end = get_sep_position(single_input, tokenizer.eos_token_id, skip=2).item()

        # Incorporate 'Answer:' to question
        question = single_input[:, :question_end+1+rationale_prefix_len]
        cot = single_input[:, question_end+1+rationale_prefix_len:cot_end+1] 
        answer = single_input[:, cot_end+1:answer_end+1]

        # Split CoT into individual rationales
        rationales = split_rationale(cot, tokenizer)

        # Process rationales
        processed_input, processed_label, removed_indices = process_rationales(model, question, rationales, single_label, max_removes_ratio)

        # Add answer to the final input
        new_input = torch.cat([processed_input, answer], dim=1)

        # Create labels for the new input
        new_label = torch.cat([processed_label, single_label[:, cot_end+1:answer_end+1]], dim=1)

        assert new_input.shape == new_label.shape, f"Shape of the new_input is {new_input.shape} but the shape of the new label is {new_label.shape}"

        new_input_ids.append(new_input.squeeze(0))
        new_labels.append(new_label.squeeze(0))

    # Pad sequences to the same length
    new_input_ids = tensorize_batch(new_input_ids)
    new_labels = tensorize_batch(new_labels)

    return new_input_ids, new_labels


def evaluate(dataloader, model, tokenizer,  max_new_tokens=512, skip=0, log_predictions=False, epoch=None, save_path=None):
    model.eval()

    total_instances = 0
    total_tokens = 0
    total_correct = 0
    total_correct_tokens = 0
    total_loss = 0

    predictions = []

    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
        input_ids_all = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()

        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        for i, (input_ids_all_i, label_i) in enumerate(zip(input_ids_all, labels)):
            output = model.compute_loss(input_ids=input_ids_all_i.unsqueeze(0), labels=label_i.unsqueeze(0))
            total_loss += output.loss

            # Truncate input_ids_all_i for generation => Generation
            sep_position = sep_positions[i].item()
            truncated_input = input_ids_all_i[:sep_position + 1]  # +1 to include the separator token
            
            # Generate using the truncated input
            stop_on_two_eos = True
            generated_output = model.generate(
                input_ids=truncated_input.unsqueeze(0),
                max_new_tokens=max_new_tokens,
                stop_on_two_eos=stop_on_two_eos,
            )
    
            tgt = input_ids_all_i[sep_position+1:] # Slicing Out CoT;Answer
            tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
            ans = extract_answer(tgt_text)
            # print(tokenizer.decode(generated_output[0][0], skip_special_tokens=True))
            # print(tokenizer.decode(generated_output[0][0][sep_position+1:], skip_special_tokens=True))
            pred_text = tokenizer.decode(generated_output[0][0][sep_position+1:], skip_special_tokens=True)
            pred_ans = extract_answer(pred_text)

            if log_predictions:
                predictions.append({
                    'PPL for Full Instance': output.loss.exp().item(),
                    'Input': tokenizer.decode(truncated_input, skip_special_tokens=True),
                    'Target': tgt_text,
                    'Predicted': pred_text,
                })

            if ans == pred_ans:
                total_correct += 1

            total_correct_tokens += 1
            total_instances += 1

    accuracy = total_correct / total_instances
    loss = total_loss / total_instances

    # Save predictions for each epoch
    if epoch is not None and save_path is not None:
        with open(os.path.join(save_path, f'epoch_{epoch}_predictions.jsonl'), 'w') as f:
            for pred in predictions:
                json.dump(pred, f)
                f.write('\n')

    return loss, accuracy, predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--save_data', type=str, required=True)
    parser.add_argument('--base_model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=150)
    args = parser.parse_args()
    print(args)

    random.seed(486)
    torch.manual_seed(486)

    dtype = 'float32'
    if args.bf16:
        dtype = 'bfloat16'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    if -1 in [local_rank, world_size]:
        raise ValueError("Environment variables not set correctly")

    setup_environ_flags(local_rank)
    setup_distributed(rank=local_rank, world_size=world_size)

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    clear_gpu_cache(local_rank)

    if local_rank == 0:
        wandb.init(
            project="Mistral-GSM8K",
            name=f"learning_config_ep{args.epochs}_bsz{args.batch_size}_lr{args.lr}",
            config=args
        )

    config = MistralConfig(base_model=args.base_model)
    model = Mistral(config).to(local_rank)

    fsdp_config = dict(
        auto_wrap_policy=size_based_auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        mixed_precision=MixedPrecision(
            param_dtype=ptdtype,
            reduce_dtype=ptdtype,
            buffer_dtype=ptdtype,
        ),
    )

    model = FSDP(model, **fsdp_config)

    tokenizer = model.tokenizer
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 512, max_size=-1, is_test=False, train_file=None, num_demonstrations=-1)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=train_sampler)
    val_dataset = CoTDataset(tokenizer, args.val_path, 512,  max_size=-1, is_test=False, train_file=None, num_demonstrations=-1)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, sampler=val_sampler)

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = min(int(0.1 * total_steps), 100)

    trainable_params = list(model.parameters())
    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, **extra_args)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)   

    model.train()
    start_time = time.time()
    max_training_time = 24 * 60 * 60  # 24 hours in seconds

    global_step = 0
    best_val = 999
    best_accuracy = 0
    best_model_path = os.path.join(args.save_model, 'best_model')

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        epoch_loss = 0
        epoch_step = 1
        train_sampler.set_epoch(epoch)
        for batch in tqdm.tqdm(train_dataloader):
            input_ids = batch['input_ids'].to(local_rank)
            labels = batch['labels'].to(local_rank)

            #### Remove the rationales after the warmup steps
            if epoch_step/len(train_dataloader) > args.warmup_ratio:
                #### Calculate entropy of the rationale and remove the unnecessary part
                input_ids, labels = process_batch_with_entropy(model, input_ids, labels, tokenizer, epoch_step/len(train_dataloader))

            optimizer.zero_grad()
            outputs = model.compute_loss(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            loss.div(args.accumulate).backward()
            if global_step % args.accumulate == 0 or global_step % len(train_dataloader)==0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if local_rank == 0 and epoch_step%100:
                training_step_data = []
                for input_, label_ in zip(input_ids, labels):
                    predictions.append({
                        'Input': tokenizer.decode(input_, skip_special_tokens=True),
                        'Target': tokenizer.decode(label_, skip_special_tokens=True),
                    })
                with open(os.path.join(args.save_data, f'epoch_{epoch}_trainig-step_{epoch_step}_data.jsonl'), 'w') as f:
                    for datum in training_step_data:
                        json.dump(datum, f)
                        f.write('\n')

            if local_rank == 0:
                wandb.log({"train_step_loss": loss.item(), "train_step_ppl": loss.exp().item(),  "step": global_step})

            if time.time() - start_time > max_training_time:
                print("Reached 24-hour time limit. Stopping training.")
                break

            epoch_loss += loss.item()
            epoch_step += 1
            global_step += 1

        avg_train_loss = epoch_loss / len(train_dataloader)

        if local_rank == 0:
            wandb.log({"train_epoch_loss": avg_train_loss, "epoch": epoch})

        model.eval()
        with torch.no_grad():
            val_loss, accuracy, predictions = evaluate(val_dataloader, model, tokenizer, args.max_new_tokens, skip=0, log_predictions=True, epoch=epoch, save_path=args.save_model)
            print(f'Accuracy: {accuracy}')
            if local_rank == 0:
                wandb.log({
                    "val_total_loss": val_loss,
                    "val_accuracy": accuracy,
                    "epoch": epoch
                })

            # Save the best model based on both loss and accuracy
            if val_loss < best_val or accuracy > best_accuracy:
                best_val = min(best_val, val_loss)
                best_accuracy = max(best_accuracy, accuracy)
                save_model(local_rank, model, tokenizer, best_model_path, epoch)
                print(f"New best model saved with loss: {best_val} and accuracy: {best_accuracy}")

        model.train()

        if time.time() - start_time > max_training_time:
            break

    if local_rank == 0:
        total_time = (time.time() - start_time) / 3600
        print(f"Training completed. Total time: {total_time:.2f} hours")
        wandb.log({"total_training_time": total_time})
        print(f"Best model saved with accuracy: {best_accuracy}")
        wandb.finish()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()