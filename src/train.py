import inspect
import torch
# Removed FSDP and distributed imports
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

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
# Removed torch.distributed as dist

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
logging.disable(logging.WARNING)

def print_memory_usage(label):
    allocated = torch.cuda.memory_allocated() / 1e9
    cached = torch.cuda.memory_reserved() / 1e9
    print(f"{label} - Allocated: {allocated:.2f} GB, Cached: {cached:.2f} GB")

def print_model_memory(model, name):
    total_params = sum(p.numel() for p in model.parameters())
    total_mem = total_params * 4 / (1024 ** 3)  # Assuming float32, convert to GB
    print(f"{name} parameters memory: {total_mem:.2f} GB")

def clear_gpu_cache():
    """Clear the GPU cache"""
    print(f"Clearing GPU cache")
    torch.cuda.empty_cache()

def save_model(model, tokenizer, model_dir, current_epoch):
    print('Saving model to', model_dir)
    outpath = os.path.join(model_dir, f"epoch_{current_epoch}")
    os.makedirs(outpath, exist_ok=True)

    print(f"SAVING MODEL")
    model.save_pretrained(outpath)
    tokenizer.save_pretrained(outpath)

def check_entropy(model, input_sequence, label_sequence, return_tokens, hidden_improve, hidden_interval, improved_ratio):
    outputs = model.compute_entropy(input_sequence, label_sequence, interval=hidden_interval, return_seq=return_tokens)
    datum_entropies = outputs.cross_entropies[0]
    return datum_entropies, compute_entropy_improvement(datum_entropies, threshold=hidden_improve, improved_ratio=improved_ratio, return_seq=return_tokens)

@torch.no_grad()
def process_rationales(model, question, rationales, cot_end_idx, single_label, current_train_ratio, return_tokens, hidden_improve, hidden_interval, improved_ratio, device):
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
    rationale_entropies = []
    label_start_idx = question.size(1)

    max_removes_per_rationales = round((current_train_ratio) * len(rationales))
    while i < len(rationales) and removed < max_removes_per_rationales:
        print(f"REMOVE TEST ==  RATIONALE SENTENCE : {i} | REMOVED: {removed} | MAXIMUM REMOVAL: {max_removes_per_rationales}")
        rationale_size = rationales[i].size(1)

        # Append rationale sentence to the input
        test_input = torch.cat([current_input, rationales[i]], dim=1)
        
        # Append corresponding rationale index to the test
        test_label = torch.cat([current_label, single_label[:, label_start_idx:label_start_idx + rationale_size]], dim=1)

        i_rationale_entropies, improve = check_entropy(model, test_input, test_label, return_tokens, hidden_improve, hidden_interval, improved_ratio)
        rationale_entropies.append(i_rationale_entropies)
        if improve:
            current_input = test_input
            current_label = test_label
        else:
            removed_indices.append(i)
            removed += 1

        # Clear unnecessary tensors from GPU memory
        del test_input, test_label
        torch.cuda.empty_cache()

        i += 1
        label_start_idx += rationale_size

    # Add remaining rationales if any
    if i < len(rationales):
        remaining_input = torch.cat(rationales[i:], dim=1)
        remaining_label = single_label[:, label_start_idx:cot_end_idx]
        current_input = torch.cat([current_input, remaining_input], dim=1)
        current_label = torch.cat([current_label, remaining_label], dim=1)

    return current_input, current_label, removed_indices, rationale_entropies

def process_batch_with_entropy(model, input_ids, labels, tokenizer, current_train_ratio, return_tokens, hidden_improve, improved_ratio, hidden_interval, device, epoch, training_step):
    batch_size = input_ids.size(0)
    new_input_ids = []
    new_labels = []
    batch_data = []

    for i in range(batch_size):
        single_input = input_ids[i].unsqueeze(0).to(device)
        single_label = labels[i].unsqueeze(0).to(device)

        question_end = get_sep_position(single_input, tokenizer.eos_token_id, skip=0).item()
        cot_end = get_sep_position(single_input, tokenizer.eos_token_id, skip=1).item()
        answer_end = get_sep_position(single_input, tokenizer.eos_token_id, skip=2).item()

        question = single_input[:, :question_end+1]
        cot = single_input[:, question_end+1:cot_end+1] 
        answer = single_input[:, cot_end+1:answer_end+1]

        # Split CoT into individual rationales
        rationales = split_rationale(cot, tokenizer)

        # Process rationales
        processed_input, processed_label, removed_indices, rationale_entropies = process_rationales(
            model, question, rationales, cot_end+1, single_label, current_train_ratio, return_tokens,
            hidden_improve, hidden_interval, improved_ratio, device
        )
        
        # Add answer to the final input
        new_input = torch.cat([processed_input, answer], dim=1)

        # Create labels for the new input
        new_label = torch.cat([processed_label, single_label[:, cot_end+1:answer_end+1]], dim=1)

        assert new_input.shape == new_label.shape, f"Shape mismatch: input {new_input.shape}, label {new_label.shape}"

        new_input_ids.append(new_input.squeeze(0).cpu())
        new_labels.append(new_label.squeeze(0).cpu())

        # Prepare data for recording
        datum_data = {
            'epoch': epoch,
            'training_step': training_step,
            'current_train_ratio': current_train_ratio,
            'original_input_ids': tokenizer.decode(single_input.squeeze(0).tolist(), skip_special_tokens=True),
            'original_labels': tokenizer.decode([t for t in single_label.squeeze(0).tolist() if t != -100], skip_special_tokens=True),
            'new_input_ids': tokenizer.decode(new_input.squeeze(0).tolist(), skip_special_tokens=True),
            'new_labels': tokenizer.decode([t for t in new_label.squeeze(0).tolist() if t != -100], skip_special_tokens=True),
        }

        for j, entropy in enumerate(rationale_entropies):  
            datum_data[f"{j}'s rationale entropies"] = entropy.tolist()
            datum_data[f"{j}'s rationale removed"] = j in removed_indices

        print(datum_data)
        batch_data.append(datum_data)

        # Clear GPU cache after processing each sample
        torch.cuda.empty_cache()

    # Pad sequences to the same length
    new_input_ids = tensorize_batch(new_input_ids)
    new_input_ids[new_input_ids.lt(0)] = tokenizer.eos_token_id
    new_labels = tensorize_batch(new_labels)

    return new_input_ids.to(device), new_labels.to(device), batch_data


def evaluate(device, dataloader, model, tokenizer, max_new_tokens=512, skip=0, log_predictions=False, epoch=None, save_path=None):
    model.eval()

    total_instances = 0
    total_correct = 0
    total_loss = 0

    predictions = []

    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
        input_ids_all = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        for i, (input_ids_all_i, label_i) in enumerate(zip(input_ids_all, labels)):
            output = model.compute_loss(input_ids=input_ids_all_i.unsqueeze(0), labels=label_i.unsqueeze(0))
            total_loss += output.loss.item()

            if log_predictions and total_instances < 3:
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
        
                tgt = input_ids_all_i[sep_position+1:]  # Slicing Out CoT;Answer
                tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
                ans = extract_answer(tgt_text, prefix='####')
                pred_text = tokenizer.decode(generated_output[0][0][sep_position+1:], skip_special_tokens=True)
                pred_ans = extract_answer(pred_text, prefix='####')

                if ans == pred_ans:
                    total_correct += 0

                predictions.append({
                        'PPL for Full Instance': output.loss.exp().item(),
                        'Input': tokenizer.decode(truncated_input, skip_special_tokens=True),
                        'Target': tgt_text,
                        'Predicted': pred_text,
                    })

            total_instances += 1
    
    with open(os.path.join(save_path, f'epoch_{epoch}_predictions.jsonl'), 'w') as f:
        for pred in predictions:
            json.dump(pred, f)
            f.write('\n')

    loss = total_loss / total_instances

    return loss, predictions

def get_parameter_names(model, forbidden_layer_types):
    result = []
    base_model = model.base_model  
    for name, param in base_model.named_parameters(): 
        if not any(isinstance(param, layer) for layer in forbidden_layer_types):
            result.append(name)
    return result

def get_optimizer(model, lr, weight_decay):
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.base_model.named_parameters()
                if n in decay_parameters and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.base_model.named_parameters()
                if n not in decay_parameters and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    use_fused = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    extra_args = dict(fused=True) if use_fused else dict()

    return torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=lr,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=weight_decay,
        **extra_args
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, required=True)
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--save_model', type=str, required=True)
    parser.add_argument('--save_data', type=str, required=True)
    parser.add_argument('--base_model', type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--warmup_ratio', type=float, default=0.01)
    parser.add_argument('--return_tokens', action='store_true')
    parser.add_argument('--hidden_improve', type=float, default=0.7)
    parser.add_argument('--improved_ratio', type=float, default=0.7)
    parser.add_argument('--hidden_interval', type=int, default=4)
    parser.add_argument('--train_orig', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=150)
    args = parser.parse_args()
    print(args)

    random.seed(486486)
    torch.manual_seed(486486)

    dtype = 'float32'
    if args.bf16:
        dtype = 'bfloat16'
    elif args.fp16:
        dtype = 'float16'
    else:
        dtype = 'float32'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    local_rank = 0  # Set local_rank to 0 for single GPU
    clear_gpu_cache()

    # Initialize wandb
    wandb.init(
        project="Mistral-GSM8K-ENTROPY",
        name=f"learning_config_ep{args.epochs}_bsz{args.batch_size}_lr{args.lr}",
        config=args
    )

    config = MistralConfig(base_model=args.base_model)
    
    # Instantiate and move the model to device with correct dtype
    model = Mistral(config).to(device, dtype=ptdtype)
    print_memory_usage("After Model Instantiation")

    tokenizer = model.tokenizer
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 800, max_size=-1, is_test=False, train_file=None, num_demonstrations=-1)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, 800,  max_size=-1, is_test=False, train_file=None, num_demonstrations=-1)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    total_steps = len(train_dataloader) * args.epochs
    warmup_steps = min(int(0.1 * total_steps), 100)

    trainable_params = list(model.base_model.parameters())
    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=warmup_steps,
    #     num_training_steps=total_steps
    # )

    # Initialize GradScaler only if bf16 is False
    if not args.bf16 and args.fp16:
        scaler = GradScaler()
        use_scaler = True
    else:
        scaler = None
        use_scaler = False

    model.train()
    start_time = time.time()
    max_training_time = 24 * 60 * 60  # 24 hours in seconds

    prev_batch_seq = None
    global_step = 1
    best_val = 999
    best_model_path = os.path.join(args.save_model, 'best_model')

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}, model.training: {model.training}")

        print("Model weight norms at the beginning of epoch:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.data.norm().item()}")

        epoch_loss = 0
        epoch_step = 1
        print(f"Optimizer state at beginning of epoch {epoch}:")
        print(optimizer.state_dict())
        for batch in tqdm.tqdm(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            if prev_batch_seq == None:
                prev_batch_seq = input_ids.shape[-1]

            ### Cache the pre-ratioanle removed data shape
            bef_process_shape = input_ids.shape

            ### Remove the rationales after the warmup steps
            current_train_ratio = global_step/total_steps
            if current_train_ratio > args.warmup_ratio and not args.train_orig:

                ### Calculate entropy of each rationale and remove the unnecessary sentence
                input_ids, labels, batch_data = process_batch_with_entropy(model, input_ids, labels, tokenizer, current_train_ratio, args.return_tokens, args.hidden_improve, args.improved_ratio, args.hidden_interval, device, epoch, epoch_step)
                if bef_process_shape != input_ids.shape: # and prev_batch_seq != input_ids.shape[-1]
                    print(f"BEFORE SHAPE: {bef_process_shape} | AFTER SHAPE: {input_ids.shape}")
                    optimizer.zero_grad(set_to_none=True)
                    # del optimizer, scheduler
                    del optimizer
                    optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
                    # scheduler = get_linear_schedule_with_warmup(
                    #     optimizer,
                    #     num_warmup_steps=0,
                    #     num_training_steps=total_steps-global_step
                    # )
                
                # Save batch_data to a file
                with open(os.path.join(args.save_data, f'epoch_{epoch}_entropy_data.jsonl'), 'a') as f:
                    for data in batch_data:
                        json.dump(data, f)
                        f.write('\n')

            # Before forward pass memory
            print_memory_usage("Before Forward Pass")

            # Forward pass with autocast if bf16 is enabled
            with autocast(dtype=ptdtype) if args.bf16 or args.fp16 else torch.cuda.amp.autocast(enabled=False):
                outputs = model.compute_loss(input_ids=input_ids, labels=labels)
                loss = outputs.loss
                print("Loss : ", loss)
            print_memory_usage("After Forward Pass")

            # Backward pass
            if use_scaler:
                scaler.scale(loss).div(args.accumulate).backward()
            else:
                loss.div(args.accumulate).backward()

            print_memory_usage("After Backward Pass")

            '''
            Check Model's Internal Parameter during the training Step
            if global_step % args.accumulate == 0:
                for name, param in model.named_parameters():
                    if torch.isnan(param.grad).any():
                        print(f"Layer: {name} | Gradient contains NaN")
                        print(f"Gradient Norm: {param.grad.norm()}")
                        print(f"Gradient: {param.grad}")
                        exit()
                    else:
                        current_lr = optimizer.param_groups[0]['lr']
                        weight_norm = param.data.norm().item()
                        print(f"Layer: {name} | Gradient Norm: {param.grad.norm()} | Gradient Type: {param.grad.dtype} | Weight Norm: {weight_norm} | current_lr: {current_lr}")
            '''

            if global_step % args.accumulate == 0 or global_step % len(train_dataloader) == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.base_model.parameters(), args.max_grad_norm)
                if use_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                # scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            wandb.log({"train_step_loss": loss.item(), "train_step_ppl": loss.exp().item(), "step": global_step})

            if time.time() - start_time > max_training_time:
                print("Reached 24-hour time limit. Stopping training.")
                break

            epoch_loss += loss.item()
            global_step += 1
            epoch_step += 1
            prev_batch_seq = input_ids.shape[-1]

        avg_train_loss = epoch_loss / len(train_dataloader)
        optimizer.zero_grad(set_to_none=True)

        wandb.log({"train_epoch_loss": avg_train_loss, "epoch": epoch})

        print("Model weight norms at the before evaluation:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.data.norm().item()}")

        model.eval()
        with torch.no_grad():
            val_loss, predictions = evaluate(device,
                val_dataloader, model, tokenizer, args.max_new_tokens, skip=0,
                log_predictions=True, epoch=epoch, save_path=args.save_model
            )
            wandb.log({
                "val_total_loss": val_loss,
                "epoch": epoch
            })

            # Save the best model based on loss
            if val_loss < best_val:
                best_val = val_loss
                print(f"New best model saved with loss: {best_val}")
            save_model(model, tokenizer, best_model_path, epoch)

        model.train()

        print("Model weight norms at the after evaluation / end of epoch:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.data.norm().item()}")

        if time.time() - start_time > max_training_time:
            break

    total_time = (time.time() - start_time) / 3600
    print(f"Training completed. Total time: {total_time:.2f} hours")
    wandb.log({"total_training_time": total_time})
    wandb.finish()

if __name__ == "__main__":
    main()