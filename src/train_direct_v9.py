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
import json
import random
from transformers import AdamW, get_linear_schedule_with_warmup

from mistral_direct import Mistral 
from configuration import MistralConfig
from utils import get_sep_position, tensorize_batch, split_rationale, compute_entropy_improvement
from data import CoTDataset, CoTDataCollator, extract_answer, extract_answer_w_prefix, extract_cot_w_prefix

from torch.nn.utils.rnn import pad_sequence
# Removed torch.distributed as dist

import pdb

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

@torch.no_grad()
def process_rationales(model, question, rationales, cot_end_token, answer, single_label, current_train_ratio, device, use_in_batch_negative=True, use_hard_negative=True, verbosity_threshold=0.1, epsilon=1, in_batch_negatives=None):
    if not rationales:
        return question, single_label[:, :question.size(1)], []
    
    current_input = question
    current_label = single_label[:, :question.size(1)]  # Initialize with actual question labels
    removed = 0
    i = 0
    removed_indices = []
    rationale_metrics = []
    label_start_idx = question.size(1)

    max_removes_per_rationales = round((current_train_ratio) * len(rationales))
    
    # Decode the original answer (y_g)
    original_answer = model.tokenizer.decode(answer[0], skip_special_tokens=True)
    answer_prefix = "####"
    # answer_value = float(extract_answer(original_answer, prefix=answer_prefix))
    def is_float(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    extracted_value = extract_answer(original_answer, prefix=answer_prefix)
    answer_value = extracted_value if is_float(extracted_value) else extracted_value

    def is_integer(x):
        if float(x).is_integer():
            return int(x)
        else: 
            return x

    # Generate hard negatives (y_w)
    num_in_batch_negatives = len(in_batch_negatives) if in_batch_negatives else 0
    hard_negatives = []
    if use_hard_negative:
        for j in range(num_in_batch_negatives):
            offset = (j + 1) * epsilon
            if j % 2 == 0:
                hard_negatives.append(f"{answer_prefix} {is_integer(answer_value + offset)}")
            else:
                hard_negatives.append(f"{answer_prefix} {is_integer(answer_value - offset)}")
                

    # Encode all wrong answers (in-batch negatives and hard negatives)
    wrong_answers = []
    if use_in_batch_negative and in_batch_negatives:
        wrong_answers.extend(in_batch_negatives)
    if use_hard_negative:
        wrong_answers.extend([model.tokenizer.encode(an, return_tensors="pt").to(device) for an in hard_negatives])

    while i < len(rationales) and removed < max_removes_per_rationales:
        rationale_size = rationales[i].size(1)

        # Full input with all rationales and cot_end_token
        full_input = torch.cat([current_input] + rationales[i:] + [cot_end_token, answer], dim=1)
        # full_label = torch.cat([-100 * torch.ones_like(current_input)] + [-100 * torch.ones_like(r) for r in rationales[i:]] + [cot_end_token] + [answer], dim=1)
        print(answer)
        full_label = torch.cat([-100 * torch.ones_like(current_input)] + [-100 * torch.ones_like(r) for r in rationales[i:]] + [torch.full((1, 2), -100, dtype=torch.long, device=device)] + [torch.cat([answer[0][1:-1],torch.tensor([-100], dtype=torch.long, device=device)]).unsqueeze(0)], dim=1)
        
        # Input without current rationale, but with cot_end_token
        input_without_rationale = torch.cat([current_input] + rationales[i+1:] + [cot_end_token, answer], dim=1)
        # label_without_rationale = torch.cat([-100 * torch.ones_like(current_input)] + [-100 * torch.ones_like(r) for r in rationales[i+1:]] + [cot_end_token] + [answer], dim=1)
        label_without_rationale = torch.cat([-100 * torch.ones_like(current_input)] + [-100 * torch.ones_like(r) for r in rationales[i+1:]] + [torch.full((1, 2), -100, dtype=torch.long, device=device)] + [torch.cat([answer[0][1:-1],torch.tensor([-100], dtype=torch.long, device=device)]).unsqueeze(0)], dim=1)

        # Calculate Verbosity
        outputs_with = model.compute_loss(input_ids=full_input, labels=full_label)
        outputs_without = model.compute_loss(input_ids=input_without_rationale, labels=label_without_rationale)
        verbosity = outputs_with.loss - outputs_without.loss # log(P(y_g | r₂, r₁, x) / P(y_g | r₃, r₂, r₁, x))

        # Calculate Conditional Information Gain and Base Information Gain
        contextual_info_gains = []
        base_info_gains = []
    
        for wrong_ans in wrong_answers:
            # Full input with wrong answer
            full_input_wrong = torch.cat([current_input] + rationales[i:] + [cot_end_token, wrong_ans], dim=1)
            # full_label_wrong = torch.cat([-100 * torch.ones_like(current_input)] + [-100 * torch.ones_like(r) for r in rationales[i:]] + [cot_end_token] + [wrong_ans], dim=1)
            full_label_wrong = torch.cat([-100 * torch.ones_like(current_input)] + [-100 * torch.ones_like(r) for r in rationales[i:]] + [torch.full((1, 2), -100, dtype=torch.long, device=device)] + [torch.cat([wrong_ans[0][1:-1],torch.tensor([-100], dtype=torch.long, device=device)]).unsqueeze(0)], dim=1)
            outputs_with_y_w = model.compute_loss(input_ids=full_input_wrong, labels=full_label_wrong)
            contextual_info_gain = outputs_with.loss - outputs_with_y_w.loss # log(P(y_w | r₃, r₂, r₁, x) / P(y_g | r3, r₂, r₁, x))
            contextual_info_gains.append(contextual_info_gain)

            # Input without current rationale and with wrong answer
            input_without_rationale_wrong = torch.cat([current_input] + rationales[i+1:] + [cot_end_token, wrong_ans], dim=1)
            label_without_rationale_wrong = torch.cat([-100 * torch.ones_like(current_input)] + [-100 * torch.ones_like(r) for r in rationales[i+1:]] + [torch.full((1, 2), -100, dtype=torch.long, device=device)] + [torch.cat([wrong_ans[0][1:-1],torch.tensor([-100], dtype=torch.long, device=device)]).unsqueeze(0)], dim=1)
            outputs_without_y_w = model.compute_loss(input_ids=input_without_rationale_wrong, labels=label_without_rationale_wrong)
            info_gain = outputs_without.loss - outputs_without_y_w.loss  # log(P(y_w | r₃, r₂, x) / P(y_g | r₃, r₂, x))
            base_info_gains.append(info_gain)

            print("======================================")
            print("full input wrong and label")
            print(full_input_wrong)
            print(full_label_wrong)
            print("input without rationale wrong and label")
            print(input_without_rationale_wrong)
            print(label_without_rationale_wrong)
            print("======================================")
            
        # Calculate Informative
        if use_in_batch_negative or use_hard_negative:
            contextual_info_gain = torch.mean(torch.stack(contextual_info_gains)) 
            info_gain = torch.mean(torch.stack(base_info_gains))
            informative = info_gain - contextual_info_gain  # log((P(y_w|r₃,r₂,x)/P(y_g|r₃,r₂,x)) / (P(y_w|r₃,r₂,r₁,x)/P(y_g|r₃,r₂,r₁,x))) 

        if use_in_batch_negative or use_hard_negative:
            is_removed = (verbosity >= verbosity_threshold and informative <= 0).item()
        else:
            contextual_info_gain = torch.zeros_like(torch.empty(1, 1))
            info_gain = torch.zeros_like(torch.empty(1, 1))
            informative = torch.zeros_like(torch.empty(1, 1))
            is_removed = (verbosity >= verbosity_threshold).item()
        
        print(f"Verbosity: {verbosity}")
        print(f"informative: {informative}")
        
        rationale_metrics.append({
            'ith_rationale': i,
            'rationale': model.tokenizer.decode(rationales[i].squeeze(0).tolist(),skip_special_tokens=False),
            'removed': is_removed,
            'gold answer': model.tokenizer.decode(answer[0], skip_special_tokens=False),
            'in-batch-negatives': [model.tokenizer.decode(negative.squeeze(0).tolist(), skip_special_tokens=False) for negative in in_batch_negatives],
            'hard-negatives': hard_negatives,
            'verbosity': verbosity.item(),
            'info_gain': info_gain.item(),
            'contextual_info_gain': contextual_info_gain.item(),
            'informative': informative.item()
        })

        if is_removed:
            removed_indices.append(i)
            removed += 1
        else:
            current_input = torch.cat([current_input, rationales[i]], dim=1)
            current_label = torch.cat([current_label, single_label[:, label_start_idx:label_start_idx+rationale_size]], dim=1)

        i += 1
        label_start_idx += rationale_size

        # Clear unnecessary tensors
        del full_input, full_label, input_without_rationale, label_without_rationale
        torch.cuda.empty_cache()
        
        print(f"REMOVE TEST == RATIONALE SENTENCE : {i} | REMOVED: {removed} | MAXIMUM REMOVAL: {max_removes_per_rationales}")

    # Add remaining rationales if any
    if i < len(rationales):
        remaining_rationales = rationales[i:]
        remaining_input = torch.cat(remaining_rationales, dim=1)
        remaining_label_size = sum(r.size(1) for r in remaining_rationales)
        remaining_label = single_label[:, label_start_idx:label_start_idx+remaining_label_size]
        current_input = torch.cat([current_input, remaining_input], dim=1)
        current_label = torch.cat([current_label, remaining_label], dim=1)
        label_start_idx += remaining_label_size

    # Add cot_end_token and answer to the final input and label
    final_input = torch.cat([current_input, cot_end_token, answer], dim=1)
    final_label = torch.cat([current_label, cot_end_token, answer], dim=1)
    # print("FINAL")
    # print(final_input)
    # print(final_label)

    assert final_input.size(1) == final_label.size(1), f"Mismatch between input and label sizes: input {final_input.size(1)}, label {final_label.size(1)}"

    return final_input, final_label, removed_indices, rationale_metrics

def process_batch_with_entropy(dataset, model, input_ids, labels, tokenizer, current_train_ratio, device, epoch, training_step, batch_indices, use_in_batch_negative=True, use_hard_negative=True, verbosity_threshold=0.1, update=True):
    batch_size = input_ids.size(0)
    new_input_ids = []
    new_labels = []
    batch_data = []

    # Extract all answers in the batch
    all_answers = []
    for i in range(batch_size):
        single_input = input_ids[i].unsqueeze(0)
        cot_end = get_sep_position(single_input, tokenizer.eos_token_id, skip=1).item()
        answer_end = get_sep_position(single_input, tokenizer.eos_token_id, skip=2).item()
        answer = single_input[:, cot_end+1:answer_end+1]
        all_answers.append(answer)

    # Determine if it's a multiple choice task
    is_multiple_choice = is_multiple_choice_task(all_answers, tokenizer)

    for i in range(batch_size):
        single_input = input_ids[i].unsqueeze(0)
        index = batch_indices[i]
        single_label = labels[i].unsqueeze(0)

        question_end = get_sep_position(single_input, tokenizer.eos_token_id, skip=0).item()
        cot_end = get_sep_position(single_input, tokenizer.eos_token_id, skip=1).item()
        answer_end = get_sep_position(single_input, tokenizer.eos_token_id, skip=2).item()

        question = single_input[:, :question_end+1]
        cot = single_input[:, question_end+1:cot_end]  # Exclude cot_end token
        cot_end_token = single_input[:, cot_end:cot_end+1]  # Separate cot_end token
        answer = single_input[:, cot_end+1:answer_end+1]

        # Split CoT into individual rationales
        rationales = split_rationale(cot, tokenizer)

        # Get in-batch negatives based on the task type
        if is_multiple_choice:
            num_negatives = len(all_answers) - 1
            in_batch_negatives = generate_multiple_choice_negatives(answer, tokenizer, num_negatives, device)
        else:
            in_batch_negatives = [a.to(device) for j, a in enumerate(all_answers) if j != i]

        processed_input, processed_label, removed_indices, rationale_metrics = process_rationales(
            model, question, rationales, cot_end_token, answer, single_label, current_train_ratio, device, use_in_batch_negative, use_hard_negative, verbosity_threshold, 1, in_batch_negatives
        )
        
        if update:
            dataset.update_example(
                    batch_indices[i], 
                    processed_input.squeeze(0).cpu(),  
                    processed_label.squeeze(0).cpu()   
                )

        new_input_ids.append(processed_input.squeeze(0).cpu())
        new_labels.append(processed_label.squeeze(0).cpu())

        # Prepare data for recording
        datum_data = {
            'epoch': epoch,
            'training_step': training_step,
            'current_train_ratio': current_train_ratio,
            'original_input_ids': tokenizer.decode(single_input.squeeze(0).tolist(), skip_special_tokens=True),
            'original_labels': tokenizer.decode([t for t in single_label.squeeze(0).tolist() if t != -100]),
            'new_input_ids': tokenizer.decode(processed_input.squeeze(0).tolist(), skip_special_tokens=True),
            'new_labels': tokenizer.decode([t for t in processed_label.squeeze(0).tolist() if t != -100]),
            'removed_indices': removed_indices,
            'rationale_metrics': rationale_metrics
        }

        batch_data.append(datum_data)

        # Clear GPU cache after processing each sample
        torch.cuda.empty_cache()

    # Pad sequences to the same length
    new_input_ids = tensorize_batch(new_input_ids)
    new_input_ids[new_input_ids.lt(0)] = tokenizer.eos_token_id
    new_labels = tensorize_batch(new_labels)

    assert new_input_ids.size() == new_labels.size(), "Mismatch between batch input and label sizes"

    return new_input_ids.to(device), new_labels.to(device), batch_data

def is_multiple_choice_task(all_answers, tokenizer):
    # Decode all answers
    decoded_answers = [tokenizer.decode(answer[0], skip_special_tokens=True).strip() for answer in all_answers]
    
    # Check if all answers start with '####'
    if not all(answer.startswith('####') for answer in decoded_answers):
        return False
    
    # Extract the content after '####'
    answer_contents = [answer[4:].strip() for answer in decoded_answers]

    print("answer_contents")
    print(answer_contents)
    
    # Check if each answer is either a single uppercase letter, single lowercase letter, or True/False
    valid_answers = [
        (len(content) == 1 and content.isupper()) or
        (len(content) == 1 and content.islower()) or
        (content in ['True', 'False'])
        for content in answer_contents
    ]
    
    # If all answers are valid multiple choice answers, return True
    return all(valid_answers)

def generate_multiple_choice_negatives(answer, tokenizer, num_negatives, device):
    import torch
    import random

    current_answer = tokenizer.decode(answer[0], skip_special_tokens=True).strip()
    
    if not current_answer.startswith('####'):
        raise ValueError(f"Unexpected answer format: {current_answer}")
    
    current_choice = current_answer[4:].strip()
    
    if current_choice in ['A', 'B', 'C', 'D', 'E']:
        choices = ['A', 'B', 'C', 'D', 'E']
    elif current_choice in ['a', 'b', 'c', 'd', 'e']:
        choices = ['a', 'b', 'c', 'd', 'e']
    elif current_choice in ['True', 'False']:
        choices = ['True', 'False']
    else:
        raise ValueError(f"Unexpected answer choice: {current_choice}")
    
    # Remove the current answer from the choices
    other_choices = [choice for choice in choices if choice != current_choice]
    
    negatives = []
    for _ in range(num_negatives):
        choice = random.choice(other_choices)
        negative = f"#### {choice}"
        # Convert the negative string to token IDs without adding special tokens
        negative_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(negative))
        negatives.append(torch.tensor([negative_ids], device=device))
    
    return negatives


def evaluate(device, dataloader, model, tokenizer, max_new_tokens=512, skip=0, log_predictions=False, epoch=None, save_path=None):
    model.eval()

    total_instances = 0
    total_correct = 0
    total_loss = 0

    predictions = []

    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
        input_ids_all = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        batch_indices = batch['batch_indices']

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
    parser.add_argument('--use_in_batch_negative', action='store_true')
    parser.add_argument('--use_hard_negative', action='store_true')
    parser.add_argument('--verbosity_threshold', type=float, default=0.1)
    parser.add_argument('--train_orig', action='store_true', default=False)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--accumulate', type=int, default=1)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--max_new_tokens', type=int, default=150)
    parser.add_argument('--update_data', action='store_true', default=False)
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

    config = MistralConfig(base_model=args.base_model)
    
    # Instantiate and move the model to device with correct dtype
    model = Mistral(config).to(device, dtype=ptdtype)
    print_memory_usage("After Model Instantiation")

    tokenizer = model.tokenizer
    collate_fn = CoTDataCollator(tokenizer)
    train_dataset = CoTDataset(tokenizer, args.train_path, 500, max_size=-1, is_test=False, train_file=None, num_demonstrations=-1)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True)
    val_dataset = CoTDataset(tokenizer, args.val_path, 500,  max_size=-1, is_test=False, train_file=None, num_demonstrations=-1)
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
    max_training_time = 36 * 60 * 60  # 24 hours in seconds

    global_step = 1
    best_val = 999
    best_model_path = os.path.join(args.save_model, 'best_model')

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}, model.training: {model.training}")

        # print("Model weight norms at the beginning of epoch:")
        # for name, param in model.named_parameters():
        #     print(f"{name}: {param.data.norm().item()}")

        epoch_loss = 0
        epoch_step = 1
        print(f"Optimizer state at beginning of epoch {epoch}:")
        print(optimizer.state_dict())
        for batch in tqdm.tqdm(train_dataloader):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            batch_indices = batch['batch_indices']

            ### Remove the rationales after the warmup steps
            current_train_ratio = global_step/total_steps
            if current_train_ratio > args.warmup_ratio and not args.train_orig:

                ### Calculate entropy of each rationale and remove the unnecessary sentence
                input_ids, labels, batch_data = process_batch_with_entropy(
                    train_dataset, model, input_ids, labels, tokenizer, current_train_ratio, device, epoch, global_step, batch_indices,
                    use_in_batch_negative=args.use_in_batch_negative, use_hard_negative=args.use_hard_negative,
                    verbosity_threshold=args.verbosity_threshold, update=args.update_data
                )
                    # scheduler = get_linear_schedule_with_warmup(
                    #     optimizer,
                    #     num_warmup_steps=0,
                    #     num_training_steps=total_steps-global_step
                    # )
                
                # Save batch_data to a file
                with open(os.path.join(args.save_data, f'epoch_{epoch}_trained_data.jsonl'), 'a') as f:
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

            if time.time() - start_time > max_training_time:
                print("Reached 24-hour time limit. Stopping training.")
                break

            epoch_loss += loss.item()
            global_step += 1
            epoch_step += 1

        avg_train_loss = epoch_loss / len(train_dataloader)
        optimizer.zero_grad(set_to_none=True)

        print("Model weight norms at the before evaluation:")
        for name, param in model.named_parameters():
            print(f"{name}: {param.data.norm().item()}")

        clear_gpu_cache()
        model.eval()
        with torch.no_grad():
            val_loss, predictions = evaluate(device,
                val_dataloader, model, tokenizer, args.max_new_tokens, skip=0,
                log_predictions=True, epoch=epoch, save_path=args.save_model
            )

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

        del optimizer
        optimizer = get_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    total_time = (time.time() - start_time) / 3600
    print(f"Training completed. Total time: {total_time:.2f} hours")

if __name__ == "__main__":
    main()