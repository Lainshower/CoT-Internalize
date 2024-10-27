import argparse
import torch
from torch.utils.data import DataLoader
import tqdm
import json
import os

from mistral import Mistral
from configuration import MistralConfig
from data import CoTDataset, CoTDataCollator, extract_answer
from utils import get_sep_position

def evaluate(device, dataloader, model, tokenizer, max_new_tokens=512, skip=0, log_predictions=False, epoch=None, save_path=None):
    model.eval()

    total_instances = 0
    total_correct = 0
    total_loss = 0
    total_generated_len = 0 

    predictions = []

    for batch in tqdm.tqdm(dataloader, desc="Evaluating"):
        input_ids_all = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        # Remove answer part
        sep_positions = get_sep_position(input_ids_all, tokenizer.eos_token_id)
        for i, (input_ids_all_i, label_i) in enumerate(zip(input_ids_all, labels)):
            output = model.compute_loss(input_ids=input_ids_all_i.unsqueeze(0), labels=label_i.unsqueeze(0))
            total_loss += output.loss.item()

            if log_predictions:
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

                # Label 
                tgt = input_ids_all_i[sep_position+1:]  # Slicing Out CoT;Answer
                tgt_text = tokenizer.decode(tgt, skip_special_tokens=True)
                ans = extract_answer(tgt_text, prefix='####')

                # Predictions
                generated_tokens = generated_output[0][0][sep_position+1:]  # Slicing Out CoT;Answer

                pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                total_generated_len += len(tokenizer.tokenize(pred_text))
                pred_ans = extract_answer(pred_text, prefix='####')

                if ans == pred_ans:
                    total_correct += 1

                predictions.append({
                        'PPL for Full Instance': output.loss.exp().item(),
                        'Input': tokenizer.decode(truncated_input, skip_special_tokens=True),
                        'Target': tgt_text,
                        'Predicted': pred_text,
                        'Target Ans': ans,
                        'Predicted Ans': pred_ans,
                        'Generated Tokens Length': len(tokenizer.tokenize(pred_text)),
                    })

            total_instances += 1
    
    with open(os.path.join(save_path, f'checkpoint_predictions.jsonl'), 'w') as f:
        for pred in predictions:
            json.dump(pred, f)
            f.write('\n')

    loss = total_loss / total_instances
    accuracy = total_correct / total_instances
    avg_generated_len = total_generated_len / total_instances 

    return loss, accuracy, avg_generated_len, predictions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val_path', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model from checkpoint
    model = Mistral.from_pretrained(args.checkpoint_path)
    model = model.to(device)
    model.eval()

    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    collate_fn = CoTDataCollator(tokenizer)

    # Load validation dataset
    val_dataset = CoTDataset(tokenizer, args.val_path, 800, max_size=-1, is_test=True, train_file=None, num_demonstrations=-1)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False)

    # Evaluate
    with torch.no_grad():
        val_loss, accuracy, avg_generated_len, redictions = evaluate(
            device,
            val_dataloader,
            model,
            tokenizer,
            args.max_new_tokens,
            skip=0,
            log_predictions=True,
            save_path=args.save_path
        )

    print(f"Evaluation Loss: {val_loss}")
    print(f"Perplexity: {torch.exp(torch.tensor(val_loss)).item()}")
    print(f"Average Generated Tokens: {avg_generated_len}")
    print(f"Evaluation Accuacy: {accuracy}")

if __name__ == "__main__":
    main()