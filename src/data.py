from dataclasses import dataclass
import os
import copy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def clean_text(text):
    return text.strip().replace(',', '')

def split_text(text, split_pattern):
    if split_pattern not in text:
        return None, clean_text(text)
    else:
        parts = text.split(split_pattern, 1)
        return clean_text(parts[0]), clean_text(parts[1])

def extract_answer_w_prefix(text):
    _, ans = split_text(text, '####')
    return f"The answer is {ans}." if '####' in text else ans

def extract_answer(text):
    _, ans = split_text(text, 'The answer is')
    return ans.replace('.', '') if 'The answer is' in text else ans

def extract_cot_w_prefix(text):
    cot, _ = split_text(text, '####')
    return f"Answer: {cot}" if cot else None

class CoTDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, max_length: int, max_size=-1, is_test=False, train_file='data/gsm8k/train_orig', num_demonstrations=5):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        print(f'Creating features from dataset file at {file_path}')

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.train_file = train_file
        self.eos_token = tokenizer.eos_token
        self.num_demonstrations = num_demonstrations

        self.examples = self.load_and_process_file(file_path, max_size)
        self.separator = tokenizer.eos_token_id

        if is_test and train_file:
            self.train_examples = self.load_and_process_file(train_file, max_size)
            self.demonstrations = self.get_random_demonstration()
        else:
            self.train_examples = None


    def load_and_process_file(self, file_path, max_size):
        with open(file_path, encoding="utf-8") as f:
            lines = [line.strip().split('||') for line in f.readlines() if (len(line.strip()) > 0 and not line.strip().isspace()
                                                                             and len(line.strip().split('||')) ==2 )]
        if max_size > 0:
            print (f'truncated to {max_size}')
            lines = lines[:max_size]
        src_lines, tgt_lines = list(zip(*lines))
        src_lines = list(src_lines)
        tgt_lines = list(tgt_lines)
        examples = []
        for src, tgt in zip(src_lines, tgt_lines):
            ans = extract_answer_w_prefix(tgt)
            cot = extract_cot_w_prefix(tgt)
            example = {
                'src': src,
                'cot': cot,
                'ans': ans
            }
            examples.append(example)
        
        return examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        if self.is_test:
            example = self.examples[i]
            input_text = (f"{self.demonstrations} {example['src']} {self.eos_token} "
                          f"{example['cot']} {self.eos_token} {example['ans']} {self.eos_token}")
        else:
            example = self.examples[i]
            input_text = f"{example['src']} {self.eos_token} {example['cot']} {self.eos_token} {example['ans']} {self.eos_token}"

        if self.max_length > 0:
            encoding = self.tokenizer(input_text, add_special_tokens=True, truncation=True, max_length=self.max_length)
        else:
            encoding = self.tokenizer(input_text, add_special_tokens=True)

        input_ids = encoding["input_ids"]
        labels = copy.deepcopy(input_ids)

        if not self.is_test:
            sep_idx = labels.index(self.separator) + 1
            labels[:sep_idx] = [-100] * sep_idx

            if i <3:
                print(f"Input: {self.tokenizer.decode(input_ids)}")
                print(f"Label: {self.tokenizer.decode([t for t in labels if t != -100])}")

        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    
    def get_random_demonstration(self):
        import random
        if not self.train_examples or self.num_demonstrations == 0:
            return ""
        demonstrations = random.sample(self.train_examples, min(self.num_demonstrations, len(self.train_examples)))
        return "".join([f"{example['src']} {self.eos_token} {example['cot']} {self.eos_token} {example['ans']} {self.eos_token}"
                        for example in demonstrations])


@dataclass
class CoTDataCollator:
    """
    VAEData collator used for language modeling.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples):
        input_ids, labels = zip(*examples)
        input_ids = self._tensorize_batch(input_ids)
        input_ids[input_ids.lt(0)] = self.tokenizer.eos_token_id
        labels = self._tensorize_batch(labels)
        return {'input_ids': input_ids, 'labels': labels}

    def _tensorize_batch(self, examples):
        # In order to accept both lists of lists and lists of Tensors
        if isinstance(examples[0], (list, tuple)):
            examples = [torch.tensor(e, dtype=torch.long) for e in examples]
        length_of_first = examples[0].size(0)
        are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
        if are_tensors_same_length:
            return torch.stack(examples, dim=0)
        else:
            return pad_sequence(examples, batch_first=True, padding_value=-100)