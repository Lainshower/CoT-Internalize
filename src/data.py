from dataclasses import dataclass
import os
import copy
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

def clean_text(text):
    return text.strip()

def split_text(text, split_pattern):
    if split_pattern not in text:
        return None, clean_text(text)
    else:
        parts = text.rsplit(split_pattern, 1)
        return clean_text(parts[0]), clean_text(parts[1]).replace(',', '')

def extract_answer_w_prefix(text, prefix='####'):
    _, ans = split_text(text, '####')
    return prefix + " " + ans if ans is not None else text

def extract_answer(text, prefix):
    _, ans = split_text(text, prefix)
    return ans if ans is not None else text

def extract_cot_w_prefix(text, prefix=""):
    cot, _ = split_text(text, '####')
    return prefix + cot if cot is not None else prefix + text

class CoTDataset(Dataset):
    def __init__(self, tokenizer, file_path: str, max_length: int, max_size=-1, is_test=False, train_file='data/gsm8k/train_orig', num_demonstrations=-1):
        assert os.path.isfile(file_path), f"Input file path {file_path} not found"
        print(f'Creating features from dataset file at {file_path}')

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test
        self.train_file = train_file
        self.eos_token = tokenizer.eos_token
        self.num_demonstrations = num_demonstrations
        self.separator = tokenizer.eos_token_id

        self.examples = self.load_and_process_file(file_path, max_size)

        if is_test and train_file and num_demonstrations>0:
            self.train_examples = self.load_and_process_file(train_file, max_size)
            self.demonstrations = self.get_random_demonstration()
            self.examples = self.add_demonstrations_to_examples()
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
            input_text = f"{src}{self.eos_token}{cot}{self.eos_token}{ans}{self.eos_token}"
            
            encoding = self.tokenizer(input_text, add_special_tokens=True, truncation=False)
            if len(encoding['input_ids']) <= self.max_length:
                input_ids = encoding['input_ids']
                labels = copy.deepcopy(input_ids)
                sep_idx = labels.index(self.separator) + 1
                labels[:sep_idx] = [-100] * sep_idx
                example = {
                    'input_ids': input_ids,
                    'labels': labels
                }
                examples.append(example)
        
        return examples
    
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
            example = self.examples[i]
            return {
                'input_ids': torch.tensor(example['input_ids'], dtype=torch.long),
                'labels': torch.tensor(example['labels'], dtype=torch.long),
                'index': i
            }
    
    def update_example(self, index, new_input_ids, new_labels):
        """
        Update the input_ids and labels for a specific example.
        """
        self.examples[index]['input_ids'] = new_input_ids
        self.examples[index]['labels'] = new_labels
    
    def get_random_demonstration(self):
        import random
        if not self.train_examples or self.num_demonstrations == 0:
            return ""
        demonstrations = random.sample(self.train_examples, min(self.num_demonstrations, len(self.train_examples)))
        return "".join([f"{self.tokenizer.decode(example['input_ids'])}{self.eos_token}" for example in demonstrations])

    def add_demonstrations_to_examples(self):
        demo_input_ids = self.tokenizer(self.demonstrations, add_special_tokens=True)['input_ids']
        new_examples = []
        for example in self.examples:
            new_input_ids = demo_input_ids + example['input_ids']
            new_labels = [-100] * len(demo_input_ids) + example['labels']
            if len(new_input_ids) <= self.max_length:
                new_example = {
                    'input_ids': new_input_ids[:self.max_length],
                    'labels': new_labels[:self.max_length]
                }
                new_examples.append(new_example)
        return new_examples


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
        input_ids = [example['input_ids'] for example in examples]
        labels = [example['labels'] for example in examples]
        batch_indices = [example['index'] for example in examples]
        
        input_ids = self._tensorize_batch(input_ids)
        input_ids[input_ids.lt(0)] = self.tokenizer.eos_token_id
        labels = self._tensorize_batch(labels)

        return {'input_ids': input_ids, 'labels': labels, 'batch_indices': batch_indices}

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