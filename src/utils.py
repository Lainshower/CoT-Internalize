import torch
from transformers import StoppingCriteria, LogitsProcessor
from torch.nn.utils.rnn import pad_sequence

'''
BASIC UTILS
'''

def get_sep_position(input_ids, sep_id, skip=0):
    batch_size = input_ids.shape[0]
    sep_positions = input_ids.new_zeros(batch_size).long()
    for batch_id in range(batch_size):
        mask = input_ids[batch_id].eq(sep_id)
        sep_position = mask.nonzero()[0, -1].item()
        for _ in range(skip):
            mask[sep_position] = False
            sep_position = mask.nonzero()[0, -1].item()
        sep_positions[batch_id] = sep_position
    return sep_positions

def tensorize_batch(examples):
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length:
        return torch.stack(examples, dim=0)
    else:
        return pad_sequence(examples, batch_first=True, padding_value=-100)

'''
ENTROPY CALCULATION
'''

def compute_entropy_improvement(entropies, threshold=0.7):
    """
    Compute if entropy shows an increasing trend across layers.
    Returns True if entropy increases for the majority of layer transitions.
    """
    increases = [entropies[i] < entropies[i+1] for i in range(len(entropies)-1)]
    return sum(increases) / len(increases) >= threshold

def split_rationale(rationale, tokenizer):
    """
    Split the rationale into sentences based on end-of-sentence tokens.
    """
    eos_tokens = [tokenizer.convert_tokens_to_ids(t) for t in ['.', '!', '?']]
    sentence_ends = [i for i, token in enumerate(rationale[0]) if token in eos_tokens]
    sentence_starts = [0] + [i + 1 for i in sentence_ends[:-1]]
    return [rationale[:, start:end+1] for start, end in zip(sentence_starts, sentence_ends)]

'''
GENERATION UTILS
'''

# Stop generation only after generating two EOSs, such as  z <eos> y <eos>
class DoubleEOSStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.init = False

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        eos_count = (input_ids == self.eos_token_id).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        return done.all()

class DoubleEOSLogitsProcessor(LogitsProcessor):
    def __init__(self, eos_token_id):
        super().__init__()
        self.eos_token_id = eos_token_id
        self.init = False
    
    def __call__(self, input_ids, scores):
        eos_count = (input_ids == self.eos_token_id).sum(dim=-1)
        if not self.init:
            self.init = True
            self.eos_count_init = eos_count
        done = (eos_count - self.eos_count_init) >= 2
        if done.any():
            scores[done, :] = float('-inf')
            scores[done, self.eos_token_id] = 0
        return scores
