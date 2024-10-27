import re
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

# def compute_entropy_improvement(entropies, threshold=0.6, improved_ratio=0.5):
#     """
#     Compute if entropy shows an increasing trend across layers.
#     Returns True if an entropy is greater than k_percent of all previous entropies
#     for the majority of layer transitions.
    
#     :param entropies: List of entropy values for each layer
#     :param threshold: Minimum ratio of improvements required 
#     :param improved_ratio: Percentage threshold for comparison
#     :return: Boolean indicating if entropy shows an increasing trend
#     """
#     improvements = []
    
#     for i in range(1, len(entropies)):
#         current_entropy = entropies[i]
#         previous_entropies = entropies[:i]
        
#         better_count = sum(1 for prev in previous_entropies if current_entropy < prev)
        
#         is_improved = (better_count / len(previous_entropies)) >= improved_ratio
#         improvements.append(is_improved)
    
#     improvement_ratio = sum(improvements) / len(improvements)
    
#     return improvement_ratio >= threshold

def compute_entropy_improvement(entropies, threshold=0.6, improved_ratio=0.5, return_seq=False):
    """
    Compute if entropy shows an increasing trend across layers.
    
    :param entropies: List of entropy values for each layer (shape depends on return_seq)
    :param threshold: Minimum ratio of improvements required 
    :param improved_ratio: Percentage threshold for comparison
    :param return_seq: Boolean indicating if entropies are for each token in the sequence
    :return: Boolean indicating if entropy shows an increasing trend
    """
    if not return_seq:
        # Original logic for return_seq=False
        improvements = []
        
        for i in range(1, len(entropies)):
            current_entropy = entropies[i]
            previous_entropies = entropies[:i]
            
            better_count = sum(1 for prev in previous_entropies if current_entropy < prev)
            
            is_improved = (better_count / len(previous_entropies)) >= improved_ratio
            improvements.append(is_improved)
        
        improvement_ratio = sum(improvements) / len(improvements)
        
        return improvement_ratio >= threshold
    
    else:
        # New logic for return_seq=True
        num_layers, seq_len = entropies.shape
        improvements = []
        
        for i in range(1, num_layers):
            current_entropy = entropies[i]
            previous_entropy = entropies[i-1]
            
            # Compare current layer's entropy with previous layer's entropy for each token
            token_improvements = current_entropy < previous_entropy
            
            # Calculate the ratio of improved tokens for this layer transition
            layer_improvement_ratio = token_improvements.sum() / seq_len
            improvements.append(layer_improvement_ratio >= improved_ratio)
        
        # Calculate the overall improvement ratio across all layer transitions
        overall_improvement_ratio = sum(improvements) / len(improvements)
        
        return overall_improvement_ratio >= threshold

# def split_rationale(rationale, tokenizer):
#     """
#     Split the rationale into sentences based on splitters.
#     """
#     sentence_splitters = [tokenizer.convert_tokens_to_ids(t) for t in ['.']]
#     sentence_ends = [i for i, token in enumerate(rationale[0]) if token in sentence_splitters]
    
#     if not sentence_ends:
#         # Return the entire rationale as a single sentence, matching the dimension of split sentences
#         return [rationale[:, :]]
    
#     sentences = []
#     start = 0
#     for end in sentence_ends:
#         sentences.append(rationale[:, start:end+1])
#         start = end + 1  # Start of next sentence is right after the end of current sentence
    
#     # Add any remaining text after the last sentence splitter
#     if start < rationale.size(1):
#         sentences[-1] = torch.cat([sentences[-1], rationale[:, start:]], dim=1)

#     return sentences

def split_rationale(rationale, tokenizer):
    """
    Split the rationale into sentences based on '.' that are not between numbers.
    """

    sentence_splitter_id = tokenizer.convert_tokens_to_ids('.')
    tokens = tokenizer.convert_ids_to_tokens(rationale[0])

    sentence_ends = []
    for i, token_id in enumerate(rationale[0]):
        if token_id == sentence_splitter_id:
            prev_token = tokens[i - 1] if i > 0 else ''
            next_token = tokens[i + 1] if i + 1 < len(tokens) else ''

            prev_is_digit = bool(re.search(r'\d$', prev_token))
            next_is_digit = bool(re.search(r'^\d', next_token))

            if not (prev_is_digit and next_is_digit):
                sentence_ends.append(i)

    if not sentence_ends:
        # Return the entire rationale as a single sentence
        return [rationale[:, :]]

    sentences = []
    start = 0
    for end in sentence_ends:
        sentences.append(rationale[:, start:end+1])
        start = end + 1  # Start of next sentence is right after the end of current sentence

    # Add any remaining text after the last sentence splitter
    if start < rationale.size(1):
        sentences.append(rationale[:, start:])

    return sentences


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