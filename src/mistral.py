import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteriaList, GenerationConfig, LogitsProcessorList
import sys
from configuration import MistralConfig
sys.path.append("..")
from utils import get_sep_position, DoubleEOSStoppingCriteria, DoubleEOSLogitsProcessor

class Mistral(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.base_model = AutoModelForCausalLM.from_pretrained(config.base_model)
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_model.resize_token_embeddings(len(self.tokenizer))
        self.lm_head = self.base_model.get_output_embeddings() # For computing Entropy
        
    def forward(self, input_ids, labels=None, output_hidden_states=False):
        outputs = self.base_model(input_ids=input_ids, labels=labels, output_hidden_states=output_hidden_states)
        return outputs

    def compute_loss(self, input_ids, labels):
        outputs = self.forward(input_ids=input_ids, labels=labels)
        logits = outputs.logits

        labels_pred = logits.argmax(-1)
        mask = labels[...,1:].ge(0)
        correct_tokens = ((labels_pred[...,:-1] == labels[...,1:]) * mask).sum()
        total_tokens = mask.sum()
        token_accuracy = correct_tokens / total_tokens

        shift_logits = logits[..., :-1, :].contiguous()
        '''
        Verify the Gradient
        shift_logits = shift_logits - shift_logits.max(dim=-1, keepdim=True)[0]
        '''
        print('Logits min:', shift_logits.min())
        print('Logits max:', shift_logits.max())
        print('Logits mean:', shift_logits.mean())

        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        outputs.loss = loss
        outputs.token_accuracy = token_accuracy
        outputs.total_correct = correct_tokens
        outputs.total_loss = loss * total_tokens
        outputs.total_tokens = total_tokens
        return outputs
    
    def compute_entropy(self, input_ids, labels, interval):
        with torch.no_grad():
            outputs = self.forward(input_ids=input_ids, labels=labels, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        # print("Last Hidden_states")
        # print(hidden_states[-1])

        for name, param in self.base_model.named_parameters():
            if param is None:
                print("NAN VALUE IS DETECTED")
                print(f"Layer: {name} | Param: {param}")
                exit()
            else:
                pass

        num_layers = len(hidden_states)
        selected_indices = list(range(0, num_layers, interval))
        if (num_layers - 1) not in selected_indices:
            selected_indices.append(num_layers - 1)  # Ensure the last layer is included

        cross_entropies = []
        for i in selected_indices:
            layer_hidden = hidden_states[i]  # Shape: [batch_size, seq_len, hidden_size]
            with torch.no_grad():
                layer_logits = self.lm_head(layer_hidden)  # Shape: [batch_size, seq_len, vocab_size]
                layer_cross_entropy = self.calculate_entropy(layer_logits, input_ids)
            cross_entropies.append(layer_cross_entropy)

        # Stack the results: [num_selected_layers, batch_size]
        cross_entropies = torch.stack(cross_entropies)

        # Transpose to get [batch_size, num_selected_layers]
        cross_entropies = cross_entropies.t()
        print("Cross Entropy List", cross_entropies)

        outputs.cross_entropies = cross_entropies
        return outputs

    def calculate_entropy(self, logits, input_ids):
        # logits shape: [batch_size, seq_len, vocab_size]
        # input_ids shape: [batch_size, seq_len]
        
        # Shift logits and input_ids for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        # Calculate cross-entropy
        loss_fct = CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id, reduction='none')
        cross_entropy = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Reshape cross-entropy to [batch_size, seq_len]
        cross_entropy = cross_entropy.view(shift_labels.size())
        
        # Create mask for non-padding tokens
        mask = (shift_labels != self.tokenizer.pad_token_id).float()
        
        # Compute mean per sample
        sample_cross_entropy = (cross_entropy * mask).sum(dim=-1) / mask.sum(dim=-1)
        
        return sample_cross_entropy  # Shape: [batch_size]
    
    def generate(self, input_ids, max_new_tokens=512, num_beams=1, stop_on_two_eos=True, test=False, use_demo=False):
        if test and use_demo:
            sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=15)
        else:
            sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        
        generation_config = GenerationConfig.from_model_config(self.base_model.config)
        
        # Since there's one eos after CoT and another after final answer, we need to wait for two eos
        generation_config = GenerationConfig.from_model_config(self.base_model.config)
        if hasattr(generation_config, 'pad_token_id'):
            #generation_config.pad_token_id = -1 #TODO: this might not be necessary
            generation_config.pad_token_id = None #TODO: this might not be necessary
        if stop_on_two_eos:
            generation_config.eos_token_id = -1
            logits_processor = LogitsProcessorList([DoubleEOSLogitsProcessor(self.tokenizer.eos_token_id)])
            stopping_criteria = StoppingCriteriaList([DoubleEOSStoppingCriteria(self.tokenizer.eos_token_id)])
        else:
            logits_processor = None
            stopping_criteria = None

        batch_size = input_ids.shape[0]
        beam_output = []
        for i in range(batch_size):
            input_ids_i = input_ids[i:i+1]
            sep_positions_i = sep_positions[i:i+1]
            input_ids_i = input_ids_i[:, :sep_positions_i+1]
            beam_output_i = self.base_model.generate(
                input_ids=input_ids_i,
                generation_config=generation_config,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                num_return_sequences=1,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
            )
            beam_output.append(beam_output_i)
        return beam_output

    @classmethod
    def from_pretrained(cls, pretrained_path):
        config = MistralConfig.from_pretrained(pretrained_path)
        model = cls(config)
        state_dict = torch.load(os.path.join(pretrained_path, 'state_dict.bin'))
        model.load_state_dict(state_dict)
        return model

    def save_pretrained(self, save_directory, state_dict=None):
        print(f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        if state_dict is None:
            state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))