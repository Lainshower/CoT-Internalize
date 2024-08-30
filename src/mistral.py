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
        
    def forward(self, input_ids, labels=None):
        outputs = self.base_model(input_ids=input_ids, labels=labels)
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
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        outputs.loss = loss
        outputs.token_accuracy = token_accuracy
        outputs.total_correct = correct_tokens
        outputs.total_loss = loss * total_tokens
        outputs.total_tokens = total_tokens
        return outputs

    def generate(self, input_ids, max_new_tokens=512, num_beams=1, stop_on_two_eos=True, test=False):
        if test == False:
            sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id)
        else:
            sep_positions = get_sep_position(input_ids, self.tokenizer.eos_token_id, skip=15)
        
        generation_config = GenerationConfig.from_model_config(self.base_model.config)
        
        if stop_on_two_eos:
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

    def save_pretrained(self, save_directory):
        print(f'Saving to {save_directory}')
        self.config.save_pretrained(save_directory)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(save_directory, 'state_dict.bin'))