from transformers import PretrainedConfig

class MistralConfig(PretrainedConfig):
    def __init__(
        self,
        base_model='mistralai/Mistral-7B-v0.1',
        tokenizer_name='mistralai/Mistral-7B-v0.1',
        **kwargs,
    ):
        self.base_model = base_model
        self.tokenizer_name = tokenizer_name
        super().__init__(**kwargs)