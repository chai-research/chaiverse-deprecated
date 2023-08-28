from axolotl.utils.models import load_model, load_tokenizer


class ChaiLLM():
    def __init__(self):
        pass

    def fit(self, dataset):
        pass

    def push_to_hub(self, model_url, private):
        pass


class LLaMA7b(ChaiLLM):
    @property
    def model_url(self):
        return 'NousResearch/Llama-2-7b-hf'

    @property
    def tokenizer_url(self):
        return 'NousResearch/Llama-2-7b-hf'
