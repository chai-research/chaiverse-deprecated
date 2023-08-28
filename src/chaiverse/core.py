import yaml
import os

import chaiverse as cv
from datasets import load_dataset as load_hf_dataset


def load_dataset(repo_url):
    dataset = load_hf_dataset(repo_url)
    dataset.repo_url = repo_url
    return dataset


class ChaiLLM():
    def __init__(self):
        self._output_dir = None

    def fit(
            self,
            dataset,
            output_dir,
            num_epochs=1,
            eval_steps=200,
            learning_rate=2e-5,
            sequence_len=1024,
            logging_steps=1,
            val_set_size=0.01,
            gradient_accumulation_steps=2,
            micro_batch_size=4,
            bf16=True,
            gradient_checkpointing=True,
            wandb_project=None,
            wandb_entity=None,
            ):
        self._set_output_dir(output_dir)
        self._save_yaml_file(
            base_model=self.model_url,
            base_model_config=self.tokenizer_url,
            dataset=dataset.repo_url,
            dataset_prepared_path='last_run_prepared',
            val_set_size=val_set_size,
            output_dir=self.output_dir,
            sequence_len=sequence_len,
            wandb_project=wandb_project,
            wandb_entity=wandb_entity,
            gradient_accumulation_steps=gradient_accumulation_steps,
            micro_batch_size=micro_batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            bf16=bf16,
            gradient_checkpointing=gradient_checkpointing,
            logging_steps=logging_steps,
            flash_attention=flash_attention,
            eval_steps=eval_steps
        )

    def push_to_hub(self, model_url, private):
        pass

    @property
    def output_dir(self):
        return self._output_dir

    def _set_output_dir(self, output_dir):
        cv.utils.ensure_dir_exists(output_dir)
        self._output_dir = output_dir
    
    def _save_yaml_file(self, **configs):
        with open(os.path.join(self.output_dir, 'trainer_config.yaml', 'w')) as f:
            yaml.dump(configs, f)


class LLaMA7b(ChaiLLM):
    @property
    def model_url(self):
        return 'NousResearch/Llama-2-7b-hf'

    @property
    def tokenizer_url(self):
        return 'NousResearch/Llama-2-7b-hf'

    @property
    def use_flash_attention(sefl):
        return True
