import torch

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import default_data_collator

from chaiverse.training_config import CausalLMLoraConfig

class LoraModel:
    def __init__(
            self,
            model,
            lora_config,
    ):
        self.lora_config = lora_config
        self.model = self._load_lora_model(model)

    def _load_lora_model(self,model):
        self.model = prepare_model_for_int8_training(model)
        return get_peft_model(model, self.lora_config)

class LoraTrainer:

    def __init__(
            self,
            model_name,
            output_dir,
            learning_rate=2e-5,
            num_train_epochs=2,
            logging_strategy='steps',
            logging_steps=50,
            device_map='auto',
            lora_params = {
                'lora_dropout':0.05,
                },
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.logging_strategy = logging_strategy
        self.logging_steps = logging_steps
        self.device_map = device_map
        self.load_in_8bit = self._check_cuda_availability()
        self.lora_params = lora_params

    def trainer_setup(self, data):
        self.instantiate_lora_model()
        self.instantiate_lora_trainer(data)

    def fit(self):
        self.trainer.train()

    def save(self, path=None):
        save_path = path or self.output_dir
        self.model.save_pretrained(save_path)

    def merge(self, path=None):
        save_path = path or self.output_dir
        base_model = self._load_base_model()
        model_to_merge = self.model.from_pretrained(base_model, save_path)
        self.model = model_to_merge.merge_and_unload()

    def push_to_hub(self, hf_path, private=True):
        self.model.push_to_hub(hf_path, private=private)

    def instantiate_lora_model(self, **kwargs):
        model = self._load_base_model()
        self.lora_config = CausalLMLoraConfig(**self.lora_params)
        self.model = LoraModel(model=model,lora_config=self.lora_config).model
        self.model.print_trainable_parameters()

    def instantiate_lora_trainer(self, data):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_config,
            data_collator=default_data_collator,
            train_dataset=data['train'])

    def _check_cuda_availability(self):
        if self.device_map == 'cpu' or (not torch.cuda.is_available()):
            return False
        else:
            return True

    def _load_base_model(self):
        model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=self.load_in_8bit,
                device_map=self.device_map)
        return model

    @property
    def training_config(self):
        return TrainingArguments(
                output_dir=self.output_dir,
                auto_find_batch_size=True,
                learning_rate=self.learning_rate,
                num_train_epochs=self.num_train_epochs,
                logging_dir=f'{self.output_dir}/logs',
                logging_strategy=self.logging_strategy,
                logging_steps=self.logging_steps,
                save_strategy='no')

