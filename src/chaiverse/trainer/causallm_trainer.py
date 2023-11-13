import torch

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import AutoModelForCausalLM
from transformers import TrainingArguments, Trainer
from transformers import default_data_collator

from chaiverse.model.training_config import CausalLMLoraConfig
from chaiverse.model.lora_model import LoraModel


class CausalLMTrainer:

    def __init__(
            self,
            model_name,
            output_dir,
            data_collator = default_data_collator,
            learning_rate=2e-5,
            num_train_epochs=2,
            logging_strategy='steps',
            logging_steps=50,
            evaluation_strategy='no',
            eval_steps=None,
            device_map='auto',
            use_lora=True,
            lora_params = {
                'lora_dropout':0.05,
                },
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.data_collator = data_collator
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.logging_strategy = logging_strategy
        self.logging_steps = logging_steps
        self.evaluation_strategy = evaluation_strategy
        self.eval_steps = eval_steps
        self.device_map = device_map
        self.load_in_8bit = self._check_cuda_availability()
        self.use_lora = use_lora
        self.lora_params = lora_params
        self._initiate_training_config()

    def trainer_setup(self, data):
        self.instantiate_causallm_model()
        self.instantiate_causallm_trainer(data)

    def fit(self):
        self.trainer.train()

    def save(self, path=None):
        save_path = path or self.output_dir
        self.model.save_pretrained(save_path)

    def merge(self, path=None):
        self.lora_model.base_model = self._load_base_model(load_in_8bit=False)
        self.model = self.lora_model.merge(path=path)

    def push_to_hub(self, hf_path, private=True):
        self.model.push_to_hub(hf_path, private=private)

    def instantiate_causallm_model(self, **kwargs):
        self.model = self._load_base_model()
        if self.use_lora:
            self.lora_config = CausalLMLoraConfig(**self.lora_params)
            self.lora_model = LoraModel(
                base_model=self.model,
                lora_config=self.lora_config,
                output_dir = self.output_dir,
                )
            self.model = self.lora_model.model
            self.model.print_trainable_parameters()

    def instantiate_causallm_trainer(self, data):
        self.trainer = Trainer(
            model=self.model,
            args=self.training_config,
            data_collator=self.data_collator,
            train_dataset=data['train'])

    def update_training_config(self,**kwargs):
        config = self.training_config.to_dict()
        config.update(kwargs)
        self.training_config = TrainingArguments(**config)

    def _check_cuda_availability(self):
        if self.device_map == 'cpu' or (not torch.cuda.is_available()):
            return False
        else:
            return True

    def _load_base_model(self,load_in_8bit=None):
        load_in_8bit = load_in_8bit or self.load_in_8bit
        model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=load_in_8bit,
                device_map=self.device_map)
        return model

    def _initiate_training_config(self):
        self.training_config = TrainingArguments(
                output_dir=self.output_dir,
                auto_find_batch_size=True,
                learning_rate=self.learning_rate,
                num_train_epochs=self.num_train_epochs,
                logging_dir=f'{self.output_dir}/logs',
                logging_strategy=self.logging_strategy,
                logging_steps=self.logging_steps,
                evaluation_strategy=self.evaluation_strategy,
                eval_steps=self.eval_steps,
                save_strategy='no')

