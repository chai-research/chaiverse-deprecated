from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import AutoModelForCausalLM
from transformers import DefaultDataCollator, default_data_collator
from transformers import TrainingArguments, Trainer


class LoraTrainer:

    def __init__(
            self,
            model_name,
            tokenizer,
            output_dir,
            learning_rate=2e-5,
            num_train_epochs=2,
            logging_strategy='steps',
            logging_steps=50,
            lora_r=16,
            lora_alpha=32,
            lora_target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            lora_dropout=0.05,
            lora_bias='none',
            lora_task_type=TaskType.CAUSAL_LM,
    ):
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.learning_rate = learning_rate
        self.num_train_epochs = num_train_epochs
        self.logging_strategy = logging_strategy
        self.logging_steps = logging_steps
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_target_modules = lora_target_modules
        self.lora_dropout = lora_dropout
        self.lora_bias = lora_bias
        self.lora_task_type = lora_task_type

    def train(self, data):
        self.instantiate_lora_model()
        self.instantiate_lora_trainer(data)
        self.trainer.train()

    def instantiate_lora_model(self, **kwargs):
        model = self._load_base_model()
        model = prepare_model_for_int8_training(model)
        self.model = self._load_lora_model(model)
        self.model.print_trainable_parameters()

    def instantiate_lora_trainer(self, data):
        # data_collator = DefulatDataCollator(
        #         self.tokenizer,
        #         # label_pad_token_id=-100,
        #         # pad_to_multiple_of=8,
        #         )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_config,
            data_collator=default_data_collator,
            train_dataset=data['train'])

    def save(self, path=''):
        model_path = self.output_dir if path=='' else path
        self.model.save_pretrained(model_path)
    
    def merge(self, path=''):
        model_path = self.output_dir if path=='' else path
        model_to_merge = self.model.from_pretrained(self._load_base_model(load_in_8bit=False), model_path)
        self.model = model_to_merge.merge_and_unload()

    def push_to_hub(self, model_url, private=True):
        self.model.push_to_hub(model_url, private=private)

    def _load_base_model(self, load_in_8bit=True):
        model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_8bit=load_in_8bit,
                device_map="auto")
        return model

    def _load_lora_model(self, model):
        model = get_peft_model(model, self.lora_config)
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

    @property
    def lora_config(self):
        return LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias=self.lora_bias,
                task_type=self.lora_task_type)
