from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training

class LoraModel:
    def __init__(
            self,
            base_model,
            lora_config,
            output_dir,
    ):
        self.base_model = base_model
        self.lora_config = lora_config
        self.output_dir = output_dir
        self.model = self._load_lora_model()

    def merge(self, path=None):
        save_path = path or self.output_dir
        model_to_merge = self.model.from_pretrained(self.base_model, save_path)
        return model_to_merge.merge_and_unload()

    def _load_lora_model(self):
        self.model = prepare_model_for_int8_training(self.base_model)
        return get_peft_model(self.model, self.lora_config)


