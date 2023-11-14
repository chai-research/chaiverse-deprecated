import logging
import os
import huggingface_hub

import chai_guanaco as chai

from chaiverse.dataset import DatasetLoader, RewardDatasetBuilder
from chaiverse.tokenizer import GPT2Tokenizer
from chaiverse.trainer.reward_trainer import RewardClassificationTrainer, RewardRegressionTrainer


if __name__ == '__main__':
    data_path = 'ChaiML/20231012_chai_prize_reward_model_data'
    data_loader = DatasetLoader(
            hf_path=data_path,
            data_samples=10000,
            validation_split_size=0.1,
            shuffle=True,
            )
    df = data_loader.load()

    tokenizer_loader = GPT2Tokenizer(
            padding_side='right',
            truncation_side='left',
            )
    data_builder = RewardDatasetBuilder(
            tokenizer_loader=tokenizer_loader,
            block_size=512,
            )
    data = data_builder.generate(df)

    lora_params = {'r':512}

    model = RewardClassificationTrainer(
            model_name='gpt2',
            tokenizer_loader=tokenizer_loader,
            output_dir='./test_reward',
            learning_rate=1e-5,
            num_train_epochs=1,
            bf16=True,
            logging_strategy='steps',
            logging_steps=2,
            eval_strategy='steps',
            eval_steps=10,
            use_lora=True,
            lora_params=lora_params
            )

    model.update_training_config(per_device_train_batch_size=32)

    model.trainer_setup(data)
    model.fit()
    model.save()
