import os
import torch
import pytest

import tempfile
from transformers import AutoModelForSequenceClassification

from chaiverse.dataset import DatasetLoader, RewardDatasetBuilder
from chaiverse.tokenizer import GPT2Tokenizer
from chaiverse.model.reward_model import RewardClassificationTrainer
from mock import patch, Mock


@pytest.fixture
def tiny_base_model_id():
    return "hf-internal-testing/tiny-random-gpt2"


@pytest.fixture
def tiny_base_model(tiny_base_model_id):
    return AutoModelForSequenceClassification.from_pretrained(
                tiny_base_model_id,
                num_labels=2,
                device_map='cpu',
                )


@pytest.fixture
def tokenizer_loader():
    return GPT2Tokenizer(
            padding_side='right',
            truncation_side='left',
            )


@pytest.fixture
@patch("chaiverse.logging_utils.requests.post", Mock())
def data(tokenizer_loader):
    data_path = 'ChaiML/tiny_chai_prize_reward_model_data'
    data_loader = DatasetLoader(
                hf_path=data_path,
                data_samples=10,
                validation_split_size=0.1,
                )
    df = data_loader.load()
    data_builder = RewardDatasetBuilder(
            tokenizer_loader=tokenizer_loader,
            block_size=32,
            )
    return data_builder.generate(df, n_jobs=1)


@pytest.fixture
@patch("chaiverse.logging_utils.requests.post", Mock())
def tiny_model(tiny_base_model_id, tokenizer_loader, tmp_path):
    model = RewardClassificationTrainer(
            model_name=tiny_base_model_id,
            tokenizer_loader=tokenizer_loader,
            device_map="cpu",
            output_dir=f'{tmp_path}/test_reward_model',
            learning_rate=1e-5,
            num_train_epochs=1,
            bf16=False,
            no_cuda=True,
            use_lora=False,
            )
    model.tokenizer = model.tokenizer_loader.load()
    model.instantiate_reward_model()
    return model


@pytest.fixture
@patch("chaiverse.logging_utils.requests.post", Mock())
def gpt2_model(tokenizer_loader, tmp_path):
    model = RewardClassificationTrainer(
            model_name="gpt2",
            tokenizer_loader=tokenizer_loader,
            device_map="cpu",
            output_dir=f'{tmp_path}/test_reward_model',
            learning_rate=1e-5,
            num_train_epochs=1,
            bf16=False,
            no_cuda=True,
            use_lora=False,
            )
    model.tokenizer = model.tokenizer_loader.load()
    model.instantiate_reward_model()
    return model


@pytest.fixture
@patch("chaiverse.logging_utils.requests.post", Mock())
def lora_gpt2_model(tokenizer_loader, tmp_path):
    model = RewardClassificationTrainer(
            model_name="gpt2",
            tokenizer_loader=tokenizer_loader,
            device_map="cpu",
            output_dir=f'{tmp_path}/test_reward_model',
            learning_rate=1e-5,
            num_train_epochs=1,
            bf16=False,
            no_cuda=True,
            )
    model.tokenizer = model.tokenizer_loader.load()
    model.instantiate_reward_model()
    return model


def test_load_base_model(tiny_base_model):
    assert tiny_base_model is not None


def test_instantiate_reward_model(tiny_model):
    assert tiny_model.model is not None
    assert "GPT2ForSequenceClassification" in str(type(tiny_model.model))


def test_check_reward_model_nb_trainable_params(tiny_model):
    r"""
    Check that the number of trainable parameters is correct.
    """
    nb_trainable_params = sum(p.numel() for p in tiny_model.model.parameters() if p.requires_grad)
    assert nb_trainable_params == 112032


def test_instantiate_reward_trainer(tiny_model, data):
    tiny_model.tokenizer = tiny_model.tokenizer_loader.load()
    tiny_model.instantiate_reward_trainer(data)
    assert tiny_model.trainer is not None


def test_save_pretrained_reward(tiny_model):
    r"""
    Check that the model can be saved and loaded properly.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_model.save(path=tmp_dir)

        # check that the files `model.safetensors` and `config.json` are in the directory
        assert os.path.isfile(f"{tmp_dir}/model.safetensors"), f"{tmp_dir}/model.safetensors does not exist, it has {os.listdir(tmp_dir)}"
        assert os.path.exists(f"{tmp_dir}/config.json"), f"{tmp_dir}/config.json does not exist, it has {os.listdir(tmp_dir)}"


def test_reward_trainer(gpt2_model, data):
    gpt2_model.instantiate_reward_trainer(data)

    previous_trainable_params = {n: param.clone() for n, param in gpt2_model.trainer.model.named_parameters()}

    gpt2_model.fit()

    assert gpt2_model.trainer.state.log_history[-1]['train_loss'] is not None

    # check the params have changed
    for n, param in previous_trainable_params.items():
        new_param = gpt2_model.trainer.model.get_parameter(n)
        # check the params have changed - ignore 0 biases
        if param.sum() != 0:
            assert not torch.allclose(param, new_param, atol=1e-12, rtol=1e-12)
    preds = gpt2_model.trainer.predict(data['train'])
    assert preds.predictions.shape == (len(data['train']), 2)


def test_lora_reward_trainer(lora_gpt2_model, data):
    lora_gpt2_model.instantiate_reward_trainer(data)

    previous_trainable_params = {}
    previous_non_trainable_params = {}

    trainable_params_name = ["lora", "modules_to_save"]

    for n, param in lora_gpt2_model.model.named_parameters():
        if any([t in n for t in trainable_params_name]):
            previous_trainable_params[n] = param.clone()
        else:
            previous_non_trainable_params[n] = param.clone()

    lora_gpt2_model.fit()

    assert lora_gpt2_model.trainer.state.log_history[-1]["train_loss"] is not None

    new_params = {}
    for n, param in lora_gpt2_model.model.named_parameters():
        new_params[n] = param.clone()

    # check the trainable params have changed
    for n, param in previous_trainable_params.items():
        if n in new_params.keys() and param.sum() != 0:
            new_param = new_params[n]
            assert not torch.allclose(param, new_param, atol=1e-12, rtol=1e-12)

    # check the non trainable params have not changed
    for n, param in previous_non_trainable_params.items():
        if n in new_params.keys() and param.sum() != 0:
            new_param = new_params[n]
            assert torch.allclose(param, new_param, atol=1e-12, rtol=1e-12)


    with tempfile.TemporaryDirectory() as tmp_dir:
        lora_gpt2_model.save(path=tmp_dir)

        # check that the files `adapter_model.bin` and `adapter_config.json` are in the directory
        assert os.path.isfile(f"{tmp_dir}/adapter_model.bin"), f"{tmp_dir}/adapter_model.bin does not exist, it has {os.listdir(tmp_dir)}"
        assert os.path.exists(f"{tmp_dir}/adapter_config.json"), f"{tmp_dir}/adapter_config.json does not exist, it has {os.listdir(tmp_dir)}"

        lora_gpt2_model.merge(path=tmp_dir)
        lora_gpt2_model.model.save_pretrained(tmp_dir+'/merged')

        # check that the files `adapter_config.json`, 'model.safetensors' are in the directory
        assert os.path.exists(f"{tmp_dir}/merged/config.json"), f"{tmp_dir}/merged/config.json does not exist, it has {os.listdir(tmp_dir+'/merged')}"
        assert os.path.exists(f"{tmp_dir}/merged/model.safetensors"), f"{tmp_dir}/merged/pytorch.safetensors does not exist, it has {os.listdir(tmp_dir+'/merged')}"
