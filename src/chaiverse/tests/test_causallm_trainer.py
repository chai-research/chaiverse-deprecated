import pytest
import os
import torch

from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType, PeftModel
from transformers import AutoModelForCausalLM
import tempfile

from chaiverse.dataset import DatasetLoader, CausalDatasetBuilder
from chaiverse.tokenizer import LlamaTokenizer

from chaiverse.model.lora_model import LoraModel
from chaiverse.model.training_config import CausalLMLoraConfig
from chaiverse.trainer.causallm_trainer import CausalLMTrainer

from mock import patch, Mock


@pytest.fixture
def tiny_base_model_id():
    return "HuggingFaceH4/tiny-random-LlamaForCausalLM"


@pytest.fixture
def lora_config():
    return LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                bias='none',
                task_type=TaskType.CAUSAL_LM,
            )


@pytest.fixture
def tiny_base_model(tiny_base_model_id):
    r"""
    Simply load a tiny llama base model
    """
    return AutoModelForCausalLM.from_pretrained(
            tiny_base_model_id,
            load_in_8bit=False,
            device_map='cpu')


@pytest.fixture
@patch("chaiverse.logging_utils.requests.post", Mock())
def tiny_model(tiny_base_model_id, tmpdir):
    r"""
    Simply creates a peft model and checks that it can be loaded.
    """
    model = CausalLMTrainer(
            model_name=tiny_base_model_id,
            output_dir=f'{tmpdir}/lora_unittest',
            device_map='cpu',
            use_lora=False,
            )
    model.instantiate_causallm_model()
    return model


@pytest.fixture
@patch("chaiverse.logging_utils.requests.post", Mock())
def tiny_lora_model(tiny_base_model_id, tmpdir):
    r"""
    Simply creates a peft model and checks that it can be loaded.
    """
    model = CausalLMTrainer(
            model_name=tiny_base_model_id,
            output_dir=f'{tmpdir}/lora_unittest',
            device_map='cpu'
            )
    model.instantiate_causallm_model()
    return model


@pytest.fixture
@patch("chaiverse.logging_utils.requests.post", Mock())
def data():
    data_path = 'ChaiML/chaiverse_lora_testing_fandom_IO'
    data_loader = DatasetLoader(
        hf_path=data_path,
        data_samples=10,
        validation_split_size=0.1,
        shuffle=True,
        )
    df = data_loader.load()
    tokenizer = LlamaTokenizer()
    data_builder = CausalDatasetBuilder(
        tokenizer_loader=tokenizer,
        block_size=1024,
        )
    return data_builder.generate(df)


def test_load_base_model(tiny_base_model):
    assert tiny_base_model is not None


def test_instantiate_calsallm_model(tiny_model, tiny_lora_model):
    assert tiny_model.model is not None
    assert tiny_lora_model.model is not None


def test_check_lora_model_nb_trainable_params(tiny_lora_model):
    r"""
    Check that the number of trainable parameters is correct.
    """
    nb_trainable_params = sum(p.numel() for p in tiny_lora_model.model.parameters() if p.requires_grad)
    assert nb_trainable_params == 2048


def test_save_pretrained_model(tiny_model):
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_model.save(path=tmp_dir)

        # check that the files `generation_config.json`, `adapter_config.json`,
        # 'model.safetensors' are in the directory
        assert os.path.isfile(f"{tmp_dir}/generation_config.json"), f"{tmp_dir}/generation_config does not exist, it has {os.listdir(tmp_dir)}"
        assert os.path.exists(f"{tmp_dir}/config.json"), f"{tmp_dir}/config.json does not exist, it has {os.listdir(tmp_dir)}"
        assert os.path.exists(f"{tmp_dir}/model.safetensors"), f"{tmp_dir}/pytorch.safetensors does not exist, it has {os.listdir(tmp_dir)}"


def test_save_pretrained_lora(tiny_lora_model):
    r"""
    Check that the model can be saved and loaded properly.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_lora_model.save(path=tmp_dir)

        # check that the files `adapter_model.bin` and `adapter_config.json` are in the directory
        assert os.path.isfile(f"{tmp_dir}/adapter_model.bin"), f"{tmp_dir}/adapter_model.bin does not exist"
        assert os.path.exists(f"{tmp_dir}/adapter_config.json"), f"{tmp_dir}/adapter_config.json does not exist"


def test_load_pretrained_lora(tiny_lora_model, tiny_base_model):
    r"""
    Check that the model can be saved and loaded properly.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_lora_model.save(path=tmp_dir)
        pretrained_lora_model = PeftModel.from_pretrained(tiny_base_model, tmp_dir)

        # check all the weights are the same
        for p1, p2 in zip(tiny_lora_model.model.named_parameters(), pretrained_lora_model.named_parameters()):
            if p1[0] not in ["v_head.summary.weight", "v_head.summary.bias"]:
                assert torch.allclose(p1[1], p2[1]), f"{p1[0]} != {p2[0]}"


def test_continue_training_lora_model(tiny_lora_model, tiny_base_model):
    r"""
    Load peft and checks that it can continue training.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_lora_model.save(path=tmp_dir)
        pretrained_lora_model = PeftModel.from_pretrained(tiny_base_model, tmp_dir, is_trainable=True)
        nb_trainable_params = sum(p.numel() for p in pretrained_lora_model.parameters() if p.requires_grad)
        assert nb_trainable_params == 2048


def test_merge_model(tiny_lora_model):
    r"""
    Check that the model can be merged and saved.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_lora_model.save(path=tmp_dir)
        tiny_lora_model.merge(path=tmp_dir)
        tiny_lora_model.model.save_pretrained(tmp_dir+'/merged')

        assert "LlamaForCausalLM" in str(type(tiny_lora_model.model))

        # check that the files `generation_config.json`, `adapter_config.json`,
        # 'model.safetensors' are in the directory
        assert os.path.isfile(f"{tmp_dir}/merged/generation_config.json"), f"{tmp_dir}/merged/generation_config does not exist, it has {os.listdir(tmp_dir)}"
        assert os.path.exists(f"{tmp_dir}/merged/config.json"), f"{tmp_dir}/merged/config.json does not exist, it has {os.listdir(tmp_dir)}"
        assert os.path.exists(f"{tmp_dir}/merged/model.safetensors"), f"{tmp_dir}/merged/pytorch.safetensors does not exist, it has {os.listdir(tmp_dir)}"


def test_training_model(tiny_model, data):
    """
    Check the training works
    """
    tiny_model.instantiate_causallm_trainer(data)
    previous_trainable_params = {n: param.clone() for n, param in tiny_model.trainer.model.named_parameters()}

    tiny_model.trainer.train()

    assert tiny_model.trainer.state.log_history[-1]["train_loss"] is not None

    # check the params have changed
    for n, param in previous_trainable_params.items():
        new_param = tiny_model.trainer.model.get_parameter(n)
        # check the params have changed - ignore 0 biases
        if param.sum() != 0:
            assert not torch.equal(param, new_param)


def test_training_lora_model(tiny_lora_model, data):
    r"""
    Check the lora training works
    """

    tiny_lora_model.instantiate_causallm_trainer(data)

    previous_trainable_params = {}
    previous_non_trainable_params = {}

    trainable_params_name = ["lora", "modules_to_save"]

    for n, param in tiny_lora_model.model.named_parameters():
        if any([t in n for t in trainable_params_name]):
            previous_trainable_params[n] = param.clone()
        else:
            previous_non_trainable_params[n] = param.clone()

    tiny_lora_model.trainer.train()

    assert tiny_lora_model.trainer.state.log_history[-1]["train_loss"] is not None

    new_params = {}
    for n, param in tiny_lora_model.model.named_parameters():
        new_params[n] = param.clone()

    # check the trainable params have changed
    for n, param in previous_trainable_params.items():
        if n in new_params.keys():
            new_param = new_params[n]
            assert not torch.allclose(param, new_param, atol=1e-12, rtol=1e-12)

    # check the non trainable params have not changed
    for n, param in previous_non_trainable_params.items():
        if n in new_params.keys():
            new_param = new_params[n]
            assert torch.allclose(param, new_param, atol=1e-12, rtol=1e-12)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_lora_model.save(path=tmp_dir)

        # check that the files `adapter_model.bin` and `adapter_config.json` are in the directory
        assert os.path.isfile(f"{tmp_dir}/adapter_model.bin"), f"{tmp_dir}/adapter_model.bin does not exist"
        assert os.path.exists(f"{tmp_dir}/adapter_config.json"), f"{tmp_dir}/adapter_config.json does not exist"

        tiny_lora_model.merge(path=tmp_dir)
        tiny_lora_model.model.save_pretrained(tmp_dir+'/merged')

        # check that the files `generation_config.json`, `adapter_config.json`,
        # 'model.safetensors' are in the directory
        assert os.path.isfile(f"{tmp_dir}/merged/generation_config.json"), f"{tmp_dir}/merged/generation_config does not exist, it has {os.listdir(tmp_dir)}"
        assert os.path.exists(f"{tmp_dir}/merged/config.json"), f"{tmp_dir}/merged/config.json does not exist, it has {os.listdir(tmp_dir)}"
        assert os.path.exists(f"{tmp_dir}/merged/model.safetensors"), f"{tmp_dir}/merged/pytorch.safetensors does not exist, it has {os.listdir(tmp_dir)}"


def test_load_merge_model(tiny_lora_model):
    r"""
    Check that the merged model can be loaded correctly.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        tiny_lora_model.save(path=tmp_dir)
        tiny_lora_model.merge(path=tmp_dir)
        tiny_lora_model.model.save_pretrained(tmp_dir+'/merged')

        merged_model = AutoModelForCausalLM.from_pretrained(tmp_dir+'/merged')

        assert "LlamaForCausalLM" in str(type(merged_model))

        # check all the weights are the same
        for p1, p2 in zip(tiny_lora_model.model.named_parameters(), merged_model.named_parameters()):
            if p1[0] not in ["v_head.summary.weight", "v_head.summary.bias"]:
                assert torch.allclose(p1[1], p2[1]), f"{p1[0]} != {p2[0]}"
