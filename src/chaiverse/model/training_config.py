from dataclasses import dataclass, field
from typing import List, Optional, Union
from peft import LoraConfig, TaskType

@dataclass
class RewardLoraConfig(LoraConfig):
    task_type: TaskType = field(default=TaskType.SEQ_CLS)
    target_modules: Optional[Union[List[str], str]] = field(default = ('c_attn','c_proj'))

@dataclass
class CausalLMLoraConfig(LoraConfig):
    task_type: TaskType = field(default=TaskType.CAUSAL_LM)
    target_modules: Optional[Union[List[str], str]] = field(default=('q_proj','k_proj','v_proj','o_proj'))
