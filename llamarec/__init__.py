# coding=utf-8
# Copyright 2025 LLaMA-Rec Team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LLaMA-Rec: 独立的推荐系统 LLaMA 实现

这个模块将 LLaMA-Rec 从 transformers 源码中解耦，
作为独立模块便于维护和修改。
"""

from .configuration import LlamaRecConfig
from .configuration_slot import LlamaRecSlotConfig
from .modeling import (
    LlamaRecModel,
    LlamaRecForCausalLM,
    LlamaRecForSequenceClassification,
    LlamaRecForQuestionAnswering,
    LlamaRecForTokenClassification,
    LlamaRecPreTrainedModel,
    LlamaRecRMSNorm,
    LlamaRecRotaryEmbedding,
    LlamaRecMLP,
    LlamaRecAttention,
    LlamaRecDecoderLayer,
)
from .modeling_slot import LlamaRecSlotForCausalLM
from .modeling_t5_slot import T5RecSlotForConditionalGeneration

__version__ = "0.1.0"

__all__ = [
    # Configuration
    "LlamaRecConfig",
    "LlamaRecSlotConfig",
    
    # Main Models
    "LlamaRecModel",
    "LlamaRecForCausalLM",  # 主要使用的模型
    "LlamaRecSlotForCausalLM",
    "T5RecSlotForConditionalGeneration",
    "LlamaRecForSequenceClassification",
    "LlamaRecForQuestionAnswering",
    "LlamaRecForTokenClassification",
    "LlamaRecPreTrainedModel",
    
    # Model Components (可以单独使用或修改)
    "LlamaRecRMSNorm",
    "LlamaRecRotaryEmbedding",
    "LlamaRecMLP",
    "LlamaRecAttention",
    "LlamaRecDecoderLayer",
]
from transformers import AutoConfig, AutoModelForCausalLM
AutoConfig.register("llama-rec", LlamaRecConfig)
AutoModelForCausalLM.register(LlamaRecConfig, LlamaRecForCausalLM)
AutoConfig.register("llama-rec-slot", LlamaRecSlotConfig)
AutoModelForCausalLM.register(LlamaRecSlotConfig, LlamaRecSlotForCausalLM)
