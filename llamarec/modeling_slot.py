# coding=utf-8
from typing import Optional, Union

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .configuration_slot import LlamaRecSlotConfig
from .modeling import KwargsForCausalLM, LlamaRecModel, LlamaRecPreTrainedModel


class LlamaRecSlotForCausalLM(LlamaRecPreTrainedModel, GenerationMixin):
    config_class = LlamaRecSlotConfig
    _tied_weights_keys = []
    _tp_plan = {}
    _pp_plan = {}

    def __init__(self, config: LlamaRecSlotConfig):
        super().__init__(config)
        self.model = LlamaRecModel(config)
        self.vocab_size = config.vocab_size
        self.item_token_size = int(config.item_token_size)
        self.use_slot_input_embeddings = bool(config.use_slot_input_embeddings)
        self.use_slot_prelogits_embeddings = bool(config.use_slot_prelogits_embeddings)
        self.use_slot_output_heads = bool(config.use_slot_output_heads)

        self.slot_embeddings = nn.Embedding(self.item_token_size, config.hidden_size)
        if self.use_slot_output_heads:
            self.lm_heads = nn.ModuleList(
                [nn.Linear(config.hidden_size, config.vocab_size, bias=False) for _ in range(self.item_token_size)]
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        if self.use_slot_output_heads:
            return self.lm_heads
        return self.lm_head

    def _compute_slot_ids(
        self,
        attention_mask: Optional[torch.Tensor],
        token_length: int,
        device: torch.device,
    ) -> torch.LongTensor:
        if attention_mask is None:
            base = torch.arange(token_length, device=device, dtype=torch.long)
            return (base.unsqueeze(0) % self.item_token_size)

        if attention_mask.dim() != 2:
            raise ValueError("Slot-aware model expects a 2D attention mask")

        full_length = attention_mask.size(1)
        visible_counts = attention_mask.to(device=device, dtype=torch.long).sum(dim=1)

        if token_length == full_length:
            running_visible = attention_mask.to(device=device, dtype=torch.long).cumsum(dim=1) - 1
            running_visible = torch.clamp(running_visible, min=0)
            return running_visible % self.item_token_size

        start_positions = visible_counts - token_length
        offsets = torch.arange(token_length, device=device, dtype=torch.long).unsqueeze(0)
        current_positions = start_positions.unsqueeze(1) + offsets
        current_positions = torch.clamp(current_positions, min=0)
        return current_positions % self.item_token_size

    def _compute_slot_and_item_ids(
        self,
        attention_mask: Optional[torch.Tensor],
        token_length: int,
        device: torch.device,
    ) -> tuple[torch.LongTensor, torch.LongTensor]:
        if attention_mask is None:
            base = torch.arange(token_length, device=device, dtype=torch.long)
            return base % self.item_token_size, torch.div(base, self.item_token_size, rounding_mode="floor")

        if attention_mask.dim() != 2:
            raise ValueError("Slot-aware model expects a 2D attention mask")

        full_length = attention_mask.size(1)
        visible_counts = attention_mask.to(device=device, dtype=torch.long).sum(dim=1)

        if token_length == full_length:
            running_visible = attention_mask.to(device=device, dtype=torch.long).cumsum(dim=1) - 1
            current_positions = torch.clamp(running_visible, min=0)
        else:
            start_positions = visible_counts - token_length
            offsets = torch.arange(token_length, device=device, dtype=torch.long).unsqueeze(0)
            current_positions = start_positions.unsqueeze(1) + offsets
            current_positions = torch.clamp(current_positions, min=0)

        slot_ids = current_positions % self.item_token_size
        item_ids = torch.div(current_positions, self.item_token_size, rounding_mode="floor")
        return slot_ids, item_ids

    def _compute_logits(
        self,
        hidden_states: torch.Tensor,
        slot_ids: torch.LongTensor,
    ) -> torch.Tensor:
        if self.use_slot_output_heads:
            all_logits = torch.stack([head(hidden_states) for head in self.lm_heads], dim=2)
            gather_index = slot_ids.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.vocab_size)
            return torch.gather(all_logits, dim=2, index=gather_index).squeeze(2)
        return self.lm_head(hidden_states)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: KwargsForCausalLM,
    ) -> CausalLMOutputWithPast:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        model_input_ids = input_ids
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
            if self.use_slot_input_embeddings:
                slot_ids_for_inputs = self._compute_slot_ids(
                    attention_mask=attention_mask,
                    token_length=input_ids.size(1),
                    device=input_ids.device,
                )
                inputs_embeds = inputs_embeds + self.slot_embeddings(slot_ids_for_inputs)
            model_input_ids = None

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=model_input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slot_ids, _ = self._compute_slot_and_item_ids(
            attention_mask=attention_mask,
            token_length=hidden_states.size(1),
            device=hidden_states.device,
        )
        if self.use_slot_prelogits_embeddings:
            hidden_states = hidden_states + self.slot_embeddings(slot_ids)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        sliced_hidden_states = hidden_states[:, slice_indices, :]
        sliced_slot_ids = slot_ids[:, slice_indices]
        logits = self._compute_logits(hidden_states=sliced_hidden_states, slot_ids=sliced_slot_ids)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            token_loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view_as(shift_labels)
            valid_mask = shift_labels.ne(-100)
            normalizer = valid_mask.sum().clamp(min=1)
            loss = (token_loss * valid_mask).sum() / normalizer

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        return output


__all__ = ["LlamaRecSlotForCausalLM"]
