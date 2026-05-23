# coding=utf-8
from __future__ import annotations

import warnings
from typing import Optional, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import __HEAD_MASK_WARNING_MSG, T5ForConditionalGeneration


class T5RecSlotForConditionalGeneration(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.item_token_size = int(getattr(config, "item_token_size", 4))
        self.use_decoder_slot_input_embeddings = bool(
            getattr(config, "use_decoder_slot_input_embeddings", True)
        )
        self.use_decoder_slot_prelogits_embeddings = bool(
            getattr(config, "use_decoder_slot_prelogits_embeddings", True)
        )
        self.decoder_slot_embeddings = nn.Embedding(self.item_token_size, config.d_model)
        self._init_weights(self.decoder_slot_embeddings)

    def _compute_slot_ids(
        self,
        attention_mask: Optional[torch.Tensor],
        token_length: int,
        device: torch.device,
    ) -> torch.LongTensor:
        if attention_mask is None:
            base = torch.arange(token_length, device=device, dtype=torch.long)
            return base.unsqueeze(0) % self.item_token_size

        if attention_mask.dim() != 2:
            raise ValueError("T5 slot-aware decoder expects a 2D decoder attention mask")

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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[tuple[tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)
                decoder_head_mask = head_mask

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self._shift_right(labels)

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        if decoder_attention_mask is None:
            if labels is not None:
                label_mask = labels.ne(-100).to(dtype=torch.long)
                decoder_attention_mask = torch.cat(
                    [
                        torch.ones(
                            (label_mask.size(0), 1),
                            dtype=label_mask.dtype,
                            device=label_mask.device,
                        ),
                        label_mask[:, :-1],
                    ],
                    dim=1,
                )
            elif decoder_input_ids is not None:
                decoder_attention_mask = torch.ones_like(decoder_input_ids, dtype=torch.long)
            elif decoder_inputs_embeds is not None:
                decoder_attention_mask = torch.ones(
                    decoder_inputs_embeds.size(0),
                    decoder_inputs_embeds.size(1),
                    dtype=torch.long,
                    device=decoder_inputs_embeds.device,
                )

        if decoder_inputs_embeds is None and decoder_input_ids is not None:
            decoder_inputs_embeds = self.decoder.embed_tokens(decoder_input_ids)
            if self.use_decoder_slot_input_embeddings:
                slot_ids = self._compute_slot_ids(
                    attention_mask=decoder_attention_mask,
                    token_length=decoder_input_ids.size(1),
                    device=decoder_input_ids.device,
                )
                decoder_inputs_embeds = decoder_inputs_embeds + self.decoder_slot_embeddings(slot_ids)
            decoder_input_ids = None

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        sequence_output = decoder_outputs[0]

        if self.use_decoder_slot_prelogits_embeddings:
            slot_ids = self._compute_slot_ids(
                attention_mask=decoder_attention_mask,
                token_length=sequence_output.size(1),
                device=sequence_output.device,
            )
            sequence_output = sequence_output + self.decoder_slot_embeddings(slot_ids)

        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


__all__ = ["T5RecSlotForConditionalGeneration"]
