import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase


ATOM_TOKEN_RE = re.compile(r"^<([a-z])_\d+>$")


class InputSIDPooler:
    """Pool fixed SID input spans into one encoder-side item embedding.

    This diagnostic keeps decoder targets unchanged. It only replaces complete
    input-side SID spans such as <a_*><b_*><c_*><d_*> with one mean embedding.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        item_token_size: int = 4,
        mode: str = "mean",
    ) -> None:
        if mode != "mean":
            raise ValueError(f"Unsupported input SID pooling mode: {mode}. Only 'mean' is supported.")
        if item_token_size <= 0 or item_token_size > 26:
            raise ValueError(f"Invalid item_token_size={item_token_size}; expected 1..26.")
        self.tokenizer = tokenizer
        self.item_token_size = int(item_token_size)
        self.mode = mode
        self.expected_slots = [chr(ord("a") + idx) for idx in range(self.item_token_size)]
        self.slot_to_index = {slot: idx for idx, slot in enumerate(self.expected_slots)}
        self._slot_lookup_cpu = self._build_slot_lookup_tensor(tokenizer)
        self._slot_lookup_by_device: Dict[torch.device, torch.Tensor] = {}

    def _slot_of_token(self, token: str) -> Optional[str]:
        match = ATOM_TOKEN_RE.match(token)
        if match is None:
            return None
        slot = match.group(1)
        return slot if slot in self.expected_slots else None

    def _build_slot_lookup_tensor(self, tokenizer: PreTrainedTokenizerBase) -> torch.Tensor:
        vocab_size = len(tokenizer)
        lookup = torch.full((vocab_size,), -1, dtype=torch.long)
        vocab = tokenizer.get_vocab()
        for token, token_id in vocab.items():
            slot = self._slot_of_token(token)
            if slot is not None and 0 <= token_id < vocab_size:
                lookup[token_id] = self.slot_to_index[slot]
        return lookup

    def _slot_lookup_for_device(self, device: torch.device) -> torch.Tensor:
        lookup = self._slot_lookup_by_device.get(device)
        if lookup is None:
            lookup = self._slot_lookup_cpu.to(device=device, non_blocking=True)
            self._slot_lookup_by_device[device] = lookup
        return lookup

    def _segments_for_slot_ids(self, slot_ids: List[int]) -> List[Tuple[int, int]]:
        segments: List[Tuple[int, int]] = []
        start: Optional[int] = None
        slots_seen = 0
        pos = 0
        while pos < len(slot_ids):
            slot = slot_ids[pos]
            if slot < 0:
                if start is not None:
                    segments.extend((idx, idx + 1) for idx in range(start, pos))
                    start = None
                    slots_seen = 0
                segments.append((pos, pos + 1))
                pos += 1
                continue

            if start is None:
                if slot == 0:
                    start = pos
                    slots_seen = 1
                else:
                    segments.append((pos, pos + 1))
                pos += 1
                continue

            expected_next = slots_seen if slots_seen < self.item_token_size else None
            if slot == expected_next:
                slots_seen += 1
                pos += 1
                if slots_seen == self.item_token_size:
                    segments.append((start, pos))
                    start = None
                    slots_seen = 0
                continue

            segments.extend((idx, idx + 1) for idx in range(start, pos))
            start = None
            slots_seen = 0
            continue

        if start is not None:
            segments.extend((idx, idx + 1) for idx in range(start, len(slot_ids)))
        return segments

    def pool_batch(
        self,
        *,
        model: torch.nn.Module,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor],
        segments: Optional[torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        embedding_layer = model.get_input_embeddings()
        token_embeds = embedding_layer(input_ids)
        fast_pooled = self._try_pool_fixed_contiguous(
            token_embeds=token_embeds,
            attention_mask=attention_mask,
            segments=segments,
        )
        if fast_pooled is not None:
            return fast_pooled

        slot_lookup = None if segments is not None else self._slot_lookup_for_device(input_ids.device)
        pooled_rows: List[torch.Tensor] = []
        pooled_masks: List[torch.Tensor] = []
        max_len = 0

        if attention_mask is None:
            valid_lengths = torch.full(
                (input_ids.size(0),),
                input_ids.size(1),
                dtype=torch.long,
                device=input_ids.device,
            )
        else:
            valid_lengths = attention_mask.to(dtype=torch.long).sum(dim=1)

        for row_idx in range(input_ids.size(0)):
            valid_len = int(valid_lengths[row_idx].item())
            if segments is not None:
                row_segments = [
                    (int(start), int(end))
                    for start, end in segments[row_idx].detach().cpu().tolist()
                    if int(start) >= 0 and int(end) > int(start) and int(end) <= valid_len
                ]
            else:
                assert slot_lookup is not None
                slot_ids = slot_lookup[input_ids[row_idx, :valid_len]].detach().cpu().tolist()
                row_segments = self._segments_for_slot_ids(slot_ids)
            pooled = torch.stack(
                [token_embeds[row_idx, start:end].mean(dim=0) for start, end in row_segments],
                dim=0,
            )
            mask = torch.ones(pooled.size(0), dtype=torch.long, device=input_ids.device)
            pooled_rows.append(pooled)
            pooled_masks.append(mask)
            max_len = max(max_len, pooled.size(0))

        batch_size, _, hidden_size = token_embeds.shape
        inputs_embeds = token_embeds.new_zeros((batch_size, max_len, hidden_size))
        new_attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=input_ids.device)
        for row_idx, pooled in enumerate(pooled_rows):
            length = pooled.size(0)
            inputs_embeds[row_idx, :length] = pooled
            new_attention_mask[row_idx, :length] = pooled_masks[row_idx]

        return {
            "inputs_embeds": inputs_embeds,
            "attention_mask": new_attention_mask,
        }

    def _try_pool_fixed_contiguous(
        self,
        *,
        token_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        segments: Optional[torch.LongTensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Fast path for fixed SID histories laid out as contiguous 4-token items."""
        batch_size, seq_len, hidden_size = token_embeds.shape
        if attention_mask is None:
            valid_lengths = torch.full(
                (batch_size,),
                seq_len,
                dtype=torch.long,
                device=token_embeds.device,
            )
        else:
            valid_lengths = attention_mask.to(dtype=torch.long).sum(dim=1)

        if torch.any(valid_lengths % self.item_token_size != 0):
            return None

        item_counts = valid_lengths // self.item_token_size
        max_items = int(item_counts.max().item()) if item_counts.numel() else 0
        if max_items <= 0:
            return None

        usable_len = max_items * self.item_token_size
        if usable_len > seq_len:
            return None

        if segments is not None:
            segment_counts = (segments[..., 0] >= 0).sum(dim=1)
            if not torch.equal(segment_counts.to(device=item_counts.device), item_counts):
                return None
            max_segments = segments.size(1)
            expected_starts = (
                torch.arange(max_segments, device=segments.device, dtype=segments.dtype)
                .unsqueeze(0)
                .expand(batch_size, -1)
                * self.item_token_size
            )
            expected_ends = expected_starts + self.item_token_size
            valid_segment_mask = segments[..., 0] >= 0
            if torch.any(segments[..., 0][valid_segment_mask] != expected_starts[valid_segment_mask]):
                return None
            if torch.any(segments[..., 1][valid_segment_mask] != expected_ends[valid_segment_mask]):
                return None

        contiguous = token_embeds[:, :usable_len, :]
        pooled = contiguous.reshape(
            batch_size,
            max_items,
            self.item_token_size,
            hidden_size,
        ).mean(dim=2)
        new_attention_mask = (
            torch.arange(max_items, device=token_embeds.device).unsqueeze(0)
            < item_counts.unsqueeze(1)
        ).to(dtype=torch.long)
        return {
            "inputs_embeds": pooled,
            "attention_mask": new_attention_mask,
        }


def build_input_sid_pooler(
    model_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> Optional[InputSIDPooler]:
    config = model_config.get("input_sid_pooling")
    if not config:
        logging.info("Input SID pooling disabled.")
        return None

    if isinstance(config, bool):
        enabled = config
        config_dict: Dict[str, Any] = {}
    elif isinstance(config, dict):
        enabled = bool(config.get("enabled", False))
        config_dict = config
    else:
        raise ValueError(f"model.input_sid_pooling must be a bool or dict, got {type(config)}")

    if not enabled:
        logging.info("Input SID pooling disabled.")
        return None

    pooler = InputSIDPooler(
        tokenizer=tokenizer,
        item_token_size=int(config_dict.get("item_token_size", 4)),
        mode=str(config_dict.get("mode", "mean")),
    )
    logging.info(
        "Input SID pooling enabled: mode=%s item_token_size=%s",
        pooler.mode,
        pooler.item_token_size,
    )
    return pooler
