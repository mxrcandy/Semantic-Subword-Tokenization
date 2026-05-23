import logging
import json
import os
import random
import re
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import LogitsProcessor, LogitsProcessorList, PreTrainedTokenizerBase

from util.eval import compute_hr_at_k, compute_ndcg_at_k

logging.basicConfig(level=logging.INFO)

SEMANTIC_TOKEN_PATTERN = re.compile(r"<(?:[a-z]_\d+|p_[^>]+)>")
GENERATION_CONSTRAINT_ALIASES = {
    "abcd": "level_only",
    "slot": "level_only",
    "level": "level_only",
    "level_only": "level_only",
    "trie": "full_trie",
    "full_trie": "full_trie",
}


def build_item_token_codebooks_dynamically(
    tokenizer: PreTrainedTokenizerBase,
    codebook_num: int,
) -> List[List[int]]:
    prefix_list = []
    start_char_code = ord("a")

    for i in range(codebook_num):
        char = chr(start_char_code + i)
        prefix_list.append(f"<{char}_")

    vocab: Dict[str, int] = tokenizer.get_vocab()
    item_token_codebooks: List[List[int]] = [[] for _ in range(codebook_num)]

    for token, token_id in vocab.items():
        for i, prefix in enumerate(prefix_list):
            if token.startswith(prefix):
                item_token_codebooks[i].append(token_id)
                break

    for codebook in item_token_codebooks:
        codebook.sort()

    return item_token_codebooks


class SlotConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, prompt_length: int, item_token_codebooks: List[List[int]], device: torch.device):
        self.prompt_length = prompt_length
        self.codebook_len = len(item_token_codebooks)
        self.allowed_tokens_tensors = [
            torch.tensor(ids, device=device, dtype=torch.long)
            for ids in item_token_codebooks
        ]
        self.neg_inf = float("-inf")

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_step = input_ids.shape[1] - self.prompt_length
        if current_step < 0 or current_step >= self.codebook_len:
            return scores

        allowed_ids = self.allowed_tokens_tensors[current_step]
        mask = torch.full_like(scores, self.neg_inf)
        mask.index_fill_(1, allowed_ids, 0)
        return scores + mask


def normalize_generation_constraint(generation_constraint: str) -> str:
    normalized = GENERATION_CONSTRAINT_ALIASES.get(str(generation_constraint).lower())
    if normalized is None:
        valid_values = ", ".join(sorted(GENERATION_CONSTRAINT_ALIASES))
        raise ValueError(
            f"Unsupported generation_constraint: {generation_constraint}. "
            f"Expected one of: {valid_values}."
        )
    return normalized


def build_sid_token_trie(
    tokenizer: PreTrainedTokenizerBase,
    sid_to_pid_mapping_path: str,
    generation_length: int,
) -> Dict[int, dict]:
    sid_to_pid = load_sid_to_pid_mapping(sid_to_pid_mapping_path)
    trie: Dict[int, dict] = {}
    skipped = 0

    for sid in sid_to_pid:
        semantic_tokens = SEMANTIC_TOKEN_PATTERN.findall(sid)
        if not semantic_tokens or len(semantic_tokens) > generation_length:
            skipped += 1
            continue
        token_ids = tokenizer.convert_tokens_to_ids(semantic_tokens)
        if any(token_id is None or token_id == tokenizer.unk_token_id for token_id in token_ids):
            skipped += 1
            continue

        node = trie
        for token_id in token_ids:
            node = node.setdefault(int(token_id), {})
        if tokenizer.eos_token_id is not None:
            node.setdefault(int(tokenizer.eos_token_id), {})

    if not trie:
        raise ValueError(f"No valid SID token sequences found in {sid_to_pid_mapping_path}")
    if skipped:
        logging.warning("Skipped %d invalid SID entries while building generation trie.", skipped)
    return trie


class TrieConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        prompt_length: int,
        trie: Dict[int, dict],
        item_token_codebooks: List[List[int]],
        device: torch.device,
        eos_token_id: Optional[int] = None,
    ):
        self.prompt_length = prompt_length
        self.trie = trie
        self.codebook_len = len(item_token_codebooks)
        self.eos_token_id = eos_token_id
        self.fallback_allowed_tokens_tensors = [
            torch.tensor(ids, device=device, dtype=torch.long)
            for ids in item_token_codebooks
        ]
        self.neg_inf = float("-inf")

    def _allowed_for_prefix(self, prefix_token_ids: List[int]) -> List[int]:
        node = self.trie
        for token_id in prefix_token_ids:
            next_node = node.get(int(token_id))
            if next_node is None:
                return []
            node = next_node
        return list(node.keys())

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        current_step = input_ids.shape[1] - self.prompt_length
        if current_step < 0 or current_step >= self.codebook_len:
            return scores

        mask = torch.full_like(scores, self.neg_inf)
        generated_prefix = input_ids[:, self.prompt_length:]
        fallback_ids = self.fallback_allowed_tokens_tensors[current_step]

        for row_idx in range(input_ids.shape[0]):
            allowed_ids = self._allowed_for_prefix(
                generated_prefix[row_idx].detach().cpu().tolist(),
            )
            if allowed_ids:
                allowed_tensor = torch.tensor(allowed_ids, device=scores.device, dtype=torch.long)
            else:
                allowed_tensor = fallback_ids
            mask[row_idx, allowed_tensor] = 0
        return scores + mask


def build_vocab_lookup_array(tokenizer: PreTrainedTokenizerBase) -> np.ndarray:
    vocab = tokenizer.get_vocab()
    max_id = max(vocab.values())
    vocab_array = np.empty(max_id + 1, dtype=object)
    vocab_array[:] = ""
    for token, token_id in vocab.items():
        vocab_array[token_id] = token
    return vocab_array


def decode_token_sequences_fast(
    token_ids: torch.Tensor,
    vocab_lookup: np.ndarray,
) -> List[str]:
    token_ids_np = token_ids.detach().cpu().numpy()
    token_strs = vocab_lookup[token_ids_np]

    if token_strs.ndim == 1:
        return token_strs.tolist()

    if token_strs.shape[1] == 1:
        return token_strs[:, 0].tolist()

    return ["".join(row.tolist()) for row in token_strs]


def decode_generated_sid_sequences(
    token_ids: torch.Tensor,
    vocab_lookup: np.ndarray,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
) -> List[str]:
    token_ids_np = token_ids.detach().cpu().numpy()
    decoded = []
    stop_ids = {token_id for token_id in [eos_token_id, pad_token_id] if token_id is not None}
    for row in token_ids_np:
        pieces = []
        for token_id in row.tolist():
            if token_id in stop_ids:
                break
            pieces.append(str(vocab_lookup[token_id]))
        decoded.append("".join(pieces))
    return decoded


def load_sid_to_pid_mapping(mapping_path: str) -> Dict[str, List[Dict[str, int | str]]]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        raw_mapping = json.load(f)

    normalized: Dict[str, List[Dict[str, int | str]]] = {}
    for sid, candidates in raw_mapping.items():
        if not isinstance(candidates, list):
            continue
        normalized_candidates: List[Dict[str, int | str]] = []
        for candidate in candidates:
            if isinstance(candidate, dict) and "pid" in candidate:
                normalized_candidates.append(
                    {
                        "pid": str(candidate["pid"]),
                        "count": int(candidate.get("count", 0)),
                    }
                )
            else:
                normalized_candidates.append({"pid": str(candidate), "count": 0})
        normalized[sid] = normalized_candidates
    return normalized


def resolve_sid_to_pid_mapping_path(dataset_dir: str) -> Optional[str]:
    if not dataset_dir or not os.path.isdir(dataset_dir):
        return None

    candidates = sorted(
        os.path.join(dataset_dir, filename)
        for filename in os.listdir(dataset_dir)
        if filename.endswith(".sid2pid.json")
    )
    if candidates:
        return candidates[0]
    return None


def map_sid_sequence_to_pid_sequence(
    sid_sequences: List[List[str]],
    sid_to_pid: Dict[str, List[Dict[str, int | str]]],
    pid_selection_strategy: str,
    random_seed: int,
) -> List[List[str]]:
    rng = random.Random(random_seed)
    mapped_sequences: List[List[str]] = []
    normalized_strategy = pid_selection_strategy.lower()

    for user_predictions in sid_sequences:
        mapped_predictions: List[str] = []
        for sid in user_predictions:
            candidates = sid_to_pid.get(sid, [])
            if not candidates:
                mapped_predictions.append("__INVALID_PID__")
                continue

            if normalized_strategy == "random":
                chosen = rng.choice(candidates)
            elif normalized_strategy in {"most_popular_originally", "most_frequent"}:
                chosen = max(candidates, key=lambda entry: (int(entry.get("count", 0)), -int(entry["pid"])))
            elif normalized_strategy == "sample_by_count":
                weights = [max(int(entry.get("count", 0)), 0) for entry in candidates]
                if sum(weights) <= 0:
                    chosen = rng.choice(candidates)
                else:
                    chosen = rng.choices(candidates, weights=weights, k=1)[0]
            else:
                raise ValueError(
                    f"Unsupported pid_selection_strategy: {pid_selection_strategy}. "
                    "Expected one of: most_popular_originally, sample_by_count, random."
                )
            mapped_predictions.append(str(chosen["pid"]))
        mapped_sequences.append(mapped_predictions)

    return mapped_sequences


def run_generation_evaluation(
    model,
    eval_dataloader,
    tokenizer: PreTrainedTokenizerBase,
    generation_length: int,
    num_beams: int,
    k_values,
    item_token_codebooks,
    device: torch.device,
    metric_key_prefix: Optional[str] = None,
    evaluation_mode: str = "sid",
    sid_to_pid_mapping_path: Optional[str] = None,
    pid_selection_strategy: str = "most_popular_originally",
    pid_random_seed: int = 42,
    generation_constraint: str = "level_only",
    input_sid_pooler=None,
):
    vocab_lookup = build_vocab_lookup_array(tokenizer)
    normalized_mode = evaluation_mode.lower()
    if normalized_mode not in {"sid", "pid", "both"}:
        raise ValueError(f"Unsupported evaluation_mode: {evaluation_mode}")
    normalized_constraint = normalize_generation_constraint(generation_constraint)

    sid_to_pid = None
    if normalized_mode in {"pid", "both"}:
        if not sid_to_pid_mapping_path:
            raise ValueError("PID evaluation requires sid_to_pid_mapping_path.")
        sid_to_pid = load_sid_to_pid_mapping(sid_to_pid_mapping_path)

    sid_token_trie = None
    if normalized_constraint == "full_trie":
        if not sid_to_pid_mapping_path:
            raise ValueError("Trie-constrained generation requires sid_to_pid_mapping_path.")
        sid_token_trie = build_sid_token_trie(
            tokenizer=tokenizer,
            sid_to_pid_mapping_path=sid_to_pid_mapping_path,
            generation_length=generation_length,
        )

    total_metrics_sum = {}
    if normalized_mode in {"sid", "both"}:
        total_metrics_sum.update({f"HR@{k}": 0.0 for k in k_values})
        total_metrics_sum.update({f"NDCG@{k}": 0.0 for k in k_values})
    if normalized_mode in {"pid", "both"}:
        total_metrics_sum.update({f"pid_HR@{k}": 0.0 for k in k_values})
        total_metrics_sum.update({f"pid_NDCG@{k}": 0.0 for k in k_values})
    total_samples = 0

    with torch.no_grad():
        for batch in eval_dataloader:
            sid_ground_truth = batch["sid_ground_truth"]
            pid_ground_truth = batch.get("pid_ground_truth")

            if "history_item_ids" in batch:
                history_item_ids = batch["history_item_ids"].to(device)
                history_attention_mask = batch["history_attention_mask"].to(device)
                batch_size_actual = history_item_ids.shape[0]
                new_tokens = model.generate_item_sequences(
                    history_item_ids=history_item_ids,
                    history_attention_mask=history_attention_mask,
                    generation_length=generation_length,
                    item_token_codebooks=item_token_codebooks,
                    num_beams=num_beams,
                )
            else:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                batch_size_actual = input_ids.shape[0]
                is_encoder_decoder = bool(getattr(model.config, "is_encoder_decoder", False))
                prompt_length = 1 if is_encoder_decoder else input_ids.shape[1]
                if normalized_constraint == "full_trie":
                    logits_processor = LogitsProcessorList([
                        TrieConstrainedLogitsProcessor(
                            prompt_length=prompt_length,
                            trie=sid_token_trie or {},
                            item_token_codebooks=item_token_codebooks,
                            device=device,
                            eos_token_id=tokenizer.eos_token_id,
                        )
                    ])
                else:
                    logits_processor = LogitsProcessorList([
                        SlotConstrainedLogitsProcessor(
                            prompt_length=prompt_length,
                            item_token_codebooks=item_token_codebooks,
                            device=device,
                        )
                    ])

                generation_kwargs = dict(
                    attention_mask=attention_mask,
                    num_beams=num_beams,
                    do_sample=False,
                    num_return_sequences=num_beams,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    logits_processor=logits_processor,
                    use_cache=True,
                )
                if input_sid_pooler is not None:
                    if not is_encoder_decoder:
                        raise ValueError("input_sid_pooler is only supported for encoder-decoder generation.")
                    pooled = input_sid_pooler.pool_batch(
                        model=model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                    generation_kwargs["inputs_embeds"] = pooled["inputs_embeds"]
                    generation_kwargs["attention_mask"] = pooled["attention_mask"]
                else:
                    generation_kwargs["input_ids"] = input_ids

                if is_encoder_decoder:
                    generated_ids = model.generate(
                        **generation_kwargs,
                        max_new_tokens=generation_length,
                        decoder_start_token_id=getattr(
                            model.config,
                            "decoder_start_token_id",
                            tokenizer.pad_token_id,
                        ),
                    )
                else:
                    generated_ids = model.generate(
                        **generation_kwargs,
                        max_length=prompt_length + generation_length,
                    )
                new_tokens = generated_ids[:, prompt_length : prompt_length + generation_length]

            predicted_token_sequences = decode_generated_sid_sequences(
                token_ids=new_tokens,
                vocab_lookup=vocab_lookup,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            reshaped_token_sequences = [
                predicted_token_sequences[i : i + num_beams]
                for i in range(0, len(predicted_token_sequences), num_beams)
            ]

            if normalized_mode in {"sid", "both"}:
                batch_hr = compute_hr_at_k(reshaped_token_sequences, sid_ground_truth, k_values)
                batch_ndcg = compute_ndcg_at_k(reshaped_token_sequences, sid_ground_truth, k_values)

                for k_val in k_values:
                    total_metrics_sum[f"HR@{k_val}"] += batch_hr[f"HR@{k_val}"] * batch_size_actual
                    total_metrics_sum[f"NDCG@{k_val}"] += batch_ndcg[f"NDCG@{k_val}"] * batch_size_actual

            if normalized_mode in {"pid", "both"}:
                if pid_ground_truth is None:
                    raise ValueError("PID evaluation requires pid_ground_truth in eval dataset.")
                mapped_pid_sequences = map_sid_sequence_to_pid_sequence(
                    sid_sequences=reshaped_token_sequences,
                    sid_to_pid=sid_to_pid or {},
                    pid_selection_strategy=pid_selection_strategy,
                    random_seed=pid_random_seed,
                )
                batch_pid_hr = compute_hr_at_k(mapped_pid_sequences, pid_ground_truth, k_values)
                batch_pid_ndcg = compute_ndcg_at_k(mapped_pid_sequences, pid_ground_truth, k_values)

                for k_val in k_values:
                    total_metrics_sum[f"pid_HR@{k_val}"] += batch_pid_hr[f"HR@{k_val}"] * batch_size_actual
                    total_metrics_sum[f"pid_NDCG@{k_val}"] += batch_pid_ndcg[f"NDCG@{k_val}"] * batch_size_actual

            total_samples += batch_size_actual

    if total_samples == 0:
        metrics = {
            f"{metric_key_prefix}_{name}" if metric_key_prefix else name: 0.0
            for name in total_metrics_sum
        }
        return {name: round(float(value), 4) for name, value in metrics.items()}

    metrics = {
        f"{metric_key_prefix}_{name}" if metric_key_prefix else name: (value / total_samples)
        for name, value in total_metrics_sum.items()
    }
    return {name: round(float(value), 4) for name, value in metrics.items()}
