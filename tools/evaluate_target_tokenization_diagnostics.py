#!/usr/bin/env python3
"""Evaluate target-side tokenization failure modes.

This keeps the original merged/unmerged target bucket evaluation, and adds
diagnostics for target length bias, EOS placement errors, and first-step
pattern-token errors.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from collections import Counter
from typing import Any

import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import LogitsProcessorList

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from util.datacollator import EvalDataCollator, preprocess_eval_dataset
from util.generation_evaluate import (
    TrieConstrainedLogitsProcessor,
    SlotConstrainedLogitsProcessor,
    build_item_token_codebooks_dynamically,
    build_sid_token_trie,
    build_vocab_lookup_array,
    decode_generated_sid_sequences,
    load_sid_to_pid_mapping,
    map_sid_sequence_to_pid_sequence,
    normalize_generation_constraint,
    resolve_sid_to_pid_mapping_path,
)
from util.eval import compute_hr_at_k, compute_ndcg_at_k
from util.runtime import (
    load_experiment_config,
    load_model_from_checkpoint,
    load_tokenizer_from_checkpoint,
    resolve_data_split_path,
)


RUNS = [
    {
        "setting": "base fixed SID",
        "dataset": "Yelp_rq_kmeans_d_reassign_seq2seq",
        "model": "t5-rec",
        "checkpoint": "experiment/Yelp_rq_kmeans_d_reassign_seq2seq/t5-rec_20260507_012840/best_model",
    },
    {
        "setting": "varlen history + fixed target",
        "dataset": "Yelp_rq_kmeans_d_reassign_varlen_bpe_strict_top128_f20_s10_seq2seq",
        "model": "t5-rec",
        "checkpoint": "experiment/Yelp_rq_kmeans_d_reassign_varlen_bpe_strict_top128_f20_s10_seq2seq/t5-rec_20260508_080743/best_model",
    },
    {
        "setting": "varlen history + varlen target",
        "dataset": "Yelp_rq_kmeans_d_reassign_varlen_bpe_strict_top128_f20_s10_var_target_seq2seq",
        "model": "t5-rec",
        "checkpoint": "experiment/Yelp_rq_kmeans_d_reassign_varlen_bpe_strict_top128_f20_s10_var_target_seq2seq/t5-rec_20260514_121331/best_model",
    },
    {
        "setting": "fixed RQ-KMeans",
        "dataset": "Beauty_rq_kmeans_d_reassign",
        "model": "llama-rec",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign/llama-rec_20260427_081036/best_model",
    },
    {
        "setting": "fixed RQ-KMeans",
        "dataset": "Beauty_rq_kmeans_d_reassign",
        "model": "mistral",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign/mistral_20260427_082614/best_model",
    },
    {
        "setting": "fixed RQ-KMeans",
        "dataset": "Beauty_rq_kmeans_d_reassign",
        "model": "qwen3",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign/qwen3_20260427_081956/best_model",
    },
    {
        "setting": "prefix usage10",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10",
        "model": "llama-rec",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10/llama-rec_20260428_140808/best_model",
    },
    {
        "setting": "prefix usage10",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10",
        "model": "mistral",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10/mistral_20260428_141315/best_model",
    },
    {
        "setting": "prefix usage10",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10",
        "model": "qwen3",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10/qwen3_20260428_141833/best_model",
    },
    {
        "setting": "prefix usage10 + sum init",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10",
        "model": "llama-rec",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10/llama-rec_20260429_062450/best_model",
    },
    {
        "setting": "prefix usage10 + sum init",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10",
        "model": "mistral",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10/mistral_20260429_063017/best_model",
    },
    {
        "setting": "prefix usage10 + sum init",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10",
        "model": "qwen3",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10/qwen3_20260429_063606/best_model",
    },
    {
        "setting": "prefix usage10 + sum init + freeze",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10",
        "model": "llama-rec",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10/llama-rec_20260429_065603/best_model",
    },
    {
        "setting": "prefix usage10 + sum init + freeze",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10",
        "model": "mistral",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10/mistral_20260429_065603/best_model",
    },
    {
        "setting": "prefix usage10 + sum init + freeze",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10",
        "model": "qwen3",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10/qwen3_20260429_065603/best_model",
    },
    {
        "setting": "dual default",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10_dual",
        "model": "llama-rec",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10_dual/llama-rec_20260429_123410/best_model",
    },
    {
        "setting": "dual default",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10_dual",
        "model": "mistral",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10_dual/mistral_20260429_123415/best_model",
    },
    {
        "setting": "dual default",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10_dual",
        "model": "qwen3",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10_dual/qwen3_20260429_123415/best_model",
    },
    {
        "setting": "history varlen + fixed target",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10_history_varlen_target_fixed",
        "model": "llama-rec",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10_history_varlen_target_fixed/llama-rec_20260430_021230/best_model",
    },
    {
        "setting": "history varlen + fixed target",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10_history_varlen_target_fixed",
        "model": "mistral",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10_history_varlen_target_fixed/mistral_20260430_021230/best_model",
    },
    {
        "setting": "history varlen + fixed target",
        "dataset": "Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10_history_varlen_target_fixed",
        "model": "qwen3",
        "checkpoint": "experiment/Beauty_rq_kmeans_d_reassign_varlen_bpe_ab_top128_usage10_history_varlen_target_fixed/qwen3_20260430_021230/best_model",
    },
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def tokenized_dataset_from_rows(rows: list[dict[str, Any]], tokenizer, max_length: int, num_proc: int | None):
    if not rows:
        return None
    dataset = Dataset.from_list(rows)
    return dataset.map(
        preprocess_eval_dataset,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=[col for col in dataset.column_names if col not in {"sid_ground_truth", "pid_ground_truth"}],
        desc="Tokenizing bucket dataset",
    )


def pid_is_merged(row: dict[str, Any], merged_pids: set[str]) -> bool:
    pid_ground_truth = row.get("pid_ground_truth", [])
    if isinstance(pid_ground_truth, list) and pid_ground_truth:
        return str(pid_ground_truth[0]) in merged_pids
    return str(pid_ground_truth) in merged_pids


def pattern_span_label(pattern_token: str) -> str:
    import re

    slots = re.findall(r"([abcd])\d+", pattern_token)
    return "".join(slot.upper() for slot in slots) if slots else "OTHER"


def pid_pattern_bucket(row: dict[str, Any], merged_items: dict[str, Any]) -> str:
    pid = row_target_pid(row)
    info = merged_items.get(str(pid))
    if not info:
        return "unmerged"
    patterns = info.get("applied_patterns", []) if isinstance(info, dict) else []
    labels = [pattern_span_label(str(pattern)) for pattern in patterns]
    labels = [label for label in labels if label and label != "OTHER"]
    if not labels:
        return "merged_unknown"
    unique_labels = sorted(set(labels))
    if len(unique_labels) == 1:
        return unique_labels[0]
    return "+".join(unique_labels)


def sid_tokens(sid: str) -> list[str]:
    import re

    return re.findall(r"<(?:[a-z]_\d+|p_[^>]+)>", sid)


def row_target_sid(row: dict[str, Any]) -> str:
    ground_truth = row.get("sid_ground_truth", [])
    if isinstance(ground_truth, list) and ground_truth:
        return str(ground_truth[0])
    return str(ground_truth)


def row_target_pid(row: dict[str, Any]) -> str:
    ground_truth = row.get("pid_ground_truth", [])
    if isinstance(ground_truth, list) and ground_truth:
        return str(ground_truth[0])
    return str(ground_truth)


def target_length(row: dict[str, Any]) -> int:
    return len(sid_tokens(row_target_sid(row)))


def is_pattern_token(token: str | None) -> bool:
    return bool(token and token.startswith("<p_"))


def is_atom_token(token: str | None) -> bool:
    import re

    return bool(token and re.fullmatch(r"<[a-z]_\d+>", token))


def round_float(value: float) -> float:
    return round(float(value), 4)


def summarize_counts(counts: dict[str, int]) -> dict[str, Any]:
    total = sum(counts.values())
    payload: dict[str, Any] = {"total": total}
    for name, count in sorted(counts.items()):
        payload[name] = count
        payload[f"{name}_rate"] = round_float(count / total) if total else 0.0
    return payload


def compute_subset_metrics(
    sid_predictions: list[list[str]],
    sid_ground_truth: list[list[str]],
    pid_predictions: list[list[str]] | None,
    pid_ground_truth: list[list[str]] | None,
    k_values: list[int],
    prefix: str | None = None,
) -> dict[str, float]:
    if not sid_ground_truth:
        names = [f"HR@{k}" for k in k_values] + [f"NDCG@{k}" for k in k_values]
        if pid_predictions is not None:
            names += [f"pid_HR@{k}" for k in k_values] + [f"pid_NDCG@{k}" for k in k_values]
        return {f"{prefix}_{name}" if prefix else name: 0.0 for name in names}

    metrics: dict[str, float] = {}
    metrics.update(compute_hr_at_k(sid_predictions, sid_ground_truth, k_values))
    metrics.update(compute_ndcg_at_k(sid_predictions, sid_ground_truth, k_values))
    if pid_predictions is not None and pid_ground_truth is not None:
        pid_hr = compute_hr_at_k(pid_predictions, pid_ground_truth, k_values)
        pid_ndcg = compute_ndcg_at_k(pid_predictions, pid_ground_truth, k_values)
        metrics.update({f"pid_{name}": value for name, value in pid_hr.items()})
        metrics.update({f"pid_{name}": value for name, value in pid_ndcg.items()})
    if prefix:
        metrics = {f"{prefix}_{name}": value for name, value in metrics.items()}
    return {name: round_float(value) for name, value in metrics.items()}


def gather_predictions(
    model,
    dataloader: DataLoader,
    tokenizer,
    generation_length: int,
    num_beams: int,
    item_token_codebooks,
    device: torch.device,
    generation_constraint: str,
    sid_to_pid_mapping_path: str | None,
) -> tuple[list[list[str]], list[list[list[str]]]]:
    vocab_lookup = build_vocab_lookup_array(tokenizer)
    normalized_constraint = normalize_generation_constraint(generation_constraint)
    sid_token_trie = None
    if normalized_constraint == "full_trie":
        if not sid_to_pid_mapping_path:
            raise ValueError("Trie-constrained diagnostics require sid_to_pid_mapping_path.")
        sid_token_trie = build_sid_token_trie(
            tokenizer=tokenizer,
            sid_to_pid_mapping_path=sid_to_pid_mapping_path,
            generation_length=generation_length,
        )

    sid_predictions: list[list[str]] = []
    raw_token_beams: list[list[list[str]]] = []

    with torch.no_grad():
        for batch in dataloader:
            if "history_item_ids" in batch:
                history_item_ids = batch["history_item_ids"].to(device)
                history_attention_mask = batch["history_attention_mask"].to(device)
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
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    num_beams=num_beams,
                    do_sample=False,
                    num_return_sequences=num_beams,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    logits_processor=logits_processor,
                    use_cache=True,
                )
                if is_encoder_decoder:
                    generated_ids = model.generate(
                        **generation_kwargs,
                        max_new_tokens=generation_length,
                        decoder_start_token_id=getattr(model.config, "decoder_start_token_id", tokenizer.pad_token_id),
                    )
                else:
                    generated_ids = model.generate(**generation_kwargs, max_length=prompt_length + generation_length)
                new_tokens = generated_ids[:, prompt_length : prompt_length + generation_length]

            decoded_sid = decode_generated_sid_sequences(
                token_ids=new_tokens,
                vocab_lookup=vocab_lookup,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
            reshaped_sid = [decoded_sid[i : i + num_beams] for i in range(0, len(decoded_sid), num_beams)]
            sid_predictions.extend(reshaped_sid)

            token_ids_np = new_tokens.detach().cpu().numpy()
            token_strs = vocab_lookup[token_ids_np]
            if token_strs.ndim == 1:
                token_str_rows = [[str(token)] for token in token_strs.tolist()]
            else:
                token_str_rows = [[str(token) for token in row.tolist()] for row in token_strs]
            reshaped_raw = [token_str_rows[i : i + num_beams] for i in range(0, len(token_str_rows), num_beams)]
            raw_token_beams.extend(reshaped_raw)

    return sid_predictions, raw_token_beams


def subset_by_indices(values: list[Any], indices: list[int]) -> list[Any]:
    return [values[index] for index in indices]


def target_length_bias_diagnostics(
    rows: list[dict[str, Any]],
    sid_predictions: list[list[str]],
    pid_predictions: list[list[str]] | None,
    k_values: list[int],
) -> dict[str, Any]:
    sid_ground_truth = [row.get("sid_ground_truth", []) for row in rows]
    pid_ground_truth = [row.get("pid_ground_truth", []) for row in rows]
    by_length: dict[int, list[int]] = {}
    for index, row in enumerate(rows):
        by_length.setdefault(target_length(row), []).append(index)

    payload: dict[str, Any] = {}
    for length, indices in sorted(by_length.items()):
        payload[str(length)] = {
            "num_examples": len(indices),
            "target_first_pattern_rate": round_float(
                sum(is_pattern_token(sid_tokens(row_target_sid(rows[index]))[0] if sid_tokens(row_target_sid(rows[index])) else None) for index in indices)
                / len(indices)
            ),
            "metrics": compute_subset_metrics(
                subset_by_indices(sid_predictions, indices),
                subset_by_indices(sid_ground_truth, indices),
                subset_by_indices(pid_predictions, indices) if pid_predictions is not None else None,
                subset_by_indices(pid_ground_truth, indices) if pid_predictions is not None else None,
                k_values,
            ),
        }
    return payload


def eos_error_diagnostics(
    rows: list[dict[str, Any]],
    raw_token_beams: list[list[list[str]]],
    generation_length: int,
    eos_token: str | None,
    pad_token: str | None,
) -> dict[str, Any]:
    counts = {
        "correct_boundary": 0,
        "early_eos": 0,
        "late_eos": 0,
        "missing_eos": 0,
        "unexpected_eos": 0,
    }
    by_target_length: dict[int, dict[str, int]] = {}
    stop_tokens = {token for token in [eos_token, pad_token] if token is not None}

    for row, pred_beams in zip(rows, raw_token_beams):
        pred_tokens = pred_beams[0] if pred_beams else []
        length = target_length(row)
        expected_eos = length < generation_length
        eos_pos = None
        for index, token in enumerate(pred_tokens):
            if token in stop_tokens:
                eos_pos = index
                break

        if expected_eos:
            if eos_pos == length:
                category = "correct_boundary"
            elif eos_pos is None:
                category = "missing_eos"
            elif eos_pos < length:
                category = "early_eos"
            else:
                category = "late_eos"
        else:
            category = "unexpected_eos" if eos_pos is not None else "correct_boundary"

        counts[category] += 1
        bucket = by_target_length.setdefault(length, {name: 0 for name in counts})
        bucket[category] += 1

    return {
        "overall": summarize_counts(counts),
        "by_target_length": {str(length): summarize_counts(bucket) for length, bucket in sorted(by_target_length.items())},
    }


def first_step_pattern_diagnostics(
    rows: list[dict[str, Any]],
    raw_token_beams: list[list[list[str]]],
) -> dict[str, Any]:
    groups = {
        "pattern_first_targets": [],
        "atom_first_targets": [],
    }
    examples = []
    beam_summary = {
        "pattern_first_targets": {
            "target_pattern_in_any_beam": 0,
            "any_pattern_in_any_beam": 0,
            "target_pattern_best_rank_sum": 0,
            "target_pattern_best_rank_count": 0,
            "first_token_counter": Counter(),
            "top1_first_token_counter": Counter(),
        },
        "atom_first_targets": {
            "any_pattern_in_any_beam": 0,
            "first_token_counter": Counter(),
            "top1_first_token_counter": Counter(),
        },
    }
    for index, (row, pred_beams) in enumerate(zip(rows, raw_token_beams)):
        target_tokens = sid_tokens(row_target_sid(row))
        if not target_tokens:
            continue
        target_first = target_tokens[0]
        pred_tokens = pred_beams[0] if pred_beams else []
        pred_first = pred_tokens[0] if pred_tokens else None
        beam_first_tokens = [beam[0] for beam in pred_beams if beam]
        entry = {
            "index": index,
            "pid": row_target_pid(row),
            "target_first": target_first,
            "pred_first": pred_first,
            "first_token_match": pred_first == target_first,
            "pred_first_is_pattern": is_pattern_token(pred_first),
            "pred_first_is_atom": is_atom_token(pred_first),
        }
        if is_pattern_token(target_first):
            groups["pattern_first_targets"].append(entry)
            summary = beam_summary["pattern_first_targets"]
            summary["first_token_counter"].update(beam_first_tokens)
            if pred_first is not None:
                summary["top1_first_token_counter"].update([pred_first])
            if any(is_pattern_token(token) for token in beam_first_tokens):
                summary["any_pattern_in_any_beam"] += 1
            if target_first in beam_first_tokens:
                summary["target_pattern_in_any_beam"] += 1
                summary["target_pattern_best_rank_sum"] += beam_first_tokens.index(target_first) + 1
                summary["target_pattern_best_rank_count"] += 1
            if pred_first != target_first and len(examples) < 20:
                examples.append(
                    {
                        "index": index,
                        "pid": row_target_pid(row),
                        "target_sid": row_target_sid(row),
                        "pred_top1_tokens": pred_tokens,
                        "beam_first_tokens": beam_first_tokens,
                    }
                )
        else:
            groups["atom_first_targets"].append(entry)
            summary = beam_summary["atom_first_targets"]
            summary["first_token_counter"].update(beam_first_tokens)
            if pred_first is not None:
                summary["top1_first_token_counter"].update([pred_first])
            if any(is_pattern_token(token) for token in beam_first_tokens):
                summary["any_pattern_in_any_beam"] += 1

    payload: dict[str, Any] = {}
    for name, entries in groups.items():
        total = len(entries)
        correct = sum(entry["first_token_match"] for entry in entries)
        pred_pattern = sum(entry["pred_first_is_pattern"] for entry in entries)
        pred_atom = sum(entry["pred_first_is_atom"] for entry in entries)
        payload[name] = {
            "num_examples": total,
            "first_token_accuracy": round_float(correct / total) if total else 0.0,
            "pred_pattern_rate": round_float(pred_pattern / total) if total else 0.0,
            "pred_atom_rate": round_float(pred_atom / total) if total else 0.0,
        }
        summary = beam_summary[name]
        payload[name]["any_pattern_in_any_beam_rate"] = (
            round_float(summary["any_pattern_in_any_beam"] / total) if total else 0.0
        )
        payload[name]["top_beam_first_tokens"] = summary["first_token_counter"].most_common(20)
        payload[name]["top1_first_tokens"] = summary["top1_first_token_counter"].most_common(20)
        if name == "pattern_first_targets":
            count = summary["target_pattern_best_rank_count"]
            payload[name]["target_pattern_in_any_beam_rate"] = (
                round_float(summary["target_pattern_in_any_beam"] / total) if total else 0.0
            )
            payload[name]["target_pattern_avg_best_rank"] = (
                round_float(summary["target_pattern_best_rank_sum"] / count) if count else None
            )
    payload["pattern_first_error_examples"] = examples
    return payload


def position_pattern_beam_diagnostics(
    rows: list[dict[str, Any]],
    raw_token_beams: list[list[list[str]]],
    positions: list[dict[str, str | int]],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for spec in positions:
        pos = int(spec["position"])
        pattern_prefix = str(spec["pattern_prefix"])
        atom_prefix = str(spec["atom_prefix"])
        label = str(spec["label"])
        total = 0
        top1_correct = 0
        top1_pattern = 0
        top1_atom = 0
        any_pattern = 0
        target_pattern = 0
        target_rank_sum = 0
        target_rank_count = 0
        top1_counter: Counter[str] = Counter()
        beam_counter: Counter[str] = Counter()

        for row, pred_beams in zip(rows, raw_token_beams):
            target_tokens = sid_tokens(row_target_sid(row))
            if len(target_tokens) <= pos:
                continue
            target = target_tokens[pos]
            if not target.startswith(pattern_prefix):
                continue
            position_tokens = [beam[pos] for beam in pred_beams if len(beam) > pos]
            if not position_tokens:
                continue
            total += 1
            top1 = position_tokens[0]
            top1_counter.update([top1])
            beam_counter.update(position_tokens)
            top1_correct += int(top1 == target)
            top1_pattern += int(is_pattern_token(top1))
            top1_atom += int(is_atom_token(top1))
            any_pattern += int(any(is_pattern_token(token) for token in position_tokens))
            if target in position_tokens:
                target_pattern += 1
                target_rank_sum += position_tokens.index(target) + 1
                target_rank_count += 1

        payload[label] = {
            "position": pos + 1,
            "pattern_prefix": pattern_prefix,
            "atom_prefix": atom_prefix,
            "num_examples": total,
            "top1_token_accuracy": round_float(top1_correct / total) if total else 0.0,
            "top1_pattern_rate": round_float(top1_pattern / total) if total else 0.0,
            "top1_atom_rate": round_float(top1_atom / total) if total else 0.0,
            "any_pattern_in_top20_rate": round_float(any_pattern / total) if total else 0.0,
            "target_pattern_in_top20_rate": round_float(target_pattern / total) if total else 0.0,
            "target_pattern_avg_best_rank": round_float(target_rank_sum / target_rank_count)
            if target_rank_count
            else None,
            "top1_tokens": top1_counter.most_common(20),
            "beam_position_tokens": beam_counter.most_common(20),
        }
    return payload


def first_step_logit_bias_diagnostics(
    rows: list[dict[str, Any]],
    model,
    dataloader: DataLoader,
    tokenizer,
    device: torch.device,
    max_examples: int,
) -> dict[str, Any]:
    if max_examples <= 0:
        return {}

    vocab = tokenizer.get_vocab()
    pattern_token_ids = {
        token_id
        for token, token_id in vocab.items()
        if is_pattern_token(token)
    }
    atom_a_token_ids = {
        token_id
        for token, token_id in vocab.items()
        if token.startswith("<a_")
    }
    pattern_rows = [
        index
        for index, row in enumerate(rows)
        if (tokens := sid_tokens(row_target_sid(row))) and is_pattern_token(tokens[0])
    ]
    selected = set(pattern_rows[:max_examples])
    if not selected:
        return {
            "num_examples": 0,
            "note": "No pattern-first targets found.",
        }

    token_id_to_token = {token_id: token for token, token_id in vocab.items()}
    ranks = []
    target_logits = []
    top1_tokens = Counter()
    top10_token_counter = Counter()
    atom_a_top1 = 0
    pattern_top1 = 0
    target_top10 = 0
    processed = 0
    global_index = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_size = input_ids.shape[0]
            wanted_local = [
                local_idx
                for local_idx in range(batch_size)
                if global_index + local_idx in selected
            ]
            if not wanted_local:
                global_index += batch_size
                continue

            is_encoder_decoder = bool(getattr(model.config, "is_encoder_decoder", False))
            if is_encoder_decoder:
                decoder_start_token_id = getattr(model.config, "decoder_start_token_id", tokenizer.pad_token_id)
                decoder_input_ids = torch.full(
                    (batch_size, 1),
                    int(decoder_start_token_id),
                    dtype=torch.long,
                    device=device,
                )
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_input_ids,
                ).logits[:, -1, :]
            else:
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1, :]

            allowed_ids = list(pattern_token_ids | atom_a_token_ids)
            disallowed = torch.ones(logits.shape[-1], dtype=torch.bool, device=logits.device)
            disallowed[allowed_ids] = False
            logits = logits.masked_fill(disallowed.unsqueeze(0), float("-inf"))

            for local_idx in wanted_local:
                row_index = global_index + local_idx
                target_first = sid_tokens(row_target_sid(rows[row_index]))[0]
                target_id = tokenizer.convert_tokens_to_ids(target_first)
                row_logits = logits[local_idx]
                sorted_ids = torch.argsort(row_logits, descending=True)
                rank_tensor = (sorted_ids == target_id).nonzero(as_tuple=False)
                rank = int(rank_tensor[0].item()) + 1 if rank_tensor.numel() else None
                top_ids = sorted_ids[:10].detach().cpu().tolist()
                top_tokens = [token_id_to_token.get(int(token_id), str(token_id)) for token_id in top_ids]
                top1 = top_tokens[0] if top_tokens else None

                if rank is not None:
                    ranks.append(rank)
                target_logits.append(float(row_logits[target_id].detach().cpu()))
                top1_tokens.update([top1])
                top10_token_counter.update(top_tokens)
                atom_a_top1 += int(is_atom_token(top1))
                pattern_top1 += int(is_pattern_token(top1))
                target_top10 += int(target_first in top_tokens)
                processed += 1

            global_index += batch_size
            if processed >= max_examples:
                break

    ranks_sorted = sorted(ranks)
    def percentile(values: list[int], q: float) -> int | None:
        if not values:
            return None
        pos = min(len(values) - 1, int(round((len(values) - 1) * q)))
        return values[pos]

    return {
        "num_examples": processed,
        "target_pattern_rank_mean": round_float(sum(ranks) / len(ranks)) if ranks else None,
        "target_pattern_rank_median": percentile(ranks_sorted, 0.5),
        "target_pattern_rank_p90": percentile(ranks_sorted, 0.9),
        "target_pattern_in_top10_rate": round_float(target_top10 / processed) if processed else 0.0,
        "top1_atom_a_rate": round_float(atom_a_top1 / processed) if processed else 0.0,
        "top1_pattern_rate": round_float(pattern_top1 / processed) if processed else 0.0,
        "top1_tokens": top1_tokens.most_common(20),
        "top10_tokens": top10_token_counter.most_common(30),
    }


def evaluate_run(run: dict[str, str], merged_items: dict[str, Any], device: torch.device) -> dict[str, Any]:
    runtime_config = load_experiment_config(run["dataset"], run["model"])
    paths_config = runtime_config["paths"]
    model_config = runtime_config["model"]
    runtime_options = runtime_config["runtime"]
    training_config = runtime_config["training"]
    tokenizer_config = runtime_config["tokenizer"]
    evaluation_config = runtime_config["evaluation"]

    test_path = Path(resolve_data_split_path(paths_config, "test", fallback_to_legacy_eval=False))
    rows = load_json(test_path)
    merged_pids = set(merged_items)
    bucket_indices = {
        "merged": [index for index, row in enumerate(rows) if pid_is_merged(row, merged_pids)],
        "unmerged": [index for index, row in enumerate(rows) if not pid_is_merged(row, merged_pids)],
    }
    pattern_bucket_indices: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        bucket_name = pid_pattern_bucket(row, merged_items)
        pattern_bucket_indices.setdefault(bucket_name, []).append(index)

    checkpoint = run["checkpoint"]
    tokenizer = load_tokenizer_from_checkpoint(checkpoint)
    model = load_model_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()

    generation_length = len(tokenizer_config["codeword_nums"])
    item_token_codebooks = build_item_token_codebooks_dynamically(tokenizer, generation_length)
    sid_to_pid_mapping_path = evaluation_config.get("sid_to_pid_mapping_path")
    if sid_to_pid_mapping_path is None:
        sid_to_pid_mapping_path = resolve_sid_to_pid_mapping_path(paths_config["dataset_path"])

    result = {
        "setting": run["setting"],
        "model": run["model"],
        "dataset": run["dataset"],
        "checkpoint": checkpoint,
        "buckets": {},
        "pattern_span_buckets": {},
        "target_length_bias": {},
        "eos_errors": {},
        "first_step_pattern_errors": {},
        "position_pattern_beam_errors": {},
        "first_step_logit_bias": {},
    }

    tokenized = tokenized_dataset_from_rows(
        rows,
        tokenizer=tokenizer,
        max_length=model_config["max_seq_length"],
        num_proc=runtime_options.get("preprocess_num_proc"),
    )
    dataloader = DataLoader(
        tokenized,
        batch_size=training_config["per_device_eval_batch_size"],
        collate_fn=EvalDataCollator(tokenizer=tokenizer, max_length=model_config["max_seq_length"]),
        shuffle=False,
        drop_last=False,
    )
    sid_predictions, raw_token_beams = gather_predictions(
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        generation_length=generation_length,
        num_beams=evaluation_config["num_beams"],
        item_token_codebooks=item_token_codebooks,
        device=device,
        generation_constraint=evaluation_config.get("generation_constraint", "full_trie"),
        sid_to_pid_mapping_path=sid_to_pid_mapping_path,
    )

    sid_ground_truth = [row.get("sid_ground_truth", []) for row in rows]
    pid_ground_truth = [row.get("pid_ground_truth", []) for row in rows]
    sid_to_pid = None
    pid_predictions = None
    if evaluation_config.get("evaluation_mode", "both").lower() in {"pid", "both"}:
        if not sid_to_pid_mapping_path:
            raise ValueError("PID diagnostics require sid_to_pid_mapping_path.")
        sid_to_pid = load_sid_to_pid_mapping(sid_to_pid_mapping_path)
        pid_predictions = map_sid_sequence_to_pid_sequence(
            sid_sequences=sid_predictions,
            sid_to_pid=sid_to_pid,
            pid_selection_strategy=evaluation_config.get("pid_selection_strategy", "most_popular_originally"),
            random_seed=evaluation_config.get("pid_random_seed", 42),
        )

    k_values = evaluation_config["eval_k_values"]
    for bucket_name, indices in bucket_indices.items():
        result["buckets"][bucket_name] = {
            "num_examples": len(indices),
            "metrics": compute_subset_metrics(
                subset_by_indices(sid_predictions, indices),
                subset_by_indices(sid_ground_truth, indices),
                subset_by_indices(pid_predictions, indices) if pid_predictions is not None else None,
                subset_by_indices(pid_ground_truth, indices) if pid_predictions is not None else None,
                k_values,
                prefix=bucket_name,
            ),
        }

    for bucket_name, indices in sorted(pattern_bucket_indices.items()):
        result["pattern_span_buckets"][bucket_name] = {
            "num_examples": len(indices),
            "metrics": compute_subset_metrics(
                subset_by_indices(sid_predictions, indices),
                subset_by_indices(sid_ground_truth, indices),
                subset_by_indices(pid_predictions, indices) if pid_predictions is not None else None,
                subset_by_indices(pid_ground_truth, indices) if pid_predictions is not None else None,
                k_values,
                prefix=bucket_name,
            ),
        }

    result["target_length_bias"] = target_length_bias_diagnostics(
        rows=rows,
        sid_predictions=sid_predictions,
        pid_predictions=pid_predictions,
        k_values=k_values,
    )
    result["eos_errors"] = eos_error_diagnostics(
        rows=rows,
        raw_token_beams=raw_token_beams,
        generation_length=generation_length,
        eos_token=tokenizer.eos_token,
        pad_token=tokenizer.pad_token,
    )
    result["first_step_pattern_errors"] = first_step_pattern_diagnostics(
        rows=rows,
        raw_token_beams=raw_token_beams,
    )
    result["position_pattern_beam_errors"] = position_pattern_beam_diagnostics(
        rows=rows,
        raw_token_beams=raw_token_beams,
        positions=[
            {"label": "pos1_p_a", "position": 0, "pattern_prefix": "<p_a", "atom_prefix": "<a_"},
            {"label": "pos2_p_b", "position": 1, "pattern_prefix": "<p_b", "atom_prefix": "<b_"},
        ],
    )
    result["first_step_logit_bias"] = first_step_logit_bias_diagnostics(
        rows=rows,
        model=model,
        dataloader=dataloader,
        tokenizer=tokenizer,
        device=device,
        max_examples=run.get("max_logit_bias_examples", 512),
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="*", default=["t5-rec"])
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--settings", nargs="*", default=None)
    parser.add_argument("--max-logit-bias-examples", type=int, default=512)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tools/analysis_outputs/target_tokenization_diagnostics.json"),
    )
    parser.add_argument(
        "--varlen-meta",
        type=Path,
        default=Path("data/Yelp_rq_kmeans_d_reassign_varlen_bpe_strict_top128_f20_s10/varlen_meta.json"),
    )
    args = parser.parse_args()

    meta = load_json(args.varlen_meta)
    merged_items = {str(pid): info for pid, info in meta["merged_items"].items()}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    selected_runs = []
    settings = set(args.settings) if args.settings else None
    models = set(args.models)
    datasets = set(args.datasets) if args.datasets else None
    for run in RUNS:
        if run["model"] not in models:
            continue
        if datasets is not None and run["dataset"] not in datasets:
            continue
        if settings is not None and run["setting"] not in settings:
            continue
        if not Path(run["checkpoint"]).exists():
            continue
        selected_run = dict(run)
        selected_run["max_logit_bias_examples"] = args.max_logit_bias_examples
        selected_runs.append(selected_run)

    payload = {
        "merged_target_pid_count": len(merged_items),
        "runs": [],
    }
    for run in selected_runs:
        print(f"Evaluating {run['model']} | {run['setting']}", flush=True)
        payload["runs"].append(evaluate_run(run, merged_items=merged_items, device=device))
        dump_json(args.output, payload)

    dump_json(args.output, payload)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
