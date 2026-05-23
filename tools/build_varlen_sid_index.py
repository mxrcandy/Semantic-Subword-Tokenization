#!/usr/bin/env python3
"""Build a variable-length SID dataset with constrained span merges.

This is a post-hoc tokenizer over existing fixed-length semantic IDs. It keeps
the original quantizer untouched, selects frequent contiguous SID spans as
pattern tokens, and rewrites index/train/eval files with canonical longest-match
segmentations.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BASE_TOKEN_RE = re.compile(r"<[a-z]_\d+>")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def infer_dataset_prefix(dataset_dir: Path) -> str:
    index_files = sorted(dataset_dir.glob("*.index.json"))
    if not index_files:
        raise FileNotFoundError(f"No *.index.json found under {dataset_dir}")
    return index_files[0].name[: -len(".index.json")]


def sid_string(tokens: list[str]) -> str:
    return "".join(tokens)


def pattern_name(span: tuple[str, ...]) -> str:
    parts: list[str] = []
    for token in span:
        inner = token[1:-1]
        parts.append(inner.replace("_", ""))
    return "<p_" + "_".join(parts) + ">"


def load_item_weights(sid2pid_path: Path | None, index_data: dict[str, list[str]]) -> dict[str, float]:
    weights = {item_id: 1.0 for item_id in index_data}
    if sid2pid_path is None or not sid2pid_path.exists():
        return weights

    sid_to_item_ids = defaultdict(list)
    for item_id, tokens in index_data.items():
        sid_to_item_ids[sid_string(tokens)].append(item_id)

    sid2pid = load_json(sid2pid_path)
    for old_sid, candidates in sid2pid.items():
        total = 0
        if isinstance(candidates, list):
            for candidate in candidates:
                if isinstance(candidate, dict):
                    total += int(candidate.get("count", 0))
                else:
                    total += 1
        for item_id in sid_to_item_ids.get(old_sid, []):
            weights[item_id] = float(max(total, 1))
    return weights


def collect_item_occurrence_weights_from_train_samples(
    train_data_path: Path,
    index_data: dict[str, list[str]],
    item_token_size: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    samples = load_json(train_data_path)
    if not isinstance(samples, list):
        raise TypeError(f"Expected train samples list at {train_data_path}, got {type(samples).__name__}")

    sid_to_item_id = {sid_string(tokens): item_id for item_id, tokens in index_data.items()}
    weights = {item_id: 0.0 for item_id in index_data}
    total_text_items = 0
    total_target_items = 0
    matched_text_items = 0
    matched_target_items = 0

    def update_sid_occurrences(raw_text: str, *, is_target: bool) -> None:
        nonlocal total_text_items, total_target_items, matched_text_items, matched_target_items
        tokens = BASE_TOKEN_RE.findall(raw_text)
        if not tokens:
            return
        if len(tokens) % item_token_size != 0:
            raise ValueError(
                f"Cannot split train sample text into item chunks of {item_token_size}: "
                f"num_tokens={len(tokens)} text={raw_text[:120]}"
            )
        for start in range(0, len(tokens), item_token_size):
            sid = sid_string(tokens[start : start + item_token_size])
            item_id = sid_to_item_id.get(sid)
            if is_target:
                total_target_items += 1
            else:
                total_text_items += 1
            if item_id is not None:
                weights[item_id] += 1.0
                if is_target:
                    matched_target_items += 1
                else:
                    matched_text_items += 1

    for sample in samples:
        update_sid_occurrences(str(sample.get("text", "")), is_target=False)
        for target in sample.get("sid_ground_truth", []):
            update_sid_occurrences(str(target), is_target=True)

    meta = {
        "train_data_path": str(train_data_path),
        "num_samples": len(samples),
        "total_text_items": total_text_items,
        "matched_text_items": matched_text_items,
        "total_target_items": total_target_items,
        "matched_target_items": matched_target_items,
        "num_items_with_positive_weight": sum(1 for weight in weights.values() if weight > 0),
    }
    return weights, meta


def collect_candidate_stats(
    index_data: dict[str, list[str]],
    item_weights: dict[str, float],
    min_span_len: int,
    max_span_len: int,
    allowed_start_slots: set[int] | None,
) -> tuple[Counter[tuple[str, ...]], Counter[tuple[str, ...]]]:
    weighted_freq: Counter[tuple[str, ...]] = Counter()
    item_support: Counter[tuple[str, ...]] = Counter()

    for item_id, tokens in index_data.items():
        weight = float(item_weights.get(item_id, 1.0))
        seen_in_item = set()
        for start in range(len(tokens)):
            for span_len in range(min_span_len, max_span_len + 1):
                end = start + span_len
                if end > len(tokens):
                    continue
                if allowed_start_slots is not None and start not in allowed_start_slots:
                    continue
                span = tuple(tokens[start:end])
                weighted_freq[span] += weight
                seen_in_item.add(span)
        for span in seen_in_item:
            item_support[span] += 1
    return weighted_freq, item_support


def span_respects_allowed_slots(span: tuple[str, ...], allowed_slots: set[int] | None) -> bool:
    if allowed_slots is None:
        return True
    for token in span:
        slot = slot_index(token)
        if slot is None or slot not in allowed_slots:
            return False
    return True


def collect_prefix_stats(
    index_data: dict[str, list[str]],
    item_weights: dict[str, float],
    min_span_len: int,
    max_span_len: int,
) -> tuple[Counter[tuple[str, ...]], Counter[tuple[str, ...]], dict[tuple[str, ...], dict[str, float]]]:
    weighted_freq: Counter[tuple[str, ...]] = Counter()
    item_support_sets: dict[tuple[str, ...], set[str]] = defaultdict(set)
    suffix_counts: dict[tuple[str, ...], Counter[tuple[str, ...]]] = defaultdict(Counter)

    for item_id, tokens in index_data.items():
        weight = float(item_weights.get(item_id, 0.0))
        if weight <= 0.0:
            continue
        max_len = min(max_span_len, len(tokens) - 1)
        for span_len in range(min_span_len, max_len + 1):
            prefix = tuple(tokens[:span_len])
            suffix = tuple(tokens[span_len:])
            if not suffix:
                continue
            weighted_freq[prefix] += weight
            item_support_sets[prefix].add(item_id)
            suffix_counts[prefix][suffix] += weight

    item_support = Counter({prefix: len(items) for prefix, items in item_support_sets.items()})
    stats: dict[tuple[str, ...], dict[str, float]] = {}
    for prefix, counter in suffix_counts.items():
        total = sum(counter.values())
        entropy = 0.0
        for count in counter.values():
            prob = count / total
            entropy -= prob * math.log(prob)
        num_suffixes = len(counter)
        max_entropy = math.log(num_suffixes) if num_suffixes > 1 else 0.0
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        purity = 1.0 - norm_entropy
        stats[prefix] = {
            "suffix_entropy": entropy,
            "suffix_norm_entropy": norm_entropy,
            "suffix_purity": purity,
            "num_suffixes": float(num_suffixes),
        }
    return weighted_freq, item_support, stats


def select_frequency_patterns(
    weighted_freq: Counter[tuple[str, ...]],
    item_support: Counter[tuple[str, ...]],
    top_k: int,
    min_weighted_freq: float,
    min_item_support: int,
) -> dict[str, dict[str, Any]]:
    candidates = []
    used_names = set()
    for span, freq in weighted_freq.items():
        support = item_support[span]
        if freq < min_weighted_freq or support < min_item_support:
            continue
        name = pattern_name(span)
        if name in used_names:
            continue
        used_names.add(name)
        gain = float(freq) * (len(span) - 1)
        candidates.append((gain, float(freq), support, len(span), name, span))

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]), reverse=True)
    selected = candidates[:top_k]

    patterns: dict[str, dict[str, Any]] = {}
    for gain, freq, support, span_len, name, span in selected:
        patterns[name] = {
            "tokens": list(span),
            "length": span_len,
            "weighted_freq": freq,
            "item_support": int(support),
            "gain": gain,
            "start_slot": slot_index(span[0]),
            "end_slot": slot_index(span[-1]),
        }
    return patterns


def select_prefix_entropy_patterns(
    weighted_freq: Counter[tuple[str, ...]],
    item_support: Counter[tuple[str, ...]],
    suffix_stats: dict[tuple[str, ...], dict[str, float]],
    top_k: int,
    min_weighted_freq: float,
    min_item_support: int,
    min_purity: float,
) -> dict[str, dict[str, Any]]:
    candidates = []
    used_names = set()
    for prefix, freq in weighted_freq.items():
        support = item_support[prefix]
        if freq < min_weighted_freq or support < min_item_support:
            continue
        stats = suffix_stats.get(prefix, {})
        purity = float(stats.get("suffix_purity", 0.0))
        if purity < min_purity:
            continue
        name = pattern_name(prefix)
        if name in used_names:
            continue
        used_names.add(name)
        gain = float(freq) * (len(prefix) - 1) * purity
        candidates.append((gain, float(freq), support, purity, len(prefix), name, prefix, stats))

    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[4], item[5]), reverse=True)
    selected = candidates[:top_k]

    patterns: dict[str, dict[str, Any]] = {}
    for score, freq, support, purity, span_len, name, prefix, stats in selected:
        patterns[name] = {
            "tokens": list(prefix),
            "length": span_len,
            "score": score,
            "weighted_freq": freq,
            "item_support": int(support),
            "suffix_entropy": float(stats.get("suffix_entropy", math.nan)),
            "suffix_norm_entropy": float(stats.get("suffix_norm_entropy", math.nan)),
            "suffix_purity": purity,
            "num_suffixes": int(stats.get("num_suffixes", 0)),
            "start_slot": 0,
            "end_slot": span_len - 1,
        }
    return patterns


def merge_token_name(left_token: str, right_token: str) -> str:
    return pattern_name(tuple(expand_pattern_token(left_token) + expand_pattern_token(right_token)))


def expand_pattern_token(token: str) -> list[str]:
    if not token.startswith("<p_") or not token.endswith(">"):
        return [token]
    body = token[3:-1]
    parts = body.split("_") if body else []
    expanded = []
    for part in parts:
        match = re.fullmatch(r"([a-z])(\d+)", part)
        if match is None:
            return [token]
        expanded.append(f"<{match.group(1)}_{match.group(2)}>")
    return expanded


def collect_adjacent_pair_stats(
    encoded_items: dict[str, list[str]],
    item_weights: dict[str, float],
) -> tuple[Counter[tuple[str, str]], Counter[str], Counter[str], Counter[tuple[str, str]]]:
    pair_freq: Counter[tuple[str, str]] = Counter()
    left_freq: Counter[str] = Counter()
    right_freq: Counter[str] = Counter()
    pair_support_sets: dict[tuple[str, str], set[str]] = defaultdict(set)

    for item_id, tokens in encoded_items.items():
        weight = float(item_weights.get(item_id, 1.0))
        for left, right in zip(tokens, tokens[1:]):
            pair = (left, right)
            pair_freq[pair] += weight
            left_freq[left] += weight
            right_freq[right] += weight
            pair_support_sets[pair].add(item_id)

    pair_support = Counter({pair: len(items) for pair, items in pair_support_sets.items()})
    return pair_freq, left_freq, right_freq, pair_support


def conditional_entropy_purity(distribution: Counter[str]) -> tuple[float, float, int]:
    positive_counts = [float(count) for count in distribution.values() if float(count) > 0.0]
    total = sum(positive_counts)
    if total <= 0.0:
        return math.nan, 0.0, 0
    entropy = 0.0
    for count in positive_counts:
        prob = count / total
        entropy -= prob * math.log(prob)
    num_options = len(positive_counts)
    max_entropy = math.log(num_options) if num_options > 1 else 0.0
    norm_entropy = entropy / max_entropy if max_entropy > 0.0 else 0.0
    purity = 1.0 - norm_entropy
    return entropy, purity, num_options


def collect_neighbor_distributions(
    pair_freq: Counter[tuple[str, str]],
) -> tuple[dict[str, Counter[str]], dict[str, Counter[str]]]:
    next_by_left: dict[str, Counter[str]] = defaultdict(Counter)
    prev_by_right: dict[str, Counter[str]] = defaultdict(Counter)
    for (left, right), freq in pair_freq.items():
        next_by_left[left][right] += freq
        prev_by_right[right][left] += freq
    return next_by_left, prev_by_right


def train_iterative_pair_merge_patterns(
    index_data: dict[str, list[str]],
    item_weights: dict[str, float],
    strategy: str,
    top_k: int,
    max_span_len: int,
    min_weighted_freq: float,
    min_item_support: int,
    allowed_slots: set[int] | None = None,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], int]:
    encoded_items = {item_id: list(tokens) for item_id, tokens in index_data.items()}
    patterns: dict[str, dict[str, Any]] = {}
    merge_rules: list[dict[str, Any]] = []
    used_names = set()

    for step in range(top_k):
        pair_freq, left_freq, right_freq, pair_support = collect_adjacent_pair_stats(encoded_items, item_weights)
        next_by_left, prev_by_right = collect_neighbor_distributions(pair_freq)
        total_pairs = max(float(sum(pair_freq.values())), 1.0)
        candidates = []
        for pair, freq in pair_freq.items():
            support = pair_support[pair]
            if freq < min_weighted_freq or support < min_item_support:
                continue
            name = merge_token_name(pair[0], pair[1])
            if name in used_names:
                continue
            merged_span = tuple(expand_pattern_token(pair[0]) + expand_pattern_token(pair[1]))
            if len(merged_span) > max_span_len:
                continue
            if not span_respects_allowed_slots(merged_span, allowed_slots):
                continue
            if strategy == "bpe":
                score = float(freq)
                pmi = math.nan
                cond_entropy_next = math.nan
                cond_entropy_prev = math.nan
                next_purity = math.nan
                prev_purity = math.nan
                next_prob = math.nan
                prev_prob = math.nan
            elif strategy == "wordpiece":
                pmi = math.log((float(freq) + 1.0) * total_pairs / ((float(left_freq[pair[0]]) + 1.0) * (float(right_freq[pair[1]]) + 1.0)))
                score = float(freq) * max(pmi, 0.0)
                if score <= 0.0:
                    continue
                cond_entropy_next = math.nan
                cond_entropy_prev = math.nan
                next_purity = math.nan
                prev_purity = math.nan
                next_prob = math.nan
                prev_prob = math.nan
            elif strategy == "cond_entropy_bpe":
                pmi = math.nan
                cond_entropy_next, next_purity, _ = conditional_entropy_purity(next_by_left[pair[0]])
                cond_entropy_prev, prev_purity, _ = conditional_entropy_purity(prev_by_right[pair[1]])
                next_prob = float(freq) / max(float(left_freq[pair[0]]), 1.0)
                prev_prob = float(freq) / max(float(right_freq[pair[1]]), 1.0)
                bidirectional_prob = math.sqrt(max(next_prob, 0.0) * max(prev_prob, 0.0))
                bidirectional_purity = math.sqrt(max(next_purity, 0.0) * max(prev_purity, 0.0))
                score = float(freq) * bidirectional_prob * bidirectional_purity
                if score <= 0.0:
                    continue
            else:
                raise ValueError(f"Unsupported iterative merge strategy: {strategy}")
            candidates.append(
                (
                    score,
                    float(freq),
                    support,
                    len(merged_span),
                    name,
                    pair,
                    merged_span,
                    pmi,
                    cond_entropy_next,
                    cond_entropy_prev,
                    next_purity,
                    prev_purity,
                    next_prob,
                    prev_prob,
                )
            )

        if not candidates:
            break

        candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]), reverse=True)
        (
            score,
            freq,
            support,
            span_len,
            name,
            pair,
            merged_span,
            pmi,
            cond_entropy_next,
            cond_entropy_prev,
            next_purity,
            prev_purity,
            next_prob,
            prev_prob,
        ) = candidates[0]
        used_names.add(name)
        patterns[name] = {
            "tokens": list(merged_span),
            "length": span_len,
            "score": score,
            "weighted_freq": freq,
            "item_support": int(support),
            "pmi": pmi,
            "cond_entropy_next": cond_entropy_next,
            "cond_entropy_prev": cond_entropy_prev,
            "next_purity": next_purity,
            "prev_purity": prev_purity,
            "next_prob": next_prob,
            "prev_prob": prev_prob,
            "merge_step": step + 1,
            "merge_pair": list(pair),
            "start_slot": slot_index(merged_span[0]),
            "end_slot": slot_index(merged_span[-1]),
        }
        merge_rules.append(
            {
                "step": step + 1,
                "left": pair[0],
                "right": pair[1],
                "new_token": name,
                "tokens": list(merged_span),
                "score": score,
                "weighted_freq": freq,
                "item_support": int(support),
                "pmi": pmi,
                "cond_entropy_next": cond_entropy_next,
                "cond_entropy_prev": cond_entropy_prev,
                "next_purity": next_purity,
                "prev_purity": prev_purity,
                "next_prob": next_prob,
                "prev_prob": prev_prob,
            }
        )
        encoded_items = {
            item_id: apply_single_pair_merge(tokens, pair, name)
            for item_id, tokens in encoded_items.items()
        }

    max_pattern_len = max((info["length"] for info in patterns.values()), default=1)
    return patterns, merge_rules, max_pattern_len


def apply_single_pair_merge(tokens: list[str], pair: tuple[str, str], new_token: str) -> list[str]:
    merged: list[str] = []
    pos = 0
    while pos < len(tokens):
        if pos + 1 < len(tokens) and tokens[pos] == pair[0] and tokens[pos + 1] == pair[1]:
            merged.append(new_token)
            pos += 2
        else:
            merged.append(tokens[pos])
            pos += 1
    return merged


def slot_index(token: str) -> int | None:
    if len(token) < 3 or token[0] != "<":
        return None
    letter = token[1]
    if not ("a" <= letter <= "z"):
        return None
    return ord(letter) - ord("a")


def build_span_lookup(patterns: dict[str, dict[str, Any]]) -> dict[tuple[str, ...], str]:
    return {tuple(info["tokens"]): name for name, info in patterns.items()}


def encode_index(
    index_data: dict[str, list[str]],
    patterns: dict[str, dict[str, Any]],
    max_pattern_len: int,
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, dict[str, Any]], Counter[str]]:
    span_lookup = build_span_lookup(patterns)
    new_index: dict[str, list[str]] = {}
    old_sid_to_new: dict[str, str] = {}
    merged_items: dict[str, dict[str, Any]] = {}
    pattern_usage: Counter[str] = Counter()

    for item_id, tokens in index_data.items():
        encoded = encode_tokens_longest_match(tokens, span_lookup, max_pattern_len)
        new_index[item_id] = encoded
        old_sid_to_new[sid_string(tokens)] = sid_string(encoded)
        applied_patterns = collect_applied_patterns(encoded, patterns)
        pattern_usage.update(applied_patterns)
        if applied_patterns:
            merged_items[item_id] = {
                "old_tokens": tokens,
                "new_tokens": encoded,
                "old_sid": sid_string(tokens),
                "new_sid": sid_string(encoded),
                "applied_patterns": applied_patterns,
            }
    return new_index, old_sid_to_new, merged_items, pattern_usage


def encode_index_with_merge_rules(
    index_data: dict[str, list[str]],
    patterns: dict[str, dict[str, Any]],
    merge_rules: list[dict[str, Any]],
) -> tuple[dict[str, list[str]], dict[str, str], dict[str, dict[str, Any]], Counter[str]]:
    new_index: dict[str, list[str]] = {}
    old_sid_to_new: dict[str, str] = {}
    merged_items: dict[str, dict[str, Any]] = {}
    pattern_usage: Counter[str] = Counter()
    ordered_rules = [
        ((str(rule["left"]), str(rule["right"])), str(rule["new_token"]))
        for rule in sorted(merge_rules, key=lambda item: int(item["step"]))
    ]

    for item_id, tokens in index_data.items():
        encoded = list(tokens)
        for pair, new_token in ordered_rules:
            encoded = apply_single_pair_merge(encoded, pair, new_token)
        new_index[item_id] = encoded
        old_sid_to_new[sid_string(tokens)] = sid_string(encoded)
        applied_patterns = collect_applied_patterns(encoded, patterns)
        pattern_usage.update(applied_patterns)
        if applied_patterns:
            merged_items[item_id] = {
                "old_tokens": tokens,
                "new_tokens": encoded,
                "old_sid": sid_string(tokens),
                "new_sid": sid_string(encoded),
                "applied_patterns": applied_patterns,
            }
    return new_index, old_sid_to_new, merged_items, pattern_usage


def prune_iterative_merge_patterns_by_usage(
    index_data: dict[str, list[str]],
    patterns: dict[str, dict[str, Any]],
    merge_rules: list[dict[str, Any]],
    min_pattern_usage: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str], dict[str, dict[str, Any]], Counter[str]]:
    current_rules = sorted(merge_rules, key=lambda item: int(item["step"]))
    current_patterns = {
        str(rule["new_token"]): patterns[str(rule["new_token"])]
        for rule in current_rules
        if str(rule["new_token"]) in patterns
    }

    while True:
        _new_index, old_sid_to_new, merged_items, pattern_usage = encode_index_with_merge_rules(
            index_data=index_data,
            patterns=current_patterns,
            merge_rules=current_rules,
        )
        generated_tokens: set[str] = set()
        kept_rules: list[dict[str, Any]] = []
        for rule in current_rules:
            left = str(rule["left"])
            right = str(rule["right"])
            new_token = str(rule["new_token"])
            dependencies_available = (
                (not left.startswith("<p_") or left in generated_tokens)
                and (not right.startswith("<p_") or right in generated_tokens)
            )
            if not dependencies_available:
                continue
            if pattern_usage.get(new_token, 0) < min_pattern_usage:
                continue
            kept_rules.append(rule)
            generated_tokens.add(new_token)

        kept_tokens = {str(rule["new_token"]) for rule in kept_rules}
        if len(kept_rules) == len(current_rules):
            kept_patterns = {
                token: info
                for token, info in current_patterns.items()
                if token in kept_tokens
            }
            return kept_patterns, kept_rules, old_sid_to_new, merged_items, pattern_usage

        current_rules = kept_rules
        current_patterns = {
            token: info
            for token, info in current_patterns.items()
            if token in kept_tokens
        }


def encode_tokens_longest_match(tokens: list[str], span_lookup: dict[tuple[str, ...], str], max_pattern_len: int) -> list[str]:
    encoded: list[str] = []
    pos = 0
    while pos < len(tokens):
        matched = None
        max_len = min(max_pattern_len, len(tokens) - pos)
        for span_len in range(max_len, 1, -1):
            span = tuple(tokens[pos : pos + span_len])
            token = span_lookup.get(span)
            if token is not None:
                matched = (token, span_len)
                break
        if matched is None:
            encoded.append(tokens[pos])
            pos += 1
        else:
            encoded.append(matched[0])
            pos += matched[1]
    return encoded


def collect_applied_patterns(encoded_tokens: list[str], patterns: dict[str, dict[str, Any]]) -> list[str]:
    return [token for token in encoded_tokens if token in patterns]


def rewrite_sid_text(text: str, old_sid_to_new: dict[str, str], item_token_size: int) -> str:
    tokens = BASE_TOKEN_RE.findall(text)
    if not tokens:
        return text
    if len(tokens) % item_token_size != 0:
        raise ValueError(
            f"Cannot split SID text into item chunks of {item_token_size}: "
            f"num_tokens={len(tokens)} text={text[:120]}"
        )
    rewritten: list[str] = []
    for start in range(0, len(tokens), item_token_size):
        old_sid = sid_string(tokens[start : start + item_token_size])
        try:
            rewritten.append(old_sid_to_new[old_sid])
        except KeyError as exc:
            raise KeyError(f"SID not found in index mapping: {old_sid}") from exc
    return "".join(rewritten)


def rewrite_split_file(
    input_path: Path,
    output_path: Path,
    old_sid_to_new: dict[str, str],
    item_token_size: int,
) -> int:
    samples = load_json(input_path)
    for sample in samples:
        sample["text"] = rewrite_sid_text(str(sample["text"]), old_sid_to_new, item_token_size)
        if "sid_ground_truth" in sample:
            sample["sid_ground_truth"] = [
                rewrite_sid_text(str(target), old_sid_to_new, item_token_size)
                for target in sample["sid_ground_truth"]
            ]
    dump_json(output_path, samples)
    return len(samples)


def rewrite_sid2pid(
    sid2pid_path: Path,
    output_path: Path,
    old_sid_to_new: dict[str, str],
) -> None:
    old_sid2pid = load_json(sid2pid_path)
    new_sid2pid = {}
    for old_sid, candidates in old_sid2pid.items():
        new_sid = old_sid_to_new.get(old_sid)
        if new_sid is not None:
            new_sid2pid.setdefault(new_sid, []).extend(candidates)
    dump_json(output_path, new_sid2pid)


def copy_support_files(input_dir: Path, output_dir: Path, prefix: str) -> None:
    filenames = [
        f"{prefix}.emb.npy",
        "quantizer.pkl",
        "quantization_meta.json",
        "semantic_embedding_projection_pca_h128_s42_scale0.0883883.npz",
        "semantic_embedding_projection_pca_h256_s42_scale0.02.npz",
        "split_stats.json",
    ]
    for filename in filenames:
        src = input_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--selection-strategy",
        choices=["frequency", "bpe", "wordpiece", "cond_entropy_bpe", "prefix_entropy"],
        default="frequency",
    )
    parser.add_argument("--top-k", type=int, default=512)
    parser.add_argument("--min-span-len", type=int, default=2)
    parser.add_argument("--max-span-len", type=int, default=4)
    parser.add_argument("--min-weighted-freq", type=float, default=2.0)
    parser.add_argument("--min-item-support", type=int, default=2)
    parser.add_argument("--min-purity", type=float, default=0.0)
    parser.add_argument(
        "--allowed-start-slots",
        type=str,
        default=None,
        help="Comma-separated zero-based SID start slots allowed for patterns, e.g. '0' for <a,b> only.",
    )
    parser.add_argument(
        "--allowed-slots",
        type=str,
        default=None,
        help="Comma-separated zero-based SID slots allowed anywhere in a pattern, e.g. '0,1,2' to block d-slot merges.",
    )
    parser.add_argument(
        "--min-pattern-usage",
        type=int,
        default=0,
        help="Drop selected patterns used by fewer than this many final item SIDs, then re-encode.",
    )
    parser.add_argument(
        "--keep-minimal-files",
        action="store_true",
        help="Only keep train_single.py files plus varlen_meta.json.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory exists: {args.output_dir}. Use --overwrite to replace it.")
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prefix = infer_dataset_prefix(args.input_dir)
    index_path = args.input_dir / f"{prefix}.index.json"
    sid2pid_path = args.input_dir / f"{prefix}.sid2pid.json"
    index_data: dict[str, list[str]] = load_json(index_path)
    if not index_data:
        raise ValueError(f"Empty index file: {index_path}")

    first_len = len(next(iter(index_data.values())))
    if args.max_span_len > first_len:
        args.max_span_len = first_len
    allowed_start_slots = None
    allowed_slots = None
    merge_rules: list[dict[str, Any]] = []
    corpus_weight_meta: dict[str, Any] | None = None

    if args.allowed_slots is not None:
        allowed_slots = {int(part) for part in args.allowed_slots.split(",") if part.strip() != ""}

    if args.selection_strategy == "frequency":
        item_weights = load_item_weights(sid2pid_path, index_data)
        if args.allowed_start_slots is not None:
            allowed_start_slots = {int(part) for part in args.allowed_start_slots.split(",") if part.strip() != ""}
        weighted_freq, item_support = collect_candidate_stats(
            index_data=index_data,
            item_weights=item_weights,
            min_span_len=args.min_span_len,
            max_span_len=args.max_span_len,
            allowed_start_slots=allowed_start_slots,
        )
        patterns = select_frequency_patterns(
            weighted_freq=weighted_freq,
            item_support=item_support,
            top_k=args.top_k,
            min_weighted_freq=args.min_weighted_freq,
            min_item_support=args.min_item_support,
        )
        encode_max_pattern_len = args.max_span_len
    elif args.selection_strategy in {"bpe", "wordpiece", "cond_entropy_bpe", "prefix_entropy"}:
        train_data_path = args.input_dir / "train_data.json"
        if not train_data_path.exists():
            raise FileNotFoundError(f"{args.selection_strategy} requires train_data.json under input dir: {train_data_path}")
        item_weights, corpus_weight_meta = collect_item_occurrence_weights_from_train_samples(
            train_data_path=train_data_path,
            index_data=index_data,
            item_token_size=first_len,
        )
        if args.selection_strategy in {"bpe", "wordpiece", "cond_entropy_bpe"}:
            args.min_span_len = 2
            args.max_span_len = min(args.max_span_len, first_len)
            patterns, merge_rules, encode_max_pattern_len = train_iterative_pair_merge_patterns(
                index_data=index_data,
                item_weights=item_weights,
                strategy=args.selection_strategy,
                top_k=args.top_k,
                max_span_len=args.max_span_len,
                min_weighted_freq=args.min_weighted_freq,
                min_item_support=args.min_item_support,
                allowed_slots=allowed_slots,
            )
        else:
            args.min_span_len = max(args.min_span_len, 2)
            args.max_span_len = min(args.max_span_len, first_len - 1)
            allowed_start_slots = {0}
            weighted_freq, item_support, suffix_stats = collect_prefix_stats(
                index_data=index_data,
                item_weights=item_weights,
                min_span_len=args.min_span_len,
                max_span_len=args.max_span_len,
            )
            patterns = select_prefix_entropy_patterns(
                weighted_freq=weighted_freq,
                item_support=item_support,
                suffix_stats=suffix_stats,
                top_k=args.top_k,
                min_weighted_freq=args.min_weighted_freq,
                min_item_support=args.min_item_support,
                min_purity=args.min_purity,
            )
            encode_max_pattern_len = args.max_span_len

    if args.selection_strategy in {"bpe", "wordpiece", "cond_entropy_bpe"}:
        new_index, old_sid_to_new, merged_items, pattern_usage_counter = encode_index_with_merge_rules(
            index_data=index_data,
            patterns=patterns,
            merge_rules=merge_rules,
        )
    else:
        new_index, old_sid_to_new, merged_items, pattern_usage_counter = encode_index(
            index_data=index_data,
            patterns=patterns,
            max_pattern_len=encode_max_pattern_len,
        )
    if args.selection_strategy in {"frequency", "prefix_entropy"} and args.min_pattern_usage > 0:
        patterns = {
            token: info
            for token, info in patterns.items()
            if pattern_usage_counter.get(token, 0) >= args.min_pattern_usage
        }
        new_index, old_sid_to_new, merged_items, pattern_usage_counter = encode_index(
            index_data=index_data,
            patterns=patterns,
            max_pattern_len=encode_max_pattern_len,
        )
    if args.selection_strategy in {"bpe", "wordpiece", "cond_entropy_bpe"} and args.min_pattern_usage > 0:
        patterns, merge_rules, old_sid_to_new, merged_items, pattern_usage_counter = prune_iterative_merge_patterns_by_usage(
            index_data=index_data,
            patterns=patterns,
            merge_rules=merge_rules,
            min_pattern_usage=args.min_pattern_usage,
        )
        new_index, old_sid_to_new, merged_items, pattern_usage_counter = encode_index_with_merge_rules(
            index_data=index_data,
            patterns=patterns,
            merge_rules=merge_rules,
        )

    dump_json(args.output_dir / f"{prefix}.index.json", new_index)

    if sid2pid_path.exists():
        rewrite_sid2pid(sid2pid_path, args.output_dir / f"{prefix}.sid2pid.json", old_sid_to_new)

    split_counts = {}
    for filename in ("train_data.json", "val_data.json", "test_data.json"):
        input_path = args.input_dir / filename
        if input_path.exists():
            split_counts[filename] = rewrite_split_file(
                input_path=input_path,
                output_path=args.output_dir / filename,
                old_sid_to_new=old_sid_to_new,
                item_token_size=first_len,
            )
    copy_support_files(args.input_dir, args.output_dir, prefix)

    lengths = [len(tokens) for tokens in new_index.values()]
    length_distribution = dict(sorted(Counter(lengths).items()))
    token_counts = Counter(token for tokens in new_index.values() for token in tokens)
    pattern_usage = {token: count for token, count in token_counts.items() if token in patterns}
    max_new_len = max(lengths) if lengths else 0
    total_old_len = sum(len(tokens) for tokens in index_data.values())
    total_new_len = sum(lengths)
    if args.selection_strategy == "frequency":
        method = "bpe_inspired_constrained_span_merge"
    elif args.selection_strategy == "bpe":
        method = "strict_bpe_iterative_pair_merge"
    elif args.selection_strategy == "wordpiece":
        method = "wordpiece_style_iterative_pair_merge"
    elif args.selection_strategy == "cond_entropy_bpe":
        method = "conditional_entropy_iterative_pair_merge"
    elif args.selection_strategy == "prefix_entropy":
        method = "prefix_entropy_suffix_purity_span_merge"
    else:
        raise ValueError(f"Unsupported selection strategy: {args.selection_strategy}")
    meta = {
        "method": method,
        "source_dataset": str(args.input_dir),
        "dataset_prefix": prefix,
        "selection_strategy": args.selection_strategy,
        "top_k": args.top_k,
        "min_span_len": args.min_span_len,
        "max_span_len": args.max_span_len,
        "min_weighted_freq": args.min_weighted_freq,
        "min_item_support": args.min_item_support,
        "allowed_start_slots": sorted(allowed_start_slots) if allowed_start_slots is not None else None,
        "min_pattern_usage": args.min_pattern_usage,
        "min_purity": args.min_purity,
        "num_items": len(index_data),
        "num_patterns": len(patterns),
        "avg_len_before": total_old_len / max(len(index_data), 1),
        "avg_len_after": total_new_len / max(len(index_data), 1),
        "max_len_after": max_new_len,
        "length_distribution_after": length_distribution,
        "p50_len_after": percentile(lengths, 50),
        "p90_len_after": percentile(lengths, 90),
        "p99_len_after": percentile(lengths, 99),
        "num_used_pattern_tokens": len(pattern_usage),
        "num_merged_items": len(merged_items),
        "split_counts": split_counts,
        "extra_semantic_tokens": sorted(patterns),
        "patterns": patterns,
        "pattern_usage": pattern_usage,
        "merged_items": merged_items,
    }
    if corpus_weight_meta is not None:
        meta["corpus_weighting"] = corpus_weight_meta
    if merge_rules:
        meta["merge_rules"] = merge_rules
    dump_json(args.output_dir / "varlen_meta.json", meta)

    print(f"Wrote varlen dataset to {args.output_dir}")
    print(f"items={len(index_data)} patterns={len(patterns)}")
    print(f"avg_len: {meta['avg_len_before']:.3f} -> {meta['avg_len_after']:.3f}; max_len={max_new_len}")
    print(f"splits={split_counts}")


def percentile(values: list[int], q: float) -> float:
    if not values:
        return math.nan
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_values[lo])
    weight = pos - lo
    return float(sorted_values[lo] * (1 - weight) + sorted_values[hi] * weight)


if __name__ == "__main__":
    main()
