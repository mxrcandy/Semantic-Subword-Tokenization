#!/usr/bin/env python3
"""Build prefix-pair augmentation train files from fixed-length SID sequences."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SEMANTIC_TOKEN_RE = re.compile(r"<[a-z]_\d+>")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def parse_sid_tokens(text: str) -> list[str]:
    return SEMANTIC_TOKEN_RE.findall(text)


def parse_fixed_items(text: str, item_len: int) -> list[str]:
    tokens = parse_sid_tokens(text)
    if len(tokens) % item_len != 0:
        raise ValueError(f"Text token length {len(tokens)} is not divisible by item_len={item_len}")
    return ["".join(tokens[i : i + item_len]) for i in range(0, len(tokens), item_len)]


def sid_prefix(sid: str, prefix_tokens: int) -> str:
    tokens = parse_sid_tokens(sid)
    if len(tokens) < prefix_tokens:
        raise ValueError(f"SID has only {len(tokens)} tokens, expected at least {prefix_tokens}: {sid}")
    return "".join(tokens[:prefix_tokens])


def try_seq2pat(prefix_sequences: list[list[str]], window_size: int) -> tuple[str, list[tuple[tuple[str, str], int]]] | None:
    try:
        from sequential.seq2pat import Seq2Pat  # type: ignore
    except Exception:
        return None

    try:
        encoded_token_to_id: dict[str, int] = {}
        next_id = 1
        encoded_sequences: list[list[int]] = []
        for seq in prefix_sequences:
            encoded: list[int] = []
            for token in seq:
                if token not in encoded_token_to_id:
                    encoded_token_to_id[token] = next_id
                    next_id += 1
                encoded.append(encoded_token_to_id[token])
            encoded_sequences.append(encoded)

        miner = Seq2Pat(encoded_sequences)
        patterns = miner.get_patterns(min_frequency=1)
        id_to_token = {v: k for k, v in encoded_token_to_id.items()}
        pair_counts: list[tuple[tuple[str, str], int]] = []
        for pattern, freq in patterns:
            if len(pattern) != 2:
                continue
            pair = (id_to_token[int(pattern[0])], id_to_token[int(pattern[1])])
            pair_counts.append((pair, int(freq)))
        pair_counts.sort(key=lambda x: (-x[1], x[0]))
        return f"seq2pat(window={window_size})", pair_counts
    except Exception:
        return None


def count_prefix_pairs(
    prefix_sequences: list[list[str]],
    window_size: int,
    exclude_self_pair: bool,
) -> tuple[str, dict[tuple[str, str], float], Counter[tuple[str, str]], Counter[str], Counter[str], float]:
    seq2pat_result = None
    if seq2pat_result is not None and not exclude_self_pair:
        method, pair_counts = seq2pat_result
        raw_counts = Counter({pair: count for pair, count in pair_counts})
        left_counts: Counter[str] = Counter()
        right_counts: Counter[str] = Counter()
        for (left_prefix, right_prefix), count in raw_counts.items():
            left_counts[left_prefix] += count
            right_counts[right_prefix] += count
        return method, {pair: float(count) for pair, count in raw_counts.items()}, raw_counts, left_counts, right_counts, float(sum(raw_counts.values()))

    weighted_scores: dict[tuple[str, str], float] = defaultdict(float)
    raw_counts: Counter[tuple[str, str]] = Counter()
    left_counts: Counter[str] = Counter()
    right_counts: Counter[str] = Counter()
    total_raw = 0
    for prefixes in prefix_sequences:
        for left_idx, left_prefix in enumerate(prefixes):
            upper = min(len(prefixes), left_idx + window_size + 1)
            for right_idx in range(left_idx + 1, upper):
                right_prefix = prefixes[right_idx]
                if exclude_self_pair and left_prefix == right_prefix:
                    continue
                pair = (left_prefix, right_prefix)
                distance = right_idx - left_idx
                weighted_scores[pair] += 1.0 / distance
                raw_counts[pair] += 1
                left_counts[left_prefix] += 1
                right_counts[right_prefix] += 1
                total_raw += 1
    method = f"sliding_window_count(window={window_size})"
    if exclude_self_pair:
        method = f"{method},exclude_self_pair"
    return method, weighted_scores, raw_counts, left_counts, right_counts, float(total_raw)


def score_prefix_pairs(
    weighted_scores: dict[tuple[str, str], float],
    raw_counts: Counter[tuple[str, str]],
    left_counts: Counter[str],
    right_counts: Counter[str],
    total_raw: float,
    pair_score: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    eps = 1e-12
    for pair, raw_count in raw_counts.items():
        left_prefix, right_prefix = pair
        weighted_count = float(weighted_scores.get(pair, raw_count))
        pmi = math.log((raw_count * total_raw + eps) / (left_counts[left_prefix] * right_counts[right_prefix] + eps))
        lift = math.exp(pmi)
        if pair_score == "count":
            score = float(raw_count)
        elif pair_score == "distance_weighted":
            score = weighted_count
        elif pair_score == "count_pmi":
            score = float(raw_count) * pmi
        elif pair_score == "lift":
            score = float(raw_count) * lift
        else:
            raise ValueError(f"Unsupported pair_score: {pair_score}")
        records.append(
            {
                "pair": pair,
                "score": score,
                "raw_count": int(raw_count),
                "weighted_count": weighted_count,
                "pmi": pmi,
                "lift": lift,
                "left_count": int(left_counts[left_prefix]),
                "right_count": int(right_counts[right_prefix]),
            }
        )
    records.sort(key=lambda x: (-float(x["score"]), x["pair"]))
    return records


def build_prefix_sequences_from_train_data(train_rows: list[dict[str, Any]], item_len: int, prefix_tokens: int) -> list[list[str]]:
    sequences: list[list[str]] = []
    for row in train_rows:
        history_items = parse_fixed_items(row["text"], item_len=item_len)
        target_sid = str(row["sid_ground_truth"][0])
        full_items = history_items + [target_sid]
        sequences.append([sid_prefix(item, prefix_tokens=prefix_tokens) for item in full_items])
    return sequences


def load_fixed_sid_to_pid(dataset_dir: Path) -> dict[str, str]:
    sid2pid_candidates = sorted(dataset_dir.glob("*.sid2pid.json"))
    if not sid2pid_candidates:
        raise FileNotFoundError(f"No *.sid2pid.json found in {dataset_dir}")
    sid2pid = load_json(sid2pid_candidates[0])
    fixed_sid_to_pid: dict[str, str] = {}
    for sid, candidates in sid2pid.items():
        if not candidates:
            continue
        first = candidates[0]
        if isinstance(first, dict):
            fixed_sid_to_pid[str(sid)] = str(first["pid"])
        else:
            fixed_sid_to_pid[str(sid)] = str(first)
    return fixed_sid_to_pid


def load_pid_to_sid(dataset_dir: Path) -> dict[str, str]:
    index_candidates = sorted(dataset_dir.glob("*.index.json"))
    if not index_candidates:
        raise FileNotFoundError(f"No *.index.json found in {dataset_dir}")
    index = load_json(index_candidates[0])
    return {str(pid): "".join(tokens) for pid, tokens in index.items()}


def load_inter_pid_sequences(dataset_dir: Path, leave_out_items: int = 2) -> tuple[list[list[str]], Path]:
    inter_candidates = sorted(dataset_dir.glob("*.inter.json"))
    if not inter_candidates:
        raise FileNotFoundError(f"No *.inter.json found in {dataset_dir}")
    inter_path = inter_candidates[0]
    inter = load_json(inter_path)
    if not isinstance(inter, dict):
        raise TypeError(f"Expected dict user->items in {inter_path}, got {type(inter).__name__}")
    sequences: list[list[str]] = []
    for items in inter.values():
        if not isinstance(items, list):
            continue
        train_items = items[:-leave_out_items] if leave_out_items > 0 else items
        if len(train_items) < 2:
            continue
        sequences.append([str(pid) for pid in train_items])
    return sequences, inter_path


def build_prefix_sequences_from_pid_sequences(
    pid_sequences: list[list[str]],
    fixed_pid_to_sid: dict[str, str],
    prefix_tokens: int,
) -> list[list[str]]:
    sequences: list[list[str]] = []
    for pids in pid_sequences:
        prefixes = []
        missing_sid = False
        for pid in pids:
            sid = fixed_pid_to_sid.get(str(pid))
            if sid is None:
                missing_sid = True
                break
            prefixes.append(sid_prefix(sid, prefix_tokens=prefix_tokens))
        if not missing_sid and len(prefixes) >= 2:
            sequences.append(prefixes)
    return sequences


def build_pair_item_pools(
    train_rows: list[dict[str, Any]],
    fixed_sid_to_pid: dict[str, str],
    item_len: int,
    prefix_tokens: int,
    window_size: int,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    pools: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in train_rows:
        history_items = parse_fixed_items(row["text"], item_len=item_len)
        target_sid = str(row["sid_ground_truth"][0])
        full_items = history_items + [target_sid]
        full_pids = []
        missing_pid = False
        for item_sid in full_items:
            pid = fixed_sid_to_pid.get(item_sid)
            if pid is None:
                missing_pid = True
                break
            full_pids.append(pid)
        if missing_pid:
            continue
        for left_idx, left_item in enumerate(full_items):
            upper = min(len(full_items), left_idx + window_size + 1)
            left_prefix = sid_prefix(left_item, prefix_tokens=prefix_tokens)
            for right_idx in range(left_idx + 1, upper):
                right_item = full_items[right_idx]
                right_prefix = sid_prefix(right_item, prefix_tokens=prefix_tokens)
                pools[(left_prefix, right_prefix)].append((full_pids[left_idx], full_pids[right_idx]))
    return pools


def build_pair_item_pools_from_pid_sequences(
    pid_sequences: list[list[str]],
    fixed_pid_to_sid: dict[str, str],
    prefix_tokens: int,
    window_size: int,
    selected_pair_keys: set[tuple[str, str]] | None = None,
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    pools: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for pids in pid_sequences:
        full_prefixes = []
        missing_sid = False
        for pid in pids:
            sid = fixed_pid_to_sid.get(str(pid))
            if sid is None:
                missing_sid = True
                break
            full_prefixes.append(sid_prefix(sid, prefix_tokens=prefix_tokens))
        if missing_sid:
            continue
        for left_idx, left_prefix in enumerate(full_prefixes):
            upper = min(len(full_prefixes), left_idx + window_size + 1)
            for right_idx in range(left_idx + 1, upper):
                right_prefix = full_prefixes[right_idx]
                pair = (left_prefix, right_prefix)
                if selected_pair_keys is not None and pair not in selected_pair_keys:
                    continue
                pools[pair].append((str(pids[left_idx]), str(pids[right_idx])))
    return pools


def build_sequence_replay_pools(
    train_rows: list[dict[str, Any]],
    fixed_sid_to_pid: dict[str, str],
    selected_pair_keys: set[tuple[str, str]],
    item_len: int,
    prefix_tokens: int,
    window_size: int,
    replay_history_items: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row_idx, row in enumerate(train_rows):
        history_items = parse_fixed_items(row["text"], item_len=item_len)
        target_sid = str(row["sid_ground_truth"][0])
        full_items = history_items + [target_sid]
        full_pids = []
        missing_pid = False
        for item_sid in full_items:
            pid = fixed_sid_to_pid.get(item_sid)
            if pid is None:
                missing_pid = True
                break
            full_pids.append(pid)
        if missing_pid:
            continue

        full_prefixes = [sid_prefix(item, prefix_tokens=prefix_tokens) for item in full_items]
        for left_idx, left_prefix in enumerate(full_prefixes):
            upper = min(len(full_prefixes), left_idx + window_size + 1)
            for right_idx in range(left_idx + 1, upper):
                right_prefix = full_prefixes[right_idx]
                pair = (left_prefix, right_prefix)
                if pair not in selected_pair_keys:
                    continue
                history_start = max(0, right_idx - replay_history_items)
                history_pids = full_pids[history_start:right_idx]
                if not history_pids:
                    continue
                pools[pair].append(
                    {
                        "history_pids": history_pids,
                        "target_pid": full_pids[right_idx],
                        "source_row_idx": row_idx,
                        "left_idx": left_idx,
                        "right_idx": right_idx,
                        "history_start": history_start,
                        "distance": right_idx - left_idx,
                    }
                )
    return pools


def build_sequence_replay_pools_from_pid_sequences(
    pid_sequences: list[list[str]],
    fixed_pid_to_sid: dict[str, str],
    selected_pair_keys: set[tuple[str, str]],
    prefix_tokens: int,
    window_size: int,
    replay_history_items: int,
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    pools: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row_idx, pids in enumerate(pid_sequences):
        full_prefixes = []
        missing_sid = False
        for pid in pids:
            sid = fixed_pid_to_sid.get(str(pid))
            if sid is None:
                missing_sid = True
                break
            full_prefixes.append(sid_prefix(sid, prefix_tokens=prefix_tokens))
        if missing_sid:
            continue

        for left_idx, left_prefix in enumerate(full_prefixes):
            upper = min(len(full_prefixes), left_idx + window_size + 1)
            for right_idx in range(left_idx + 1, upper):
                right_prefix = full_prefixes[right_idx]
                pair = (left_prefix, right_prefix)
                if pair not in selected_pair_keys:
                    continue
                history_start = max(0, right_idx - replay_history_items)
                history_pids = [str(pid) for pid in pids[history_start:right_idx]]
                if not history_pids:
                    continue
                pools[pair].append(
                    {
                        "history_pids": history_pids,
                        "target_pid": str(pids[right_idx]),
                        "source_row_idx": row_idx,
                        "left_idx": left_idx,
                        "right_idx": right_idx,
                        "history_start": history_start,
                        "distance": right_idx - left_idx,
                    }
                )
    return pools


def build_augmented_rows(
    base_rows: list[dict[str, Any]],
    selected_pairs: list[dict[str, Any]],
    pair_item_pools: dict[tuple[str, str], list[tuple[str, str]]],
    history_pid_to_sid: dict[str, str],
    target_pid_to_sid: dict[str, str],
    max_augmentations_per_pair: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    augmented_rows: list[dict[str, Any]] = []
    for pair_record in selected_pairs:
        left_prefix, right_prefix = pair_record["pair"]
        pool = pair_item_pools.get((left_prefix, right_prefix), [])
        if not pool:
            continue
        sample_size = min(len(pool), max_augmentations_per_pair)
        if sample_size <= 0:
            continue
        chosen = rng.sample(pool, sample_size) if len(pool) > sample_size else list(pool)
        for left_pid, right_pid in chosen:
            item_left = history_pid_to_sid.get(left_pid)
            item_right = target_pid_to_sid.get(right_pid)
            if item_left is None or item_right is None:
                continue
            augmented_rows.append(
                {
                    "text": item_left,
                    "sid_ground_truth": [item_right],
                    "pid_ground_truth": [right_pid],
                    "user_id": f"aug::{left_prefix}=>{right_prefix}",
                    "augmentation_meta": {
                        "type": "prefix_pair_sw",
                        "left_prefix": left_prefix,
                        "right_prefix": right_prefix,
                        "pair_count": int(pair_record["raw_count"]),
                        "pair_score": float(pair_record["score"]),
                        "weighted_count": float(pair_record["weighted_count"]),
                        "pmi": float(pair_record["pmi"]),
                        "lift": float(pair_record["lift"]),
                        "left_pid": left_pid,
                        "right_pid": right_pid,
                    },
                }
            )
    return base_rows + augmented_rows


def build_sequence_replay_augmented_rows(
    base_rows: list[dict[str, Any]],
    selected_pairs: list[dict[str, Any]],
    sequence_replay_pools: dict[tuple[str, str], list[dict[str, Any]]],
    history_pid_to_sid: dict[str, str],
    target_pid_to_sid: dict[str, str],
    max_augmentations_per_pair: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    augmented_rows: list[dict[str, Any]] = []
    for pair_record in selected_pairs:
        left_prefix, right_prefix = pair_record["pair"]
        pool = sequence_replay_pools.get((left_prefix, right_prefix), [])
        if not pool:
            continue
        sample_size = min(len(pool), max_augmentations_per_pair)
        if sample_size <= 0:
            continue
        chosen = rng.sample(pool, sample_size) if len(pool) > sample_size else list(pool)
        for replay_record in chosen:
            history_sids = []
            missing_sid = False
            for pid in replay_record["history_pids"]:
                sid = history_pid_to_sid.get(pid)
                if sid is None:
                    missing_sid = True
                    break
                history_sids.append(sid)
            target_pid = replay_record["target_pid"]
            target_sid = target_pid_to_sid.get(target_pid)
            if missing_sid or target_sid is None:
                continue
            augmented_rows.append(
                {
                    "text": "".join(history_sids),
                    "sid_ground_truth": [target_sid],
                    "pid_ground_truth": [target_pid],
                    "user_id": f"augseq::{left_prefix}=>{right_prefix}",
                    "augmentation_meta": {
                        "type": "prefix_pair_sequence_replay",
                        "left_prefix": left_prefix,
                        "right_prefix": right_prefix,
                        "pair_count": int(pair_record["raw_count"]),
                        "pair_score": float(pair_record["score"]),
                        "weighted_count": float(pair_record["weighted_count"]),
                        "pmi": float(pair_record["pmi"]),
                        "lift": float(pair_record["lift"]),
                        "history_pids": replay_record["history_pids"],
                        "target_pid": target_pid,
                        "source_row_idx": int(replay_record["source_row_idx"]),
                        "left_idx": int(replay_record["left_idx"]),
                        "right_idx": int(replay_record["right_idx"]),
                        "history_start": int(replay_record["history_start"]),
                        "distance": int(replay_record["distance"]),
                    },
                }
            )
    return base_rows + augmented_rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    hist_counts = [row["text"].count("<") for row in rows]
    tgt_counts = [str(row["sid_ground_truth"][0]).count("<") for row in rows]
    aug_rows = sum(1 for row in rows if "augmentation_meta" in row)
    return {
        "rows": len(rows),
        "augmented_rows": aug_rows,
        "history_tokens_avg": sum(hist_counts) / len(hist_counts) if hist_counts else 0.0,
        "target_tokens_avg": sum(tgt_counts) / len(tgt_counts) if tgt_counts else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dataset-dir",
        type=Path,
        required=True,
        help=(
            "Fixed source dataset dir containing *.inter.json and *.index.json. "
            "When set, mining uses each user sequence with the last two items removed."
        ),
    )
    parser.add_argument("--target-train-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-suffix", type=str, default="_prefixpair_sw5")
    parser.add_argument(
        "--augmentation-mode",
        choices=["pair", "sequence_replay"],
        default="pair",
        help=(
            "pair keeps the original left-item -> right-item augmentation; "
            "sequence_replay replays the source subsequence before the right item."
        ),
    )
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument(
        "--replay-history-items",
        type=int,
        default=None,
        help="Max history items for sequence_replay. Defaults to --window-size.",
    )
    parser.add_argument("--prefix-tokens", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=128)
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument(
        "--pair-score",
        choices=["count", "distance_weighted", "count_pmi", "lift"],
        default="count",
    )
    parser.add_argument("--max-augmentations-per-pair", type=int, default=10)
    parser.add_argument("--exclude-self-pair", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    fixed_pid_to_sid = load_pid_to_sid(args.source_dataset_dir)
    source_pid_sequences, source_inter_path = load_inter_pid_sequences(args.source_dataset_dir)
    prefix_sequences = build_prefix_sequences_from_pid_sequences(
        source_pid_sequences,
        fixed_pid_to_sid=fixed_pid_to_sid,
        prefix_tokens=args.prefix_tokens,
    )
    pattern_method, weighted_scores, raw_counts, left_counts, right_counts, total_raw = count_prefix_pairs(
        prefix_sequences=prefix_sequences,
        window_size=args.window_size,
        exclude_self_pair=args.exclude_self_pair,
    )
    scored_pairs = score_prefix_pairs(
        weighted_scores=weighted_scores,
        raw_counts=raw_counts,
        left_counts=left_counts,
        right_counts=right_counts,
        total_raw=total_raw,
        pair_score=args.pair_score,
    )
    selected_pairs = [
        record
        for record in scored_pairs
        if int(record["raw_count"]) >= args.min_count
    ][: args.top_k]
    pair_item_pools = {}
    sequence_replay_pools = {}
    replay_history_items = args.replay_history_items or args.window_size
    selected_pair_keys = {record["pair"] for record in selected_pairs}
    if args.augmentation_mode == "pair":
        pair_item_pools = build_pair_item_pools_from_pid_sequences(
            source_pid_sequences,
            fixed_pid_to_sid=fixed_pid_to_sid,
            prefix_tokens=args.prefix_tokens,
            window_size=args.window_size,
            selected_pair_keys=selected_pair_keys,
        )
    else:
        sequence_replay_pools = build_sequence_replay_pools_from_pid_sequences(
            source_pid_sequences,
            fixed_pid_to_sid=fixed_pid_to_sid,
            selected_pair_keys=selected_pair_keys,
            prefix_tokens=args.prefix_tokens,
            window_size=args.window_size,
            replay_history_items=replay_history_items,
        )

    outputs: dict[str, Any] = {}
    for target_train_file in args.target_train_files:
        base_rows = load_json(target_train_file)
        target_dataset_dir = target_train_file.parent
        history_pid_to_sid = load_pid_to_sid(target_dataset_dir)
        if "fixed_target" in target_train_file.name:
            target_pid_to_sid = fixed_pid_to_sid
        else:
            target_pid_to_sid = history_pid_to_sid
        if args.augmentation_mode == "pair":
            augmented_rows = build_augmented_rows(
                base_rows=base_rows,
                selected_pairs=selected_pairs,
                pair_item_pools=pair_item_pools,
                history_pid_to_sid=history_pid_to_sid,
                target_pid_to_sid=target_pid_to_sid,
                max_augmentations_per_pair=args.max_augmentations_per_pair,
                seed=args.seed,
            )
        else:
            augmented_rows = build_sequence_replay_augmented_rows(
                base_rows=base_rows,
                selected_pairs=selected_pairs,
                sequence_replay_pools=sequence_replay_pools,
                history_pid_to_sid=history_pid_to_sid,
                target_pid_to_sid=target_pid_to_sid,
                max_augmentations_per_pair=args.max_augmentations_per_pair,
                seed=args.seed,
            )
        output_path = target_train_file.with_name(
            f"{target_train_file.stem}{args.output_suffix}{target_train_file.suffix}"
        )
        dump_json(output_path, augmented_rows)
        outputs[str(target_train_file)] = {
            "output_path": str(output_path),
            "summary": summarize(augmented_rows),
        }

    meta = {
        "source_dataset_dir": str(args.source_dataset_dir),
        "source_inter_path": str(source_inter_path) if source_inter_path is not None else None,
        "augmentation_mode": args.augmentation_mode,
        "pattern_method": pattern_method,
        "window_size": args.window_size,
        "replay_history_items": replay_history_items,
        "prefix_tokens": args.prefix_tokens,
        "top_k": args.top_k,
        "min_count": args.min_count,
        "pair_score": args.pair_score,
        "max_augmentations_per_pair": args.max_augmentations_per_pair,
        "exclude_self_pair": args.exclude_self_pair,
        "seed": args.seed,
        "num_selected_pairs": len(selected_pairs),
        "top_pairs": [
            {
                "left_prefix": left_prefix,
                "right_prefix": right_prefix,
                "count": int(record["raw_count"]),
                "score": float(record["score"]),
                "weighted_count": float(record["weighted_count"]),
                "pmi": float(record["pmi"]),
                "lift": float(record["lift"]),
                "left_count": int(record["left_count"]),
                "right_count": int(record["right_count"]),
                "pool_size": len(
                    pair_item_pools.get((left_prefix, right_prefix), [])
                    if args.augmentation_mode == "pair"
                    else sequence_replay_pools.get((left_prefix, right_prefix), [])
                ),
            }
            for record in selected_pairs[:50]
            for left_prefix, right_prefix in [record["pair"]]
        ],
        "outputs": outputs,
    }
    meta_path = args.source_dataset_dir / f"prefix_pair_augmentation{args.output_suffix}_meta.json"
    dump_json(meta_path, meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
