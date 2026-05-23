#!/usr/bin/env python3
"""Build seq2seq sliding-window splits for fixed-target SID experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def sid_text(index: dict[str, list[str]], item_id: int | str) -> str:
    return "".join(index[str(item_id)])


def build_history_text(index: dict[str, list[str]], history_items: list[int | str]) -> str:
    return "".join(sid_text(index, item_id) for item_id in history_items)


def trim_history_items(history_items: list[int | str], max_history_items: int) -> list[int | str]:
    if max_history_items > 0:
        return history_items[-max_history_items:]
    return history_items


def build_train_samples(
    inter: dict[str, list[int]],
    history_index: dict[str, list[str]],
    target_index: dict[str, list[str]],
    min_history_len: int,
    max_history_items: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped_short = 0
    for user_id, items in inter.items():
        if len(items) < min_history_len + 3:
            skipped_short += 1
            continue
        for target_pos in range(min_history_len, len(items) - 2):
            history_items = trim_history_items(items[:target_pos], max_history_items)
            target_item = items[target_pos]
            rows.append(
                {
                    "text": build_history_text(history_index, history_items),
                    "sid_ground_truth": [sid_text(target_index, target_item)],
                    "pid_ground_truth": [str(target_item)],
                    "user_id": str(user_id),
                }
            )
    return rows


def build_leave_one_out_samples(
    inter: dict[str, list[int]],
    history_index: dict[str, list[str]],
    target_index: dict[str, list[str]],
    split: str,
    max_history_items: int,
) -> list[dict[str, Any]]:
    if split not in {"val", "test"}:
        raise ValueError(split)

    rows: list[dict[str, Any]] = []
    target_offset = -2 if split == "val" else -1
    for user_id, items in inter.items():
        if len(items) < 3:
            continue
        target_item = items[target_offset]
        history_items = trim_history_items(items[:target_offset], max_history_items)
        rows.append(
            {
                "text": build_history_text(history_index, history_items),
                "sid_ground_truth": [sid_text(target_index, target_item)],
                "pid_ground_truth": [str(target_item)],
                "user_id": str(user_id),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    history_token_counts = [row["text"].count("<") for row in rows]
    target_token_counts = [row["sid_ground_truth"][0].count("<") for row in rows]
    if not rows:
        return {"rows": 0}
    return {
        "rows": len(rows),
        "history_tokens_avg": sum(history_token_counts) / len(history_token_counts),
        "history_tokens_max": max(history_token_counts),
        "target_tokens_avg": sum(target_token_counts) / len(target_token_counts),
        "target_tokens_max": max(target_token_counts),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inter-path", type=Path, required=True)
    parser.add_argument("--history-index-path", type=Path, required=True)
    parser.add_argument("--target-index-path", type=Path, required=True)
    parser.add_argument("--prefix", type=str, required=True)
    parser.add_argument("--min-history-len", type=int, default=2)
    parser.add_argument(
        "--max-history-items",
        type=int,
        default=0,
        help="Keep only the last N history items before SID string construction; <=0 keeps full history.",
    )
    parser.add_argument("--meta-name", type=str, default="seq2seq_sliding_meta.json")
    args = parser.parse_args()

    inter = load_json(args.inter_path)
    history_index = load_json(args.history_index_path)
    target_index = load_json(args.target_index_path)

    train_rows = build_train_samples(
        inter=inter,
        history_index=history_index,
        target_index=target_index,
        min_history_len=args.min_history_len,
        max_history_items=args.max_history_items,
    )
    val_rows = build_leave_one_out_samples(
        inter=inter,
        history_index=history_index,
        target_index=target_index,
        split="val",
        max_history_items=args.max_history_items,
    )
    test_rows = build_leave_one_out_samples(
        inter=inter,
        history_index=history_index,
        target_index=target_index,
        split="test",
        max_history_items=args.max_history_items,
    )

    train_path = args.output_dir / f"train_{args.prefix}.json"
    val_path = args.output_dir / f"val_{args.prefix}.json"
    test_path = args.output_dir / f"test_{args.prefix}.json"
    dump_json(train_path, train_rows)
    dump_json(val_path, val_rows)
    dump_json(test_path, test_rows)

    meta = {
        "method": "seq2seq_sliding_window_fixed_target",
        "min_history_len": args.min_history_len,
        "max_history_items": args.max_history_items,
        "inter_path": str(args.inter_path),
        "history_index_path": str(args.history_index_path),
        "target_index_path": str(args.target_index_path),
        "splits": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
        },
        "stats": {
            "train": summarize(train_rows),
            "val": summarize(val_rows),
            "test": summarize(test_rows),
        },
    }
    dump_json(args.output_dir / args.meta_name, meta)
    print(json.dumps(meta["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
