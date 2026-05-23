#!/usr/bin/env python3
"""Token-level intra-item attention workload analysis.

This script diagnoses the mechanism behind fixed-length semantic IDs from a
more direct angle than item-level Intra/Inter share.  It measures how much
token-level attention is spent on reconstructing tokens inside the same item
span, and how many intra-item token pairs the encoder has to maintain.

Example:

CUDA_VISIBLE_DEVICES=1 python tools/token_level_attention_workload_analysis.py \
  --run Fixed::experiment/Yelp_rq_kmeans_d_reassign_seq2seq/t5-rec_20260517_063311/best_model::data/Yelp_rq_kmeans_d_reassign/test_seq2seq_letter.json \
  --run Final::experiment/Yelp_rq_kmeans_d_reassign_varlen_bpe_strict_top128_f20_s10_seq2seq_behseq_sw3_top512_c5_aug50/t5-rec_20260517_081429/best_model::data/Yelp_rq_kmeans_d_reassign_varlen_bpe_strict_top128_f20_s10/test_seq2seq_letter_fixed_target.json \
  --max-examples 0 \
  --batch-size 4 \
  --output-dir tools/analysis_outputs/token_level_attention_workload/yelp_rqk
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from transformers import AutoTokenizer

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from tools.item_span_attention_analysis import (  # noqa: E402
    ExampleSpans,
    RunSpec,
    choose_indices,
    choose_per_run_indices,
    covered_slots,
    load_json_records,
    parse_index_selector,
    parse_run_specs,
    prepare_examples,
    shifted_spans_for_padded_batch,
)
from util.runtime import load_model_from_checkpoint  # noqa: E402


TOKEN_FLOW_COLUMNS = [
    "atom_to_atom_same_item_mass",
    "atom_to_pattern_same_item_mass",
    "pattern_to_atom_same_item_mass",
    "pattern_to_pattern_same_item_mass",
]


@dataclass
class WorkloadAccumulator:
    count: int = 0
    item_count_sum: float = 0.0
    semantic_token_sum: float = 0.0
    covered_semantic_token_sum: float = 0.0
    avg_span_len_sum: float = 0.0
    total_intra_token_pairs_sum: float = 0.0
    avg_intra_token_pairs_per_item_sum: float = 0.0
    avg_intra_other_candidates_per_query_sum: float = 0.0
    same_item_mass_sum: float = 0.0
    same_item_self_mass_sum: float = 0.0
    same_item_other_mass_sum: float = 0.0
    same_item_other_total_mass_sum: float = 0.0
    inter_item_mass_sum: float = 0.0
    inter_item_total_mass_sum: float = 0.0
    item_span_mass_sum: float = 0.0
    same_item_share_sum: float = 0.0
    same_item_other_share_sum: float = 0.0
    inter_item_share_sum: float = 0.0
    attention_weighted_intra_pair_workload_sum: float = 0.0
    attention_weighted_inter_item_workload_sum: float = 0.0
    inter_per_intra_workload_ratio_sum: float = 0.0
    intra_per_inter_workload_ratio_sum: float = 0.0
    atom_query_count_sum: float = 0.0
    pattern_query_count_sum: float = 0.0
    atom_query_same_item_other_mass_sum: float = 0.0
    atom_query_inter_item_mass_sum: float = 0.0
    pattern_query_same_item_other_mass_sum: float = 0.0
    pattern_query_inter_item_mass_sum: float = 0.0
    atom_to_atom_same_item_mass_sum: float = 0.0
    atom_to_pattern_same_item_mass_sum: float = 0.0
    pattern_to_atom_same_item_mass_sum: float = 0.0
    pattern_to_pattern_same_item_mass_sum: float = 0.0
    atom_to_atom_same_item_total_mass_sum: float = 0.0
    atom_to_pattern_same_item_total_mass_sum: float = 0.0
    pattern_to_atom_same_item_total_mass_sum: float = 0.0
    pattern_to_pattern_same_item_total_mass_sum: float = 0.0

    def add(self, metrics: Dict[str, float]) -> None:
        self.count += 1
        for key, value in metrics.items():
            attr = f"{key}_sum"
            if hasattr(self, attr):
                setattr(self, attr, getattr(self, attr) + float(value))

    def as_row(self, label: str, layer: str, head: str) -> Dict[str, object]:
        denom = max(self.count, 1)

        def avg(name: str) -> float:
            return getattr(self, f"{name}_sum") / denom

        atom_count = self.atom_query_count_sum
        pattern_count = self.pattern_query_count_sum
        return {
            "label": label,
            "layer": layer,
            "head": head,
            "examples": self.count,
            "avg_history_items": avg("item_count"),
            "avg_semantic_tokens": avg("semantic_token"),
            "avg_covered_semantic_tokens": avg("covered_semantic_token"),
            "avg_item_span_len": avg("avg_span_len"),
            "total_intra_token_pairs": avg("total_intra_token_pairs"),
            "avg_intra_token_pairs_per_item": avg("avg_intra_token_pairs_per_item"),
            "avg_intra_other_candidates_per_query": avg("avg_intra_other_candidates_per_query"),
            "same_item_mass": avg("same_item_mass"),
            "same_item_self_mass": avg("same_item_self_mass"),
            "same_item_other_mass": avg("same_item_other_mass"),
            "same_item_other_total_mass": avg("same_item_other_total_mass"),
            "inter_item_mass": avg("inter_item_mass"),
            "inter_item_total_mass": avg("inter_item_total_mass"),
            "item_span_mass": avg("item_span_mass"),
            "same_item_share": avg("same_item_share"),
            "same_item_other_share": avg("same_item_other_share"),
            "inter_item_share": avg("inter_item_share"),
            "attention_weighted_intra_pair_workload": avg("attention_weighted_intra_pair_workload"),
            "attention_weighted_inter_item_workload": avg("attention_weighted_inter_item_workload"),
            "inter_per_intra_workload_ratio": avg("inter_per_intra_workload_ratio"),
            "intra_per_inter_workload_ratio": avg("intra_per_inter_workload_ratio"),
            "atom_query_count": atom_count / denom,
            "pattern_query_count": pattern_count / denom,
            "atom_query_same_item_other_mass": (
                self.atom_query_same_item_other_mass_sum / atom_count if atom_count > 0 else 0.0
            ),
            "atom_query_inter_item_mass": (
                self.atom_query_inter_item_mass_sum / atom_count if atom_count > 0 else 0.0
            ),
            "atom_to_atom_same_item_mass_per_atom_query": (
                self.atom_to_atom_same_item_total_mass_sum / atom_count if atom_count > 0 else 0.0
            ),
            "atom_to_pattern_same_item_mass_per_atom_query": (
                self.atom_to_pattern_same_item_total_mass_sum / atom_count if atom_count > 0 else 0.0
            ),
            "pattern_query_same_item_other_mass": (
                self.pattern_query_same_item_other_mass_sum / pattern_count if pattern_count > 0 else 0.0
            ),
            "pattern_query_inter_item_mass": (
                self.pattern_query_inter_item_mass_sum / pattern_count if pattern_count > 0 else 0.0
            ),
            "pattern_to_atom_same_item_mass_per_pattern_query": (
                self.pattern_to_atom_same_item_total_mass_sum / pattern_count if pattern_count > 0 else 0.0
            ),
            "pattern_to_pattern_same_item_mass_per_pattern_query": (
                self.pattern_to_pattern_same_item_total_mass_sum / pattern_count if pattern_count > 0 else 0.0
            ),
            "atom_to_atom_same_item_mass": avg("atom_to_atom_same_item_mass"),
            "atom_to_pattern_same_item_mass": avg("atom_to_pattern_same_item_mass"),
            "pattern_to_atom_same_item_mass": avg("pattern_to_atom_same_item_mass"),
            "pattern_to_pattern_same_item_mass": avg("pattern_to_pattern_same_item_mass"),
            "atom_to_atom_same_item_total_mass": avg("atom_to_atom_same_item_total_mass"),
            "atom_to_pattern_same_item_total_mass": avg("atom_to_pattern_same_item_total_mass"),
            "pattern_to_atom_same_item_total_mass": avg("pattern_to_atom_same_item_total_mass"),
            "pattern_to_pattern_same_item_total_mass": avg("pattern_to_pattern_same_item_total_mass"),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="LABEL::CHECKPOINT::DATASET_JSON",
        help="One run to analyze. Can be repeated.",
    )
    parser.add_argument("--label", default=None, help="Single-run label.")
    parser.add_argument("--checkpoint", default=None, help="Single-run checkpoint path.")
    parser.add_argument("--dataset-path", default=None, help="Single-run seq2seq JSON path.")
    parser.add_argument("--max-examples", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-source-length", type=int, default=512)
    parser.add_argument("--min-history-items", type=int, default=2)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--heads", default="mean")
    parser.add_argument(
        "--sample-mode",
        choices=["shared_indices", "per_run"],
        default="shared_indices",
        help="Use shared sampled example indices across runs when dataset lengths match.",
    )
    parser.add_argument("--save-per-example", action="store_true")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT_DIR, "tools", "analysis_outputs", "token_level_attention_workload"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def batch_iter(examples: Sequence[ExampleSpans], batch_size: int) -> Iterable[List[ExampleSpans]]:
    for start in range(0, len(examples), batch_size):
        yield list(examples[start : start + batch_size])


def collate_examples(examples: Sequence[ExampleSpans], tokenizer) -> Dict[str, torch.Tensor]:
    features = [{"input_ids": example.input_ids, "attention_mask": example.attention_mask} for example in examples]
    return tokenizer.pad(features, padding=True, return_tensors="pt")


def token_kind(token: str) -> str:
    slots = covered_slots(token)
    if len(slots) == 1:
        return "atom"
    if len(slots) > 1:
        return "pattern"
    return "other"


def safe_div(numerator: torch.Tensor, denominator: torch.Tensor | float) -> torch.Tensor:
    if not isinstance(denominator, torch.Tensor):
        denominator = numerator.new_tensor(float(denominator))
    return numerator / torch.clamp(denominator, min=1.0e-12)


def token_workload_metrics(
    attn: torch.Tensor,
    example: ExampleSpans,
    padded_spans: Sequence[Tuple[int, int]],
) -> Dict[str, float]:
    """Compute token-level intra-item workload metrics for one attention matrix."""
    item_count = len(example.spans)
    span_lens = [end - start for start, end in example.spans]
    total_query_tokens = max(sum(span_lens), 1)
    total_intra_pairs = sum(length * max(length - 1, 0) for length in span_lens)
    avg_intra_pairs_per_item = total_intra_pairs / max(item_count, 1)
    avg_other_candidates = total_intra_pairs / total_query_tokens

    all_item_positions: List[int] = []
    for start, end in padded_spans:
        all_item_positions.extend(range(start, end))

    same_item_mass = attn.new_tensor(0.0)
    same_item_self_mass = attn.new_tensor(0.0)
    same_item_other_mass = attn.new_tensor(0.0)
    inter_item_mass = attn.new_tensor(0.0)
    item_span_mass = attn.new_tensor(0.0)
    weighted_intra_pair_workload = attn.new_tensor(0.0)
    weighted_inter_item_workload = attn.new_tensor(0.0)

    atom_query_count = 0
    pattern_query_count = 0
    atom_query_same_other = attn.new_tensor(0.0)
    atom_query_inter = attn.new_tensor(0.0)
    pattern_query_same_other = attn.new_tensor(0.0)
    pattern_query_inter = attn.new_tensor(0.0)
    flow = {key: attn.new_tensor(0.0) for key in TOKEN_FLOW_COLUMNS}

    for item_idx, ((orig_start, orig_end), (pad_start, pad_end)) in enumerate(zip(example.spans, padded_spans)):
        query_positions = list(range(pad_start, pad_end))
        same_positions = query_positions
        inter_positions = [
            pos
            for other_idx, (other_start, other_end) in enumerate(padded_spans)
            if other_idx != item_idx
            for pos in range(other_start, other_end)
        ]
        length = len(query_positions)
        if length == 0:
            continue

        same_block = attn[query_positions][:, same_positions]
        row_same = same_block.sum(dim=1)
        row_self = torch.diagonal(same_block)
        row_other = row_same - row_self
        if inter_positions:
            row_inter = attn[query_positions][:, inter_positions].sum(dim=1)
        else:
            row_inter = attn.new_zeros((length,))

        same_item_mass += row_same.sum()
        same_item_self_mass += row_self.sum()
        same_item_other_mass += row_other.sum()
        inter_item_mass += row_inter.sum()
        item_span_mass += row_same.sum() + row_inter.sum()
        weighted_intra_pair_workload += (row_other * max(length - 1, 0)).sum()
        weighted_inter_item_workload += (row_inter * max(total_query_tokens - length, 0)).sum()

        orig_positions = list(range(orig_start, orig_end))
        for local_query_idx, orig_query_pos in enumerate(orig_positions):
            query_kind = token_kind(example.tokens[orig_query_pos])
            if query_kind == "atom":
                atom_query_count += 1
                atom_query_same_other += row_other[local_query_idx]
                atom_query_inter += row_inter[local_query_idx]
            elif query_kind == "pattern":
                pattern_query_count += 1
                pattern_query_same_other += row_other[local_query_idx]
                pattern_query_inter += row_inter[local_query_idx]

            for local_key_idx, orig_key_pos in enumerate(orig_positions):
                if local_key_idx == local_query_idx:
                    continue
                key_kind = token_kind(example.tokens[orig_key_pos])
                flow_key = f"{query_kind}_to_{key_kind}_same_item_mass"
                if flow_key in flow:
                    flow[flow_key] += same_block[local_query_idx, local_key_idx]

    denom_tokens = attn.new_tensor(float(total_query_tokens))
    same_avg = safe_div(same_item_mass, denom_tokens)
    self_avg = safe_div(same_item_self_mass, denom_tokens)
    other_avg = safe_div(same_item_other_mass, denom_tokens)
    inter_avg = safe_div(inter_item_mass, denom_tokens)
    item_mass_avg = safe_div(item_span_mass, denom_tokens)
    share_denom = torch.clamp(same_avg + inter_avg, min=1.0e-12)
    other_share_denom = torch.clamp(other_avg + inter_avg, min=1.0e-12)

    return {
        "item_count": float(item_count),
        "semantic_token": float(example.semantic_tokens),
        "covered_semantic_token": float(example.covered_semantic_tokens),
        "avg_span_len": sum(span_lens) / max(len(span_lens), 1),
        "total_intra_token_pairs": float(total_intra_pairs),
        "avg_intra_token_pairs_per_item": float(avg_intra_pairs_per_item),
        "avg_intra_other_candidates_per_query": float(avg_other_candidates),
        "same_item_mass": float(same_avg.item()),
        "same_item_self_mass": float(self_avg.item()),
        "same_item_other_mass": float(other_avg.item()),
        "same_item_other_total_mass": float(same_item_other_mass.item()),
        "inter_item_mass": float(inter_avg.item()),
        "inter_item_total_mass": float(inter_item_mass.item()),
        "item_span_mass": float(item_mass_avg.item()),
        "same_item_share": float((same_avg / share_denom).item()),
        "same_item_other_share": float((other_avg / other_share_denom).item()),
        "inter_item_share": float((inter_avg / share_denom).item()),
        "attention_weighted_intra_pair_workload": float(
            safe_div(weighted_intra_pair_workload, denom_tokens).item()
        ),
        "attention_weighted_inter_item_workload": float(
            safe_div(weighted_inter_item_workload, denom_tokens).item()
        ),
        "inter_per_intra_workload_ratio": float(
            safe_div(weighted_inter_item_workload, weighted_intra_pair_workload).item()
        ),
        "intra_per_inter_workload_ratio": float(
            safe_div(weighted_intra_pair_workload, weighted_inter_item_workload).item()
        ),
        "atom_query_count": float(atom_query_count),
        "pattern_query_count": float(pattern_query_count),
        "atom_query_same_item_other_mass": float(atom_query_same_other.item()),
        "atom_query_inter_item_mass": float(atom_query_inter.item()),
        "pattern_query_same_item_other_mass": float(pattern_query_same_other.item()),
        "pattern_query_inter_item_mass": float(pattern_query_inter.item()),
        **{key: float(safe_div(value, denom_tokens).item()) for key, value in flow.items()},
        **{key.replace("_mass", "_total_mass"): float(value.item()) for key, value in flow.items()},
    }


def analyze_run(
    *,
    spec: RunSpec,
    records: Sequence[Dict[str, object]],
    indices: Sequence[int],
    args: argparse.Namespace,
) -> Dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(spec.checkpoint, trust_remote_code=False)
    examples, skipped = prepare_examples(
        records=records,
        indices=indices,
        tokenizer=tokenizer,
        max_source_length=args.max_source_length,
        min_history_items=args.min_history_items,
    )
    if not examples:
        raise RuntimeError(f"No usable examples for run {spec.label}")

    model = load_model_from_checkpoint(spec.checkpoint)
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "eager"
    model.config.output_attentions = True
    model.to(args.device)
    model.eval()
    encoder = model.get_encoder() if hasattr(model, "get_encoder") else model

    device = torch.device(args.device)
    layer_accumulators: Dict[Tuple[int, str], WorkloadAccumulator] = {}
    selected_layers: Optional[List[int]] = None
    selected_heads: Optional[List[int]] = None
    num_layers = None
    num_heads = None
    per_example_rows: List[Dict[str, object]] = []

    with torch.no_grad():
        for batch_examples in batch_iter(examples, args.batch_size):
            batch = collate_examples(batch_examples, tokenizer)
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = encoder(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_attentions=True,
                return_dict=True,
            )
            attentions = outputs.attentions
            if attentions is None:
                raise RuntimeError("Model did not return encoder attentions.")

            if selected_layers is None:
                num_layers = len(attentions)
                num_heads = int(attentions[0].shape[1])
                selected_layers = parse_index_selector(args.layers, num_layers)
                if selected_layers is None:
                    selected_layers = list(range(num_layers))
                selected_heads = parse_index_selector(args.heads, num_heads)

            assert selected_layers is not None
            for layer_idx in selected_layers:
                layer_attn = attentions[layer_idx].detach().float().cpu()
                for batch_idx, example in enumerate(batch_examples):
                    padded_spans = shifted_spans_for_padded_batch(
                        example,
                        batch["attention_mask"][batch_idx].detach().cpu(),
                    )
                    if selected_heads is None:
                        head_attn = layer_attn[batch_idx].mean(dim=0)
                        metrics = token_workload_metrics(head_attn, example, padded_spans)
                        key = (layer_idx, "mean")
                        layer_accumulators.setdefault(key, WorkloadAccumulator())
                        layer_accumulators[key].add(metrics)
                        if args.save_per_example:
                            per_example_rows.append({
                                "label": spec.label,
                                "example_index": example.example_index,
                                "layer": layer_idx,
                                "head": "mean",
                                **metrics,
                            })
                    else:
                        for head_idx in selected_heads:
                            metrics = token_workload_metrics(
                                layer_attn[batch_idx, head_idx],
                                example,
                                padded_spans,
                            )
                            key = (layer_idx, str(head_idx))
                            layer_accumulators.setdefault(key, WorkloadAccumulator())
                            layer_accumulators[key].add(metrics)
                            if args.save_per_example:
                                per_example_rows.append({
                                    "label": spec.label,
                                    "example_index": example.example_index,
                                    "layer": layer_idx,
                                    "head": head_idx,
                                    **metrics,
                                })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    layer_rows = [
        accumulator.as_row(spec.label, str(layer_idx), str(head))
        for (layer_idx, head), accumulator in sorted(layer_accumulators.items(), key=lambda item: (item[0][0], item[0][1]))
    ]
    mean_acc = WorkloadAccumulator()
    for acc in layer_accumulators.values():
        for field in mean_acc.__dataclass_fields__:
            setattr(mean_acc, field, getattr(mean_acc, field) + getattr(acc, field))

    return {
        "spec": asdict(spec),
        "num_records": len(records),
        "sampled_indices": len(indices),
        "usable_examples": len(examples),
        "skipped": skipped,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "layer_rows": layer_rows,
        "summary_row": mean_acc.as_row(spec.label, "mean", "mean"),
        "per_example_rows": per_example_rows,
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_delta_csv(path: Path, summary_rows: Sequence[Dict[str, object]]) -> None:
    if len(summary_rows) < 2:
        return
    baseline = summary_rows[0]
    metric_keys = [
        key
        for key, value in baseline.items()
        if key not in {"label", "layer", "head"} and isinstance(value, (int, float))
    ]
    rows = []
    for row in summary_rows[1:]:
        out = {
            "baseline": baseline["label"],
            "label": row["label"],
            "layer": row["layer"],
            "head": row["head"],
        }
        for key in metric_keys:
            base_value = float(baseline[key])
            value = float(row[key])
            out[f"delta_{key}"] = value - base_value
            out[f"pct_delta_{key}"] = ((value / base_value - 1.0) * 100.0) if abs(base_value) > 1.0e-12 else math.nan
        rows.append(out)
    write_csv(path, rows)


def main() -> None:
    args = parse_args()
    specs = parse_run_specs(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets = [load_json_records(spec.dataset_path) for spec in specs]
    shared_indices = choose_indices(
        datasets=datasets,
        max_examples=args.max_examples,
        seed=args.seed,
        sample_mode=args.sample_mode,
    )

    all_layer_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []
    all_per_example_rows: List[Dict[str, object]] = []
    manifest = {
        "args": vars(args),
        "runs": [],
    }

    for spec, records in zip(specs, datasets):
        indices = shared_indices
        if indices is None:
            indices = choose_per_run_indices(records, args.max_examples, args.seed)
        result = analyze_run(spec=spec, records=records, indices=indices, args=args)
        all_layer_rows.extend(result["layer_rows"])
        summary_rows.append(result["summary_row"])
        all_per_example_rows.extend(result["per_example_rows"])
        manifest["runs"].append({
            "spec": result["spec"],
            "num_records": result["num_records"],
            "sampled_indices": result["sampled_indices"],
            "usable_examples": result["usable_examples"],
            "skipped": result["skipped"],
            "num_layers": result["num_layers"],
            "num_heads": result["num_heads"],
        })

    write_csv(output_dir / "layer_metrics.csv", all_layer_rows)
    write_csv(output_dir / "summary_metrics.csv", summary_rows)
    write_delta_csv(output_dir / "delta_vs_first_run.csv", summary_rows)
    if args.save_per_example:
        write_csv(output_dir / "per_example_metrics.csv", all_per_example_rows)
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(json.dumps({"summary_rows": summary_rows, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
