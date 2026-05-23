import json
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from util.datacollator import EvalDataCollator
from util.generation_evaluate import build_item_token_codebooks_dynamically, run_generation_evaluation
from util.runtime import (
    load_experiment_config,
    load_model_from_checkpoint,
    load_tokenized_eval_dataset,
    load_tokenizer_from_checkpoint,
)


TOKEN_PATTERN = re.compile(r"<[a-d]_\d+>")


@dataclass
class ExperimentSpec:
    dataset: str
    model_name: str
    label: str
    checkpoint: str


def history_item_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text)) // 4


def bucket_label_from_count(count: int) -> str:
    # Cap long histories at 50 items because longer ones are truncated by evaluation anyway.
    effective = min(count, 50)
    if effective < 10:
        return "0-10"
    if effective < 20:
        return "10-20"
    if effective < 30:
        return "20-30"
    if effective < 40:
        return "30-40"
    return "40-50"


def build_fixed_buckets(dataset_path: str) -> List[Dict]:
    with open(dataset_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    bucket_order = ["0-10", "10-20", "20-30", "30-40", "40-50"]
    bucket_to_indices: Dict[str, List[int]] = {name: [] for name in bucket_order}
    raw_counts: List[int] = []
    effective_counts: List[int] = []

    for idx, example in enumerate(raw):
        count = history_item_count(example["text"])
        raw_counts.append(count)
        effective = min(count, 50)
        effective_counts.append(effective)
        bucket_to_indices[bucket_label_from_count(count)].append(idx)

    buckets = []
    for bucket_id, name in enumerate(bucket_order):
        indices = bucket_to_indices[name]
        bucket_raw = [raw_counts[idx] for idx in indices]
        bucket_eff = [effective_counts[idx] for idx in indices]
        buckets.append(
            {
                "bucket_id": bucket_id,
                "bucket_label": name,
                "indices": indices,
                "num_examples": len(indices),
                "min_raw_history_items": min(bucket_raw) if bucket_raw else None,
                "max_raw_history_items": max(bucket_raw) if bucket_raw else None,
                "min_effective_history_items": min(bucket_eff) if bucket_eff else None,
                "max_effective_history_items": max(bucket_eff) if bucket_eff else None,
                "mean_effective_history_items": round(sum(bucket_eff) / len(bucket_eff), 3) if bucket_eff else None,
            }
        )
    return buckets


def evaluate_experiment(spec: ExperimentSpec, buckets: List[Dict]) -> Dict:
    runtime = load_experiment_config(spec.dataset, spec.model_name)
    dataset_path = os.path.join(runtime["paths"]["dataset_path"], "train_data.json")
    tokenizer = load_tokenizer_from_checkpoint(spec.checkpoint)
    eval_dataset = load_tokenized_eval_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_length=runtime["model"]["max_seq_length"],
        num_proc=runtime["runtime"]["preprocess_num_proc"],
    )

    model = load_model_from_checkpoint(spec.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    collator = EvalDataCollator(tokenizer=tokenizer, max_length=runtime["model"]["max_seq_length"])
    generation_length = len(runtime["tokenizer"]["codeword_nums"])
    item_token_codebooks = build_item_token_codebooks_dynamically(tokenizer, generation_length)

    metrics_by_bucket = []
    for bucket in buckets:
        subset = eval_dataset.select(bucket["indices"])
        dataloader = DataLoader(
            subset,
            batch_size=runtime["training"]["per_device_eval_batch_size"],
            collate_fn=collator,
            shuffle=False,
            drop_last=False,
        )
        metrics = run_generation_evaluation(
            model=model,
            eval_dataloader=dataloader,
            tokenizer=tokenizer,
            generation_length=generation_length,
            num_beams=runtime["evaluation"]["num_beams"],
            k_values=[10],
            item_token_codebooks=item_token_codebooks,
            device=device,
        )
        metrics_by_bucket.append(
            {
                "bucket_id": bucket["bucket_id"],
                "bucket_label": bucket["bucket_label"],
                "num_examples": bucket["num_examples"],
                "min_raw_history_items": bucket["min_raw_history_items"],
                "max_raw_history_items": bucket["max_raw_history_items"],
                "min_effective_history_items": bucket["min_effective_history_items"],
                "max_effective_history_items": bucket["max_effective_history_items"],
                "mean_effective_history_items": bucket["mean_effective_history_items"],
                "NDCG@10": metrics["NDCG@10"],
                "HR@10": metrics["HR@10"],
            }
        )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dataset": spec.dataset,
        "label": spec.label,
        "checkpoint": spec.checkpoint,
        "buckets": metrics_by_bucket,
    }


def main():
    experiments = [
        ExperimentSpec(
            dataset="Beauty_LETTER",
            model_name="llama-rec",
            label="Beauty_LETTER baseline",
            checkpoint=os.path.join(ROOT_DIR, "experiment", "Beauty_LETTER", "llama-rec_20260312_072405", "best_model"),
        ),
        ExperimentSpec(
            dataset="Beauty_LETTER",
            model_name="llama-rec-slot-emb-head",
            label="Beauty_LETTER emb_head",
            checkpoint=os.path.join(
                ROOT_DIR,
                "experiment",
                "Beauty_LETTER_ablation",
                "no_weight",
                "llama-rec-slot_20260324_141752",
                "best_model",
            ),
        ),
        ExperimentSpec(
            dataset="Beauty_rvq",
            model_name="llama-rec",
            label="Beauty_rvq baseline",
            checkpoint=os.path.join(ROOT_DIR, "experiment", "Beauty_rvq", "llama-rec_20260326_145734", "best_model"),
        ),
        ExperimentSpec(
            dataset="Beauty_rvq",
            model_name="llama-rec-slot-emb-head",
            label="Beauty_rvq emb_head",
            checkpoint=os.path.join(ROOT_DIR, "experiment", "Beauty_rvq", "llama-rec-slot_20260326_150458", "best_model"),
        ),
    ]

    output_dir = os.path.join(ROOT_DIR, "tools", "analysis_outputs", "length_bucket_ndcg_fixed")
    os.makedirs(output_dir, exist_ok=True)

    bucket_specs = {}
    for dataset in sorted(set(exp.dataset for exp in experiments)):
        dataset_path = os.path.join(ROOT_DIR, "data", dataset, "train_data.json")
        bucket_specs[dataset] = build_fixed_buckets(dataset_path)

    results = {"bucket_specs": bucket_specs, "results": []}
    for spec in experiments:
        results["results"].append(evaluate_experiment(spec, bucket_specs[spec.dataset]))

    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
