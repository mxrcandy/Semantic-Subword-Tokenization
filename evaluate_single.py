import os
import json
import logging
import argparse
import torch
from torch.utils.data import DataLoader
from util.datacollator import EvalDataCollator
from util.generation_evaluate import (
    build_item_token_codebooks_dynamically,
    resolve_sid_to_pid_mapping_path,
    run_generation_evaluation,
)
from util.input_sid_pooling import build_input_sid_pooler
from util.runtime import (
    load_experiment_config,
    load_model_from_checkpoint,
    resolve_data_split_path,
    load_tokenized_eval_dataset,
    load_tokenizer_from_checkpoint,
    setup_logging,
)
logging.basicConfig(level=logging.INFO)
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

# Main 函数
def main():
    # 获取配置文件路径
    parser = argparse.ArgumentParser(description="Train a LlamaRec model using a YAML config file.")
    parser.add_argument("--dataset", type=str, default='Beauty_LETTER')
    parser.add_argument("--model_name", type=str, default='llama-rec')
    parser.add_argument("--checkpoint", type=str, default='experiment/Beauty_LETTER/llama-rec_20251225_115133/checkpoint-2100')
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Evaluation split.",
    )
    parser.add_argument(
        "--metric_key_prefix",
        type=str,
        default=None,
        help="Optional metric prefix override. If omitted, split-based evaluation uses the split name.",
    )
    parser.add_argument(
        "--generation_constraint",
        type=str,
        default=None,
        choices=["level_only", "full_trie", "abcd", "slot", "trie"],
        help="Generation constraint override. level_only is the a/b/c/d codebook constraint; full_trie follows valid SID prefixes.",
    )
    args = parser.parse_args()

    runtime_config = load_experiment_config(args.dataset, args.model_name)
    config_data = runtime_config["config_data"]
    paths_config = runtime_config["paths"]
    model_config = runtime_config["model"]
    runtime_options = runtime_config["runtime"]
    training_config = runtime_config["training"]
    tokenizer_config = runtime_config["tokenizer"]
    evaluation_config = runtime_config["evaluation"]

    # 使用从配置中读取的参数
    dataset_path = resolve_data_split_path(
        paths_config,
        args.eval_split,
        fallback_to_legacy_eval=False,
    )
    max_seq_length = model_config['max_seq_length']
    generation_length = len(tokenizer_config['codeword_nums'])
    preprocess_num_proc = runtime_options["preprocess_num_proc"]

    checkpoint_path = args.checkpoint
    output_dir = os.path.dirname(checkpoint_path)
    resolved_split_name = args.eval_split

    setup_logging(
        output_dir,
        config_data,
        log_filename=f"evaluation_{resolved_split_name}.log",
    )
    logging.info("Resolved eval dataset path (%s): %s", args.eval_split, dataset_path)

    tokenizer = load_tokenizer_from_checkpoint(checkpoint_path=checkpoint_path)

    # 数据集加载
    test_dataset = load_tokenized_eval_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        max_length=max_seq_length,
        num_proc=preprocess_num_proc,
    )

    # 直接从checkpoint加载模型
    try:
        model = load_model_from_checkpoint(checkpoint_path)
    except Exception as e:
        logging.error(f"Failed to load model from checkpoint: {e}")
        raise
    
    # 将模型移到GPU（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_sid_pooler = build_input_sid_pooler(model_config, tokenizer)

    # DataCollator实例化
    test_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=training_config['per_device_eval_batch_size'],
        collate_fn=test_collator,
        shuffle=False,
        drop_last=False,
    )

    # 评估
    logging.info("Starting custom evaluation...")
    model.eval()
    
    item_token_codebooks = build_item_token_codebooks_dynamically(tokenizer, generation_length)
    logging.info(f"Starting manual evaluation with num_beams={evaluation_config['num_beams']}...")
    evaluation_mode = evaluation_config.get("evaluation_mode", "sid")
    generation_constraint = args.generation_constraint or evaluation_config.get(
        "generation_constraint",
        evaluation_config.get("constraint_mode", "level_only"),
    )
    logging.info("Generation constraint: %s", generation_constraint)
    sid_to_pid_mapping_path = evaluation_config.get("sid_to_pid_mapping_path")
    if sid_to_pid_mapping_path is None:
        sid_to_pid_mapping_path = resolve_sid_to_pid_mapping_path(paths_config["dataset_path"])
    metric_key_prefix = args.metric_key_prefix
    if metric_key_prefix is None and args.eval_split != "train":
        metric_key_prefix = args.eval_split
    metrics = run_generation_evaluation(
        model=model,
        eval_dataloader=test_dataloader,
        tokenizer=tokenizer,
        generation_length=generation_length,
        num_beams=evaluation_config['num_beams'],
        k_values=evaluation_config['eval_k_values'],
        item_token_codebooks=item_token_codebooks,
        device=device,
        metric_key_prefix=metric_key_prefix,
        evaluation_mode=evaluation_mode,
        sid_to_pid_mapping_path=sid_to_pid_mapping_path,
        pid_selection_strategy=evaluation_config.get("pid_selection_strategy", "most_popular_originally"),
        pid_random_seed=evaluation_config.get("pid_random_seed", 42),
        generation_constraint=generation_constraint,
        input_sid_pooler=input_sid_pooler,
    )

    results_payload = {
        "dataset": args.dataset,
        "model_name": args.model_name,
        "checkpoint": checkpoint_path,
        "eval_split": args.eval_split,
        "resolved_eval_path": dataset_path,
        "metric_key_prefix": metric_key_prefix,
        "generation_constraint": generation_constraint,
        "metrics": metrics,
    }
    metrics_output_path = os.path.join(output_dir, f"evaluation_{resolved_split_name}_metrics.json")
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, ensure_ascii=False, indent=2)

    logging.info(f"Evaluation results: {metrics}")
    logging.info("Saved evaluation metrics to %s", metrics_output_path)
    logging.info("All operations complete!")

if __name__ == "__main__":
    main()
