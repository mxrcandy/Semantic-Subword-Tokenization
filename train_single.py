import os
import logging
import argparse
from datetime import datetime
from torch.utils.data import DataLoader
from transformers import EarlyStoppingCallback
from transformers import (
    Trainer,
    TrainingArguments,
)
from util.datacollator import (
    EvalDataCollator,
    Seq2SeqDataCollator,
    TrainDataCollator,
)
from util.generation_evaluate import (
    build_item_token_codebooks_dynamically,
    resolve_sid_to_pid_mapping_path,
    run_generation_evaluation,
)
from util.input_sid_pooling import build_input_sid_pooler
from util.runtime import (
    build_model_from_config,
    create_tokenizer_from_config,
    load_experiment_config,
    resolve_data_split_path,
    load_tokenized_eval_dataset,
    load_tokenized_loss_eval_dataset,
    load_tokenized_seq2seq_dataset,
    load_tokenized_train_dataset,
    setup_logging,
)
import warnings
# 忽略特定的 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.trainer")
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '1')


class InputPoolingTrainer(Trainer):
    def __init__(self, input_sid_pooler=None, **kwargs):
        super().__init__(**kwargs)
        self.input_sid_pooler = input_sid_pooler

    def _maybe_pool_inputs(self, model, inputs):
        if self.input_sid_pooler is None or "input_ids" not in inputs:
            return inputs
        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")
        attention_mask = inputs.pop("attention_mask", None)
        segments = inputs.pop("input_sid_pool_segments", None)
        pooled = self.input_sid_pooler.pool_batch(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            segments=segments,
        )
        inputs["inputs_embeds"] = pooled["inputs_embeds"]
        inputs["attention_mask"] = pooled["attention_mask"]
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = self._maybe_pool_inputs(model, inputs)
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss

        if return_outputs:
            return loss, outputs
        return loss


class CustomTrainer(InputPoolingTrainer):
    def __init__(self, eval_collator, generation_config_params, **kwargs):
        super().__init__(**kwargs)
        self.eval_collator = eval_collator
        self.generation_config_params = generation_config_params
        self.gen_len = generation_config_params['generation_length']
        self.num_beams = generation_config_params['num_beams']
        self.k_values = generation_config_params['k_values']
        self.item_token_codebooks = generation_config_params['item_token_codebooks']

    # 重写 evaluate 方法
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=self.args.eval_batch_size,
            collate_fn=self.eval_collator,
            shuffle=False,
            drop_last=False,
        )
        model = self._wrap_model(self.model, training=False, dataloader=eval_dataloader)
        model.eval()
        logging.info("***** Running Custom Evaluation (Generation) *****")
        metrics = run_generation_evaluation(
            model=model,
            eval_dataloader=eval_dataloader,
            tokenizer=self.processing_class,
            generation_length=self.gen_len,
            num_beams=self.num_beams,
            k_values=self.k_values,
            item_token_codebooks=self.item_token_codebooks,
            device=self.args.device,
            metric_key_prefix=metric_key_prefix,
            evaluation_mode=self.generation_config_params["evaluation_mode"],
            sid_to_pid_mapping_path=self.generation_config_params["sid_to_pid_mapping_path"],
            pid_selection_strategy=self.generation_config_params["pid_selection_strategy"],
            pid_random_seed=self.generation_config_params["pid_random_seed"],
            generation_constraint=self.generation_config_params["generation_constraint"],
            input_sid_pooler=self.input_sid_pooler,
        )
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        logging.info(f"Evaluation metrics: {metrics}")
        return metrics

# Main 函数
def main():
    # 获取配置文件路径
    parser = argparse.ArgumentParser(description="Train a LlamaRec model using a YAML config file.")
    parser.add_argument("--dataset", type=str, default='Beauty')
    parser.add_argument("--model_name", type=str, default='llama-rec')
    parser.add_argument(
        "--eval_split",
        type=str,
        default=None,
        choices=["train", "val"],
        help="Validation split. Overrides evaluation.eval_split when provided.",
    )
    parser.add_argument(
        "--validation_objective",
        type=str,
        default=None,
        choices=["generation", "loss"],
        help="Validation objective. Overrides evaluation.validation_objective when provided.",
    )
    parser.add_argument(
        "--generation_constraint",
        type=str,
        default=None,
        choices=["level_only", "full_trie", "abcd", "slot", "trie"],
        help="Generation constraint for validation. Overrides evaluation.generation_constraint when provided.",
    )
    parser.add_argument(
        "--semantic_embedding_init",
        type=str,
        default=None,
        choices=["none", "input", "input_only", "output", "output_only", "input_output", "both"],
        help="Semantic codebook initialization branch. Overrides model.semantic_embedding_init when provided.",
    )
    parser.add_argument(
        "--semantic_embedding_init_path",
        type=str,
        default=None,
        help="Path to quantizer.pkl for semantic initialization. Defaults to paths.dataset_path/quantizer.pkl.",
    )
    parser.add_argument(
        "--semantic_embedding_projection",
        type=str,
        default=None,
        choices=["random", "pca"],
        help="Projection used to map quantizer codebooks into model hidden space.",
    )
    parser.add_argument(
        "--semantic_embedding_source_path",
        type=str,
        default=None,
        help="Embedding source path for semantic_embedding_projection=pca. Defaults to paths.dataset_path/*.emb.npy.",
    )
    parser.add_argument(
        "--tie_input_output_embeddings",
        action="store_true",
        help="Tie input embeddings and output lm_head weights for non-slot models.",
    )
    parser.add_argument(
        "--semantic_embedding_freeze_scope",
        type=str,
        default=None,
        choices=["none", "input", "input_only", "output", "output_only", "both"],
        help="Freeze semantic token rows in input embeddings, output head, both, or none.",
    )
    args = parser.parse_args()

    runtime_config = load_experiment_config(args.dataset, args.model_name)
    config_data = runtime_config["config_data"]
    paths_config = runtime_config["paths"]
    model_config = runtime_config["model"]
    if args.semantic_embedding_init is not None:
        model_config["semantic_embedding_init"] = args.semantic_embedding_init
    if args.semantic_embedding_init_path is not None:
        model_config["semantic_embedding_init_path"] = args.semantic_embedding_init_path
    if args.semantic_embedding_projection is not None:
        model_config["semantic_embedding_projection"] = args.semantic_embedding_projection
    if args.semantic_embedding_source_path is not None:
        model_config["semantic_embedding_source_path"] = args.semantic_embedding_source_path
    if args.tie_input_output_embeddings:
        model_config["tie_input_output_embeddings"] = True
    if args.semantic_embedding_freeze_scope is not None:
        model_config["semantic_embedding_freeze_scope"] = args.semantic_embedding_freeze_scope
    runtime_options = runtime_config["runtime"]
    training_config = runtime_config["training"]
    tokenizer_config = runtime_config["tokenizer"]
    evaluation_config = runtime_config["evaluation"]
    validation_objective = (
        args.validation_objective
        or evaluation_config.get("validation_objective", "generation")
    ).lower()
    eval_split = (args.eval_split or evaluation_config.get("eval_split", "train")).lower()
    
    # 使用从配置中读取的参数
    dataset_path = resolve_data_split_path(paths_config, "train")
    eval_dataset_path = resolve_data_split_path(
        paths_config,
        eval_split,
        fallback_to_legacy_eval=False,
    )
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(paths_config['output_dir'], f"{model_config['model_type']}_{timestamp}")
    max_seq_length = model_config['max_seq_length']
    generation_length = len(tokenizer_config['codeword_nums'])
    model_arch = str(model_config.get("model_arch", "causal")).lower()
    is_seq2seq = model_arch == "seq2seq"
    preprocess_num_proc = runtime_options["preprocess_num_proc"]
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 初始化日志
    setup_logging(output_dir, config_data)
    logging.info("Resolved train dataset path: %s", dataset_path)
    logging.info("Validation objective: %s", validation_objective)
    logging.info("Resolved eval dataset path (%s): %s", eval_split, eval_dataset_path)
    
    # 初始化分词器
    tokenizer = create_tokenizer_from_config(
        max_seq_length=max_seq_length,
        generation_length=generation_length,
        tokenizer_config=tokenizer_config,
    )
    input_sid_pooling_config = model_config.get("input_sid_pooling") or {}
    input_sid_pool_item_token_size = None
    if input_sid_pooling_config:
        if isinstance(input_sid_pooling_config, dict):
            input_sid_pool_enabled = bool(input_sid_pooling_config.get("enabled", False))
            input_sid_pool_item_token_size = (
                int(input_sid_pooling_config.get("item_token_size", 4)) if input_sid_pool_enabled else None
            )
        elif isinstance(input_sid_pooling_config, bool):
            input_sid_pool_item_token_size = 4 if input_sid_pooling_config else None
        else:
            raise ValueError(
                f"model.input_sid_pooling must be a bool or dict, got {type(input_sid_pooling_config)}"
            )
    
    # 数据集加载
    if is_seq2seq:
        train_dataset = load_tokenized_seq2seq_dataset(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            max_source_length=max_seq_length,
            max_target_length=generation_length + 1,
            num_proc=preprocess_num_proc,
            input_sid_pool_item_token_size=input_sid_pool_item_token_size,
        )
        eval_dataset = load_tokenized_seq2seq_dataset(
            dataset_path=eval_dataset_path,
            tokenizer=tokenizer,
            max_source_length=max_seq_length,
            max_target_length=generation_length + 1,
            num_proc=preprocess_num_proc,
            input_sid_pool_item_token_size=input_sid_pool_item_token_size,
        )
    else:
        train_dataset = load_tokenized_train_dataset(
            dataset_path=dataset_path,
            tokenizer=tokenizer,
            max_length=max_seq_length + generation_length,
            num_proc=preprocess_num_proc,
        )
        if validation_objective == "generation":
            eval_dataset = load_tokenized_eval_dataset(
                dataset_path=eval_dataset_path,
                tokenizer=tokenizer,
                max_length=max_seq_length,
                num_proc=preprocess_num_proc,
            )
        elif validation_objective == "loss":
            eval_dataset = load_tokenized_loss_eval_dataset(
                dataset_path=eval_dataset_path,
                tokenizer=tokenizer,
                max_length=max_seq_length + generation_length,
                num_proc=preprocess_num_proc,
            )
        else:
            raise ValueError(f"Unsupported validation_objective: {validation_objective}")

    # 模型构建
    logging.info("Creating model from scratch...")
    model = build_model_from_config(
        model_config=model_config,
        tokenizer=tokenizer,
        max_position_embeddings=max_seq_length + generation_length,
        dataset_path=paths_config["dataset_path"],
    )
    logging.info(f"Model created with {model.num_parameters() / 1e6:.2f} M parameters.")
    input_sid_pooler = build_input_sid_pooler(model_config, tokenizer)

    # TrainingArguments 构建
    training_args_dict = dict(training_config)
    early_stopping_patience = int(training_args_dict.pop("early_stopping_patience", 10))
    training_args_dict['output_dir'] = output_dir
    training_args_dict['logging_dir'] = os.path.join(output_dir, 'logs')
    if validation_objective == "loss":
        training_args_dict["metric_for_best_model"] = "eval_loss"
        training_args_dict["greater_is_better"] = False
    # 使用字典解包来创建 TrainingArguments 实例
    training_args = TrainingArguments(**training_args_dict)

    # DataCollator实例化
    train_collator = (
        Seq2SeqDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
        if is_seq2seq
        else TrainDataCollator(tokenizer=tokenizer, max_length=max_seq_length)
    )
    eval_collator = EvalDataCollator(tokenizer=tokenizer, max_length=max_seq_length)

    item_token_codebooks = build_item_token_codebooks_dynamically(tokenizer, generation_length)
    evaluation_mode = evaluation_config.get("evaluation_mode", "sid")
    generation_constraint = args.generation_constraint or evaluation_config.get(
        "generation_constraint",
        evaluation_config.get("constraint_mode", "level_only"),
    )
    sid_to_pid_mapping_path = evaluation_config.get("sid_to_pid_mapping_path")
    if sid_to_pid_mapping_path is None:
        sid_to_pid_mapping_path = resolve_sid_to_pid_mapping_path(paths_config["dataset_path"])
    generation_config_params = {
        "generation_length": generation_length,
        "num_beams": evaluation_config['num_beams'],
        "k_values": evaluation_config['eval_k_values'],
        "item_token_codebooks": item_token_codebooks,
        "evaluation_mode": evaluation_mode,
        "sid_to_pid_mapping_path": sid_to_pid_mapping_path,
        "pid_selection_strategy": evaluation_config.get("pid_selection_strategy", "most_popular_originally"),
        "pid_random_seed": evaluation_config.get("pid_random_seed", 42),
        "generation_constraint": generation_constraint,
    }

    if validation_objective == "generation":
        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=train_collator,
            eval_collator=eval_collator,
            generation_config_params=generation_config_params,
            input_sid_pooler=input_sid_pooler,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
    else:
        trainer = InputPoolingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=train_collator,
            input_sid_pooler=input_sid_pooler,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )

    # 训练
    logging.info("Starting training...")
    trainer.train()

    # 打印历史最优指标
    if trainer.state.best_model_checkpoint:
        best_metric_name = training_args.metric_for_best_model
        best_metric_val = trainer.state.best_metric
        logging.info("=" * 40)
        logging.info(f"🏆 训练结束，历史最优结果如下：")
        logging.info(f"最优模型路径: {trainer.state.best_model_checkpoint}")
        # 如果你配置了 metric_for_best_model，这里会显示具体数值
        logging.info(f"最优指标 ({best_metric_name}): {best_metric_val}")
        logging.info("=" * 40)
    else:
        logging.info("⚠️ 未找到最优模型记录 (请检查 YAML 中是否设置了 load_best_model_at_end=True)")

    # 保存模型
    # 注意：如果 load_best_model_at_end=True，trainer.train() 结束时模型参数已经是“最优的”了
    final_model_path = os.path.join(output_dir, "best_model")
    logging.info(f"Saving model to {final_model_path}")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    logging.info("All operations complete!")

if __name__ == "__main__":
    main()
