import json
import logging
import os
import re
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch
import transformers
import yaml
from datasets import load_dataset
from sklearn.decomposition import PCA
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    T5Config,
    T5ForConditionalGeneration,
)

from llamarec import (
    LlamaRecConfig,
    LlamaRecForCausalLM,
    LlamaRecSlotConfig,
    LlamaRecSlotForCausalLM,
    T5RecSlotForConditionalGeneration,
)
from util.datacollator import (
    preprocess_eval_dataset,
    preprocess_loss_eval_dataset,
    preprocess_seq2seq_dataset,
    preprocess_train_dataset,
)
from quantization.io import load_quantizer
from util.semantic_tokenizer import create_semantic_id_tokenizer


HF_CAUSAL_MODEL_ALIASES = {
    "hf-llama": "llama",
    "llama": "llama",
    "qwen": "qwen2",
    "qwen2": "qwen2",
    "qwen3": "qwen3",
    "mistral": "mistral",
    "gemma": "gemma",
    "gemma2": "gemma2",
    "phi": "phi",
    "phi3": "phi3",
    "gpt-neox": "gpt_neox",
    "gpt_neox": "gpt_neox",
    "mixtral": "mixtral",
}

SEQ2SEQ_MODEL_TYPES = {"t5-rec", "t5", "t5-rec-slot"}


DEFAULT_DATA_SPLIT_FILENAMES = {
    "train": "train_data.json",
    "val": "val_data.json",
    "test": "test_data.json",
}

SEMANTIC_TOKEN_PATTERN = re.compile(r"^<([a-z])_(\d+)>$")


def resolve_experiment_config_path(dataset: str, model_name: str) -> str:
    filename = f"{dataset}_{model_name}.yaml"
    direct_path = os.path.join("pretrain_config", filename)
    if os.path.exists(direct_path):
        return direct_path

    for root, _, files in os.walk("pretrain_config"):
        if filename in files:
            return os.path.join(root, filename)

    raise FileNotFoundError(f"Could not find config file {filename} under pretrain_config/")


def load_experiment_config(dataset: str, model_name: str) -> Dict[str, Any]:
    config_path = resolve_experiment_config_path(dataset, model_name)
    with open(config_path, "r") as f:
        config_data = yaml.safe_load(f)

    return {
        "config_path": config_path,
        "config_data": config_data,
        "paths": config_data["paths"],
        "model": config_data["model"],
        "runtime": config_data["runtime"],
        "training": config_data["training"],
        "tokenizer": config_data["tokenizer"],
        "evaluation": config_data["evaluation"],
    }


def resolve_data_split_path(
    paths_config: Dict[str, Any],
    split_name: Optional[str],
    *,
    fallback_to_legacy_eval: bool = False,
) -> str:
    dataset_dir = paths_config["dataset_path"]
    normalized_split = (split_name or "").strip().lower()
    split_data_paths = paths_config.get("split_data_paths", {}) or {}

    if normalized_split in {"", "default"}:
        if fallback_to_legacy_eval:
            return (
                paths_config.get("eval_data_path")
                or paths_config.get("train_data_path")
                or os.path.join(dataset_dir, DEFAULT_DATA_SPLIT_FILENAMES["train"])
            )
        return paths_config.get("train_data_path") or os.path.join(
            dataset_dir,
            DEFAULT_DATA_SPLIT_FILENAMES["train"],
        )

    if normalized_split == "eval":
        return (
            paths_config.get("eval_data_path")
            or split_data_paths.get("val")
            or split_data_paths.get("eval")
            or os.path.join(dataset_dir, DEFAULT_DATA_SPLIT_FILENAMES["train"])
        )

    if normalized_split == "train":
        return split_data_paths.get("train") or paths_config.get("train_data_path") or os.path.join(
            dataset_dir,
            DEFAULT_DATA_SPLIT_FILENAMES["train"],
        )

    explicit_key = f"{normalized_split}_data_path"
    resolved_path = (
        split_data_paths.get(normalized_split)
        or paths_config.get(explicit_key)
        or (
            os.path.join(dataset_dir, DEFAULT_DATA_SPLIT_FILENAMES[normalized_split])
            if normalized_split in DEFAULT_DATA_SPLIT_FILENAMES
            else None
        )
    )
    if resolved_path is None:
        raise ValueError(f"Unsupported data split: {split_name}")

    if not os.path.exists(resolved_path):
        raise FileNotFoundError(
            f"Resolved data split path does not exist for split '{split_name}': {resolved_path}"
        )

    return resolved_path


def setup_logging(output_dir: str, config_data: Dict[str, Any], log_filename: str = "training_process.log") -> str:
    log_file_path = os.path.join(output_dir, log_filename)
    file_handler = logging.FileHandler(log_file_path, mode="w", encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)

    transformers.utils.logging.set_verbosity_info()
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    transformers_logger = transformers.utils.logging.get_logger("transformers")
    transformers_logger.addHandler(file_handler)

    logging.info(f"Logging started. Output file: {log_file_path}")
    logging.info("Loaded Configuration:\n%s", json.dumps(config_data, indent=4, ensure_ascii=False))
    return log_file_path


def ensure_special_token_ids(tokenizer: PreTrainedTokenizerBase) -> PreTrainedTokenizerBase:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if tokenizer.bos_token_id is None:
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("[BOS]")
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("[EOS]")

    assert tokenizer.pad_token_id is not None
    assert tokenizer.bos_token_id is not None
    assert tokenizer.eos_token_id is not None
    return tokenizer


def create_tokenizer_from_config(
    max_seq_length: int,
    generation_length: int,
    tokenizer_config: Dict[str, Any],
) -> PreTrainedTokenizerBase:
    logging.info("Creating tokenizer from config...")
    tokenizer = create_semantic_id_tokenizer(
        codeword_nums=tokenizer_config["codeword_nums"],
        max_length=max_seq_length + generation_length,
        extra_semantic_tokens=tokenizer_config.get("extra_semantic_tokens"),
        extra_semantic_tokens_path=tokenizer_config.get("extra_semantic_tokens_path"),
    )
    tokenizer = ensure_special_token_ids(tokenizer)
    logging.info(
        "Final check - pad_token_id: %s, bos_token_id: %s, eos_token_id: %s",
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    )
    return tokenizer


def load_tokenizer_from_checkpoint(checkpoint_path: str) -> PreTrainedTokenizerBase:
    logging.info("Loading tokenizer from checkpoint: %s", checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=False)
    return ensure_special_token_ids(tokenizer)


def _resolve_semantic_embedding_init_mode(model_config: Dict[str, Any]) -> str:
    mode = str(model_config.get("semantic_embedding_init", "none")).strip().lower()
    aliases = {
        "none": "none",
        "input": "input",
        "input_only": "input",
        "output": "output",
        "output_only": "output",
        "input_output": "input_output",
        "both": "input_output",
    }
    if mode not in aliases:
        raise ValueError(
            "Unsupported semantic_embedding_init: "
            f"{model_config.get('semantic_embedding_init')}. "
            "Expected one of ['none', 'input', 'output', 'input_output']."
        )
    return aliases[mode]


def _resolve_semantic_embedding_init_path(
    model_config: Dict[str, Any],
    dataset_path: Optional[str],
) -> Optional[str]:
    explicit_path = model_config.get("semantic_embedding_init_path")
    if explicit_path:
        return explicit_path
    if not dataset_path:
        return None
    candidate = os.path.join(dataset_path, "quantizer.pkl")
    if os.path.exists(candidate):
        return candidate
    return None


def _resolve_semantic_embedding_source_path(
    model_config: Dict[str, Any],
    dataset_path: Optional[str],
) -> Optional[str]:
    explicit_path = model_config.get("semantic_embedding_source_path")
    if explicit_path:
        return explicit_path
    if not dataset_path:
        return None
    for filename in sorted(os.listdir(dataset_path)) if os.path.isdir(dataset_path) else []:
        if filename.endswith(".emb.npy"):
            return os.path.join(dataset_path, filename)
    return None


def _resolve_semantic_embedding_projection(model_config: Dict[str, Any]) -> str:
    projection = str(model_config.get("semantic_embedding_projection", "random")).strip().lower()
    if projection not in {"random", "pca"}:
        raise ValueError(
            "Unsupported semantic_embedding_projection: "
            f"{model_config.get('semantic_embedding_projection')}. "
            "Expected one of ['random', 'pca']."
        )
    return projection


def _resolve_semantic_embedding_freeze_scope(model_config: Dict[str, Any]) -> str:
    scope = str(model_config.get("semantic_embedding_freeze_scope", "none")).strip().lower()
    aliases = {
        "none": "none",
        "false": "none",
        "0": "none",
        "input": "input",
        "input_only": "input",
        "output": "output",
        "output_only": "output",
        "both": "both",
        "true": "both",
        "1": "both",
    }
    if scope not in aliases:
        raise ValueError(
            "Unsupported semantic_embedding_freeze_scope: "
            f"{model_config.get('semantic_embedding_freeze_scope')}. "
            "Expected one of ['none', 'input', 'output', 'both']."
        )
    return aliases[scope]


def _resolve_pattern_embedding_init_mode(model_config: Dict[str, Any]) -> str:
    mode = str(model_config.get("pattern_embedding_init", "none")).strip().lower()
    aliases = {
        "none": "none",
        "false": "none",
        "0": "none",
        "sum": "sum_atoms",
        "sum_atoms": "sum_atoms",
        "mean": "mean_atoms",
        "mean_atoms": "mean_atoms",
    }
    if mode not in aliases:
        raise ValueError(
            "Unsupported pattern_embedding_init: "
            f"{model_config.get('pattern_embedding_init')}. "
            "Expected one of ['none', 'sum_atoms', 'mean_atoms']."
        )
    return aliases[mode]


def _resolve_pattern_embedding_init_scope(model_config: Dict[str, Any]) -> str:
    scope = str(model_config.get("pattern_embedding_init_scope", "input")).strip().lower()
    aliases = {
        "input": "input",
        "input_only": "input",
        "output": "output",
        "output_only": "output",
        "input_output": "input_output",
        "both": "input_output",
    }
    if scope not in aliases:
        raise ValueError(
            "Unsupported pattern_embedding_init_scope: "
            f"{model_config.get('pattern_embedding_init_scope')}. "
            "Expected one of ['input', 'output', 'input_output']."
        )
    return aliases[scope]


def _resolve_pattern_embedding_init_path(
    model_config: Dict[str, Any],
    dataset_path: Optional[str],
) -> Optional[str]:
    explicit_path = model_config.get("pattern_embedding_init_path")
    if explicit_path:
        return explicit_path
    if not dataset_path:
        return None
    candidate = os.path.join(dataset_path, "varlen_meta.json")
    if os.path.exists(candidate):
        return candidate
    return None


def _should_freeze_pattern_embeddings(model_config: Dict[str, Any]) -> bool:
    value = model_config.get("pattern_embedding_freeze", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _extract_quantizer_codebooks(quantizer: Any) -> list[np.ndarray]:
    codebooks = getattr(quantizer, "codebooks_", None)
    if not codebooks:
        raise ValueError(f"Quantizer does not expose codebooks_: {type(quantizer)}")
    normalized = []
    for codebook in codebooks:
        array = np.asarray(codebook, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError(f"Expected 2D codebook array, got shape={array.shape}")
        normalized.append(array)
    return normalized


def _semantic_token_rows_from_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
) -> list[tuple[int, int, int, str]]:
    semantic_rows = []
    vocab = tokenizer.get_vocab()
    for token, token_id in vocab.items():
        match = SEMANTIC_TOKEN_PATTERN.match(token)
        if match is None:
            continue
        prefix, cluster_id = match.groups()
        level_idx = ord(prefix) - ord("a")
        semantic_rows.append((int(token_id), level_idx, int(cluster_id), token))
    semantic_rows.sort(key=lambda row: row[0])
    return semantic_rows


def _project_codebook_to_hidden(
    codebook: np.ndarray,
    hidden_size: int,
    seed: int,
    scale: float,
) -> np.ndarray:
    input_dim = int(codebook.shape[1])
    if input_dim == hidden_size:
        projected = codebook.astype(np.float32, copy=True)
    else:
        generator = np.random.default_rng(seed)
        projection = generator.standard_normal((input_dim, hidden_size), dtype=np.float32)
        projection /= np.sqrt(max(input_dim, 1))
        projected = codebook @ projection

    projected = projected.astype(np.float32, copy=False)
    row_norm = np.linalg.norm(projected, axis=1, keepdims=True)
    row_norm = np.clip(row_norm, a_min=1e-12, a_max=None)
    projected = projected / row_norm
    return (projected * float(scale)).astype(np.float32, copy=False)


def _project_codebooks_with_pca(
    codebooks: list[np.ndarray],
    embedding_source_path: str,
    hidden_size: int,
    seed: int,
    scale: float,
    cache_path: Optional[str] = None,
) -> dict[int, np.ndarray]:
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path)
        projected_by_level = {}
        for level_idx in range(len(codebooks)):
            key = f"level_{level_idx}"
            if key not in cached:
                raise ValueError(f"PCA projection cache missing key {key}: {cache_path}")
            projected = cached[key].astype(np.float32, copy=False)
            if projected.shape != (codebooks[level_idx].shape[0], hidden_size):
                raise ValueError(
                    f"PCA projection cache shape mismatch for {key}: "
                    f"{projected.shape} vs {(codebooks[level_idx].shape[0], hidden_size)}"
                )
            projected_by_level[level_idx] = projected
        logging.info("Loaded cached semantic PCA projection from %s", cache_path)
        return projected_by_level

    embeddings = np.load(embedding_source_path).astype(np.float32, copy=False)
    if embeddings.ndim != 2:
        raise ValueError(f"Expected 2D embedding source, got shape={embeddings.shape}: {embedding_source_path}")
    if hidden_size > min(embeddings.shape):
        raise ValueError(
            f"PCA projection hidden_size={hidden_size} exceeds max components {min(embeddings.shape)} "
            f"for {embedding_source_path}"
        )

    pca = PCA(n_components=hidden_size, svd_solver="randomized", random_state=seed)
    pca.fit(embeddings)

    projected_by_level = {}
    for level_idx, codebook in enumerate(codebooks):
        if int(codebook.shape[1]) != int(embeddings.shape[1]):
            raise ValueError(
                f"Codebook level {level_idx} dim {codebook.shape[1]} does not match "
                f"embedding source dim {embeddings.shape[1]} for PCA projection."
            )
        projected = pca.transform(codebook).astype(np.float32, copy=False)
        row_norm = np.linalg.norm(projected, axis=1, keepdims=True)
        row_norm = np.clip(row_norm, a_min=1e-12, a_max=None)
        projected_by_level[level_idx] = (projected / row_norm * float(scale)).astype(np.float32, copy=False)

    if cache_path:
        os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
        tmp_path = f"{cache_path}.tmp.{os.getpid()}.npz"
        np.savez(tmp_path, **{f"level_{idx}": value for idx, value in projected_by_level.items()})
        os.replace(tmp_path, cache_path)
        logging.info("Saved semantic PCA projection cache to %s", cache_path)
    return projected_by_level


def _resolve_initializer_scale(model: nn.Module, hidden_size: int) -> float:
    initializer_range = getattr(model.config, "initializer_range", None)
    if initializer_range is not None:
        return float(initializer_range)
    return float(hidden_size) ** -0.5


def _copy_rows_to_embedding(embedding: nn.Embedding, row_values: Dict[int, np.ndarray]) -> int:
    copied = 0
    with torch.no_grad():
        for token_id, value in row_values.items():
            if token_id < 0 or token_id >= embedding.num_embeddings:
                continue
            tensor_value = torch.as_tensor(value, dtype=embedding.weight.dtype, device=embedding.weight.device)
            if tensor_value.shape[-1] != embedding.embedding_dim:
                raise ValueError(
                    f"Embedding dim mismatch for token_id={token_id}: "
                    f"{tensor_value.shape[-1]} vs {embedding.embedding_dim}"
                )
            embedding.weight[token_id].copy_(tensor_value)
            copied += 1
    return copied


def _copy_rows_to_linear(linear: nn.Linear, row_values: Dict[int, np.ndarray]) -> int:
    copied = 0
    out_features, in_features = linear.weight.shape
    with torch.no_grad():
        for token_id, value in row_values.items():
            if token_id < 0 or token_id >= out_features:
                continue
            tensor_value = torch.as_tensor(value, dtype=linear.weight.dtype, device=linear.weight.device)
            if tensor_value.shape[-1] != in_features:
                raise ValueError(
                    f"Linear dim mismatch for token_id={token_id}: "
                    f"{tensor_value.shape[-1]} vs {in_features}"
                )
            linear.weight[token_id].copy_(tensor_value)
            copied += 1
    return copied


def _load_pattern_atom_token_map(path: str) -> Dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        data: Any = json.load(f)
    patterns = data.get("patterns") if isinstance(data, dict) else None
    if not isinstance(patterns, dict):
        raise ValueError(f"Pattern embedding init requires a varlen_meta.json with a patterns dict: {path}")

    pattern_atom_tokens: Dict[str, list[str]] = {}
    for pattern_token, pattern_info in patterns.items():
        if isinstance(pattern_info, dict):
            atom_tokens = pattern_info.get("tokens")
        else:
            atom_tokens = None
        if not isinstance(atom_tokens, list) or not atom_tokens:
            raise ValueError(f"Pattern {pattern_token} has no non-empty tokens list in {path}")
        pattern_atom_tokens[str(pattern_token)] = [str(token) for token in atom_tokens]
    return pattern_atom_tokens


def _pattern_token_rows_from_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    pattern_atom_tokens: Dict[str, list[str]],
) -> Dict[int, list[int]]:
    vocab = tokenizer.get_vocab()
    pattern_rows: Dict[int, list[int]] = {}
    for pattern_token, atom_tokens in pattern_atom_tokens.items():
        if pattern_token not in vocab:
            raise ValueError(f"Pattern token {pattern_token} is not in tokenizer vocab.")
        atom_ids = []
        for atom_token in atom_tokens:
            if atom_token not in vocab:
                raise ValueError(f"Atom token {atom_token} for pattern {pattern_token} is not in tokenizer vocab.")
            atom_ids.append(int(vocab[atom_token]))
        pattern_rows[int(vocab[pattern_token])] = atom_ids
    return pattern_rows


def _copy_pattern_rows_to_embedding(
    embedding: nn.Embedding,
    pattern_rows: Dict[int, list[int]],
    mode: str,
) -> int:
    copied = 0
    with torch.no_grad():
        for pattern_id, atom_ids in pattern_rows.items():
            if pattern_id < 0 or pattern_id >= embedding.num_embeddings:
                continue
            atom_ids_tensor = torch.tensor(atom_ids, dtype=torch.long, device=embedding.weight.device)
            atom_values = embedding.weight.index_select(0, atom_ids_tensor)
            pattern_value = atom_values.sum(dim=0)
            if mode == "mean_atoms":
                pattern_value = pattern_value / max(len(atom_ids), 1)
            embedding.weight[pattern_id].copy_(pattern_value)
            copied += 1
    return copied


def _copy_pattern_rows_to_linear(
    linear: nn.Linear,
    pattern_rows: Dict[int, list[int]],
    mode: str,
) -> int:
    copied = 0
    out_features, _ = linear.weight.shape
    with torch.no_grad():
        for pattern_id, atom_ids in pattern_rows.items():
            if pattern_id < 0 or pattern_id >= out_features:
                continue
            atom_ids_tensor = torch.tensor(atom_ids, dtype=torch.long, device=linear.weight.device)
            atom_values = linear.weight.index_select(0, atom_ids_tensor)
            pattern_value = atom_values.sum(dim=0)
            if mode == "mean_atoms":
                pattern_value = pattern_value / max(len(atom_ids), 1)
            linear.weight[pattern_id].copy_(pattern_value)
            copied += 1
    return copied


def _register_row_freeze_hook(weight: torch.nn.Parameter, token_ids: list[int], label: str) -> bool:
    if not weight.requires_grad:
        logging.info("Skipping semantic row freeze for %s because the parameter does not require grad.", label)
        return False
    if not token_ids:
        logging.warning("Skipping semantic row freeze for %s because no semantic token ids were found.", label)
        return False

    token_ids_tensor = torch.tensor(sorted(set(token_ids)), dtype=torch.long)

    def freeze_semantic_rows(grad: torch.Tensor) -> torch.Tensor:
        if grad is None:
            return grad
        ids = token_ids_tensor.to(device=grad.device)
        if grad.is_sparse:
            dense_grad = grad.to_dense()
            dense_grad.index_fill_(0, ids, 0)
            return dense_grad.to_sparse()
        masked_grad = grad.clone()
        masked_grad.index_fill_(0, ids, 0)
        return masked_grad

    weight.register_hook(freeze_semantic_rows)
    logging.info("Registered semantic row freeze hook for %s on %s rows.", label, len(token_ids_tensor))
    return True


def apply_semantic_embedding_freeze(
    model: nn.Module,
    model_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: Optional[str] = None,
) -> None:
    scope = _resolve_semantic_embedding_freeze_scope(model_config)
    if scope == "none":
        logging.info("Semantic embedding freeze disabled.")
        return

    semantic_token_ids = [token_id for token_id, _, _, _ in _semantic_token_rows_from_tokenizer(tokenizer)]
    pattern_token_ids: list[int] = []
    if _should_freeze_pattern_embeddings(model_config):
        pattern_path = _resolve_pattern_embedding_init_path(model_config, dataset_path)
        if not pattern_path:
            raise FileNotFoundError(
                "pattern_embedding_freeze=true, but no varlen metadata path could be resolved. "
                "Set model.pattern_embedding_init_path or ensure dataset_path/varlen_meta.json exists."
            )
        pattern_atom_tokens = _load_pattern_atom_token_map(pattern_path)
        pattern_rows = _pattern_token_rows_from_tokenizer(tokenizer, pattern_atom_tokens)
        pattern_token_ids = sorted(pattern_rows)
        logging.info("Including %s pattern rows in semantic embedding freeze.", len(pattern_token_ids))

    freeze_token_ids = sorted(set(semantic_token_ids + pattern_token_ids))
    if not freeze_token_ids:
        logging.warning("No semantic or pattern token ids found in tokenizer; skipping embedding freeze.")
        return

    seen_params: set[int] = set()

    def maybe_freeze_weight(weight: torch.nn.Parameter, label: str) -> None:
        param_id = id(weight)
        if param_id in seen_params:
            logging.info("Skipping duplicate semantic row freeze hook for tied parameter: %s", label)
            return
        seen_params.add(param_id)
        _register_row_freeze_hook(weight, freeze_token_ids, label)

    if scope in {"input", "both"}:
        input_embeddings = model.get_input_embeddings()
        maybe_freeze_weight(input_embeddings.weight, "input embeddings")

    if scope in {"output", "both"}:
        output_embeddings = model.get_output_embeddings()
        if isinstance(output_embeddings, nn.ModuleList):
            for idx, head in enumerate(output_embeddings):
                maybe_freeze_weight(head.weight, f"slot output head {idx}")
        elif isinstance(output_embeddings, nn.Linear):
            maybe_freeze_weight(output_embeddings.weight, "output lm_head")
        else:
            raise TypeError(
                "Unsupported output embedding module for semantic freeze: "
                f"{type(output_embeddings)}"
            )


def _should_tie_input_output_embeddings(model_config: Dict[str, Any]) -> bool:
    value = model_config.get("tie_input_output_embeddings", False)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def apply_input_output_embedding_tie(model: nn.Module, model_config: Dict[str, Any]) -> None:
    if not _should_tie_input_output_embeddings(model_config):
        logging.info("Input/output embedding tying disabled.")
        return

    output_embeddings = model.get_output_embeddings()
    if isinstance(output_embeddings, nn.ModuleList):
        logging.warning("Ignoring tie_input_output_embeddings=true for slot-output model.")
        return
    if not isinstance(output_embeddings, nn.Linear):
        raise TypeError(f"Unsupported output embedding module for tying: {type(output_embeddings)}")

    input_embeddings = model.get_input_embeddings()
    if output_embeddings.weight.shape != input_embeddings.weight.shape:
        raise ValueError(
            "Cannot tie input/output embeddings with different shapes: "
            f"input={tuple(input_embeddings.weight.shape)} output={tuple(output_embeddings.weight.shape)}"
        )

    if hasattr(model.config, "tie_word_embeddings"):
        model.config.tie_word_embeddings = True
    if hasattr(model, "tie_weights"):
        model.tie_weights()
    else:
        output_embeddings.weight = input_embeddings.weight

    tied = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    if not tied:
        model.get_output_embeddings().weight = model.get_input_embeddings().weight
        tied = model.get_output_embeddings().weight.data_ptr() == model.get_input_embeddings().weight.data_ptr()
    if not tied:
        raise RuntimeError("Failed to tie input/output embeddings.")
    logging.info("Tied input embeddings and output lm_head weights.")


def apply_semantic_embedding_init(
    model: nn.Module,
    model_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: Optional[str] = None,
) -> None:
    mode = _resolve_semantic_embedding_init_mode(model_config)
    if mode == "none":
        logging.info("Semantic embedding init disabled.")
        return

    quantizer_path = _resolve_semantic_embedding_init_path(model_config, dataset_path)
    if not quantizer_path:
        raise FileNotFoundError(
            "Semantic embedding init is enabled, but no quantizer path could be resolved. "
            "Set model.semantic_embedding_init_path or ensure dataset_path/quantizer.pkl exists."
        )

    semantic_rows = _semantic_token_rows_from_tokenizer(tokenizer)
    if not semantic_rows:
        logging.warning("No semantic ID tokens found in tokenizer; skipping semantic embedding init.")
        return

    quantizer = load_quantizer(quantizer_path)
    codebooks = _extract_quantizer_codebooks(quantizer)
    hidden_size = int(model.get_input_embeddings().embedding_dim)
    init_seed = int(model_config.get("semantic_embedding_init_seed", 42))
    init_scale = float(model_config.get("semantic_embedding_init_scale", _resolve_initializer_scale(model, hidden_size)))
    projection = _resolve_semantic_embedding_projection(model_config)

    if projection == "pca":
        embedding_source_path = _resolve_semantic_embedding_source_path(model_config, dataset_path)
        if not embedding_source_path:
            raise FileNotFoundError(
                "semantic_embedding_projection=pca requires an embedding source. "
                "Set model.semantic_embedding_source_path or ensure dataset_path/*.emb.npy exists."
            )
        cache_path = model_config.get("semantic_embedding_projection_cache_path")
        if not cache_path and dataset_path:
            cache_path = os.path.join(
                dataset_path,
                f"semantic_embedding_projection_pca_h{hidden_size}_s{init_seed}_scale{init_scale:g}.npz",
            )
        logging.info(
            "Applying semantic embedding init with PCA projection: quantizer=%s source=%s hidden_size=%s",
            quantizer_path,
            embedding_source_path,
            hidden_size,
        )
        projected_by_level = _project_codebooks_with_pca(
            codebooks=codebooks,
            embedding_source_path=embedding_source_path,
            hidden_size=hidden_size,
            seed=init_seed,
            scale=init_scale,
            cache_path=cache_path,
        )
    else:
        logging.info(
            "Applying semantic embedding init with random projection: quantizer=%s hidden_size=%s",
            quantizer_path,
            hidden_size,
        )
        projected_by_level = {
            level_idx: _project_codebook_to_hidden(
                codebook=codebook,
                hidden_size=hidden_size,
                seed=init_seed + level_idx,
                scale=init_scale,
            )
            for level_idx, codebook in enumerate(codebooks)
        }

    row_values: Dict[int, np.ndarray] = {}
    for token_id, level_idx, cluster_id, token in semantic_rows:
        if level_idx not in projected_by_level:
            raise ValueError(f"Semantic token {token} refers to missing level {level_idx} in quantizer.")
        projected_codebook = projected_by_level[level_idx]
        if cluster_id >= projected_codebook.shape[0]:
            raise ValueError(
                f"Semantic token {token} refers to cluster_id={cluster_id}, "
                f"but codebook level {level_idx} size is {projected_codebook.shape[0]}."
            )
        row_values[token_id] = projected_codebook[cluster_id]

    if mode in {"input", "input_output"}:
        copied = _copy_rows_to_embedding(model.get_input_embeddings(), row_values)
        logging.info("Applied semantic input embedding init for %s tokenizer rows.", copied)

    if mode in {"output", "input_output"}:
        output_embeddings = model.get_output_embeddings()
        if isinstance(output_embeddings, nn.ModuleList):
            copied_counts = []
            for head in output_embeddings:
                copied_counts.append(_copy_rows_to_linear(head, row_values))
            logging.info(
                "Applied semantic output embedding init to %s slot heads; copied rows per head=%s",
                len(output_embeddings),
                copied_counts,
            )
        elif isinstance(output_embeddings, nn.Linear):
            copied = _copy_rows_to_linear(output_embeddings, row_values)
            logging.info("Applied semantic output embedding init for %s tokenizer rows.", copied)
        else:
            raise TypeError(
                "Unsupported output embedding module for semantic init: "
                f"{type(output_embeddings)}"
            )


def apply_pattern_embedding_init(
    model: nn.Module,
    model_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: Optional[str] = None,
) -> None:
    mode = _resolve_pattern_embedding_init_mode(model_config)
    if mode == "none":
        logging.info("Pattern embedding init disabled.")
        return

    pattern_path = _resolve_pattern_embedding_init_path(model_config, dataset_path)
    if not pattern_path:
        raise FileNotFoundError(
            "Pattern embedding init is enabled, but no varlen metadata path could be resolved. "
            "Set model.pattern_embedding_init_path or ensure dataset_path/varlen_meta.json exists."
        )

    pattern_atom_tokens = _load_pattern_atom_token_map(pattern_path)
    pattern_rows = _pattern_token_rows_from_tokenizer(tokenizer, pattern_atom_tokens)
    if not pattern_rows:
        logging.warning("No pattern tokens found in %s; skipping pattern embedding init.", pattern_path)
        return

    scope = _resolve_pattern_embedding_init_scope(model_config)
    if scope in {"input", "input_output"}:
        copied = _copy_pattern_rows_to_embedding(model.get_input_embeddings(), pattern_rows, mode)
        logging.info("Applied %s pattern input embedding init for %s tokenizer rows.", mode, copied)

    if scope in {"output", "input_output"}:
        output_embeddings = model.get_output_embeddings()
        if isinstance(output_embeddings, nn.ModuleList):
            copied_counts = []
            for head in output_embeddings:
                copied_counts.append(_copy_pattern_rows_to_linear(head, pattern_rows, mode))
            logging.info(
                "Applied %s pattern output embedding init to %s slot heads; copied rows per head=%s",
                mode,
                len(output_embeddings),
                copied_counts,
            )
        elif isinstance(output_embeddings, nn.Linear):
            copied = _copy_pattern_rows_to_linear(output_embeddings, pattern_rows, mode)
            logging.info("Applied %s pattern output embedding init for %s tokenizer rows.", mode, copied)
        else:
            raise TypeError(
                "Unsupported output embedding module for pattern init: "
                f"{type(output_embeddings)}"
            )


def finalize_model_parameter_layout(
    model: nn.Module,
    model_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    dataset_path: Optional[str] = None,
) -> nn.Module:
    apply_input_output_embedding_tie(model=model, model_config=model_config)
    apply_semantic_embedding_init(
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
    )
    apply_pattern_embedding_init(
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
    )
    apply_semantic_embedding_freeze(
        model=model,
        model_config=model_config,
        tokenizer=tokenizer,
        dataset_path=dataset_path,
    )
    return model


def build_model_from_config(
    model_config: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    max_position_embeddings: int,
    dataset_path: Optional[str] = None,
) -> LlamaRecForCausalLM:
    model_type = model_config["model_type"]
    if model_type in SEQ2SEQ_MODEL_TYPES:
        logging.info("Creating T5 seq2seq model from scratch: model_type=%s", model_type)
        t5_config_kwargs = dict(
            vocab_size=len(tokenizer),
            d_model=model_config["hidden_size"],
            d_ff=model_config["intermediate_size"],
            num_layers=model_config["num_hidden_layers"],
            num_decoder_layers=model_config.get("num_decoder_layers", model_config["num_hidden_layers"]),
            num_heads=model_config["num_attention_heads"],
            dropout_rate=model_config.get("dropout_rate", 0.0),
            layer_norm_epsilon=model_config.get("rms_norm_eps", 1.0e-6),
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            decoder_start_token_id=tokenizer.pad_token_id,
            feed_forward_proj=model_config.get("feed_forward_proj", "relu"),
            tie_word_embeddings=model_config.get("tie_word_embeddings", True),
        )
        if model_type == "t5-rec-slot":
            t5_config_kwargs.update(
                item_token_size=model_config.get("item_token_size", 4),
                use_decoder_slot_input_embeddings=model_config.get(
                    "use_decoder_slot_input_embeddings",
                    True,
                ),
                use_decoder_slot_prelogits_embeddings=model_config.get(
                    "use_decoder_slot_prelogits_embeddings",
                    True,
                ),
                is_t5_rec_slot=True,
            )
        config = T5Config(**t5_config_kwargs)
        config.model_type = "t5"
        if model_type == "t5-rec-slot":
            model = T5RecSlotForConditionalGeneration(config)
        else:
            model = T5ForConditionalGeneration(config)
        model.config.use_cache = False
        finalize_model_parameter_layout(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
        )
        return model

    if model_type in HF_CAUSAL_MODEL_ALIASES:
        hf_model_type = HF_CAUSAL_MODEL_ALIASES[model_type]
        config_kwargs = dict(
            vocab_size=len(tokenizer),
            hidden_size=model_config["hidden_size"],
            intermediate_size=model_config["intermediate_size"],
            num_hidden_layers=model_config["num_hidden_layers"],
            num_attention_heads=model_config["num_attention_heads"],
            max_position_embeddings=max_position_embeddings,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
        optional_config_keys = [
            "num_key_value_heads",
            "hidden_act",
            "rms_norm_eps",
            "rope_theta",
            "attention_bias",
            "attention_dropout",
            "tie_word_embeddings",
            "head_dim",
            "initializer_range",
            "sliding_window",
            "num_local_experts",
            "num_experts_per_tok",
        ]
        for key in optional_config_keys:
            if key in model_config:
                config_kwargs[key] = model_config[key]

        logging.info("Creating HF causal LM from scratch: model_type=%s", hf_model_type)
        config = AutoConfig.for_model(hf_model_type, **config_kwargs)
        model = AutoModelForCausalLM.from_config(config)
        model.config.use_cache = False
        finalize_model_parameter_layout(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
        )
        return model

    common_kwargs = dict(
        hidden_size=model_config["hidden_size"],
        intermediate_size=model_config["intermediate_size"],
        num_hidden_layers=model_config["num_hidden_layers"],
        num_attention_heads=model_config["num_attention_heads"],
        max_position_embeddings=max_position_embeddings,
        rms_norm_eps=model_config["rms_norm_eps"],
        model_type=model_type,
        vocab_size=len(tokenizer),
        use_cache=False,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        item_token_size=model_config.get("item_token_size", 4),
    )

    if model_type == "llama-rec-slot":
        config = LlamaRecSlotConfig(
            use_slot_input_embeddings=model_config.get("use_slot_input_embeddings", True),
            use_slot_prelogits_embeddings=model_config.get("use_slot_prelogits_embeddings", True),
            use_slot_output_heads=model_config.get("use_slot_output_heads", True),
            **common_kwargs,
        )
        model = LlamaRecSlotForCausalLM(config)
        finalize_model_parameter_layout(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
        )
        return model


    if model_type == "llama-rec":
        config = LlamaRecConfig(**common_kwargs)
        model = LlamaRecForCausalLM(config)
        finalize_model_parameter_layout(
            model=model,
            model_config=model_config,
            tokenizer=tokenizer,
            dataset_path=dataset_path,
        )
        return model

    supported = sorted(["llama-rec", "llama-rec-slot", *HF_CAUSAL_MODEL_ALIASES, *SEQ2SEQ_MODEL_TYPES])
    raise ValueError(f"Unsupported model_type: {model_type}. Expected one of: {supported}")


def load_model_from_checkpoint(checkpoint_path: str) -> LlamaRecForCausalLM:
    logging.info("Loading model from checkpoint: %s", checkpoint_path)
    config = AutoConfig.from_pretrained(checkpoint_path, trust_remote_code=False)
    architectures = set(getattr(config, "architectures", []) or [])
    if config.model_type == "llama-rec-slot":
        model = LlamaRecSlotForCausalLM.from_pretrained(checkpoint_path)
    elif config.model_type == "llama-rec":
        model = LlamaRecForCausalLM.from_pretrained(checkpoint_path)
    elif getattr(config, "is_encoder_decoder", False) and (
        bool(getattr(config, "is_t5_rec_slot", False))
        or "T5RecSlotForConditionalGeneration" in architectures
    ):
        model = T5RecSlotForConditionalGeneration.from_pretrained(checkpoint_path)
    elif getattr(config, "is_encoder_decoder", False):
        model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_path, trust_remote_code=False)
    else:
        model = AutoModelForCausalLM.from_pretrained(checkpoint_path, trust_remote_code=False)
    logging.info("Model loaded successfully from checkpoint")
    return model


def load_tokenized_train_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    num_proc: Optional[int] = None,
):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    return dataset.map(
        preprocess_train_dataset,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=dataset.column_names,
        desc="Tokenizing train dataset",
    )


def load_tokenized_eval_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    num_proc: Optional[int] = None,
):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    keep_columns = {"sid_ground_truth", "pid_ground_truth"}
    return dataset.map(
        preprocess_eval_dataset,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=[col for col in dataset.column_names if col not in keep_columns],
        desc="Tokenizing eval dataset",
    )


def load_tokenized_loss_eval_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    num_proc: Optional[int] = None,
):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    return dataset.map(
        preprocess_loss_eval_dataset,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
        remove_columns=dataset.column_names,
        desc="Tokenizing loss eval dataset",
    )


def load_tokenized_seq2seq_dataset(
    dataset_path: str,
    tokenizer: PreTrainedTokenizerBase,
    max_source_length: int,
    max_target_length: int,
    num_proc: Optional[int] = None,
    input_sid_pool_item_token_size: Optional[int] = None,
):
    dataset = load_dataset("json", data_files=dataset_path, split="train")
    keep_columns = {"sid_ground_truth", "pid_ground_truth"}
    return dataset.map(
        preprocess_seq2seq_dataset,
        batched=True,
        num_proc=num_proc,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_source_length": max_source_length,
            "max_target_length": max_target_length,
            "input_sid_pool_item_token_size": input_sid_pool_item_token_size,
        },
        remove_columns=[col for col in dataset.column_names if col not in keep_columns],
        desc="Tokenizing seq2seq dataset",
    )
