import json
import logging
import os
import tempfile
from typing import Any, List

from transformers import AddedToken, PreTrainedTokenizerBase

try:
    from transformers import Qwen2Tokenizer
except ImportError:  # pragma: no cover - compatibility fallback
    Qwen2Tokenizer = None


def build_semantic_tokens(codeword_nums: List[int]):
    tokens = []
    for codebook_idx, codeword_num in enumerate(codeword_nums):
        prefix = chr(ord("a") + codebook_idx)
        for token_idx in range(0, codeword_num):
            tokens.append(
                AddedToken(
                    f"<{prefix}_{token_idx}>",
                    special=True,
                    normalized=False,
                    lstrip=False,
                    rstrip=False,
                )
            )
    return tokens


def build_extra_semantic_tokens(extra_semantic_tokens: List[str] | None):
    tokens = []
    for token in extra_semantic_tokens or []:
        tokens.append(
            AddedToken(
                token,
                special=True,
                normalized=False,
                lstrip=False,
                rstrip=False,
            )
        )
    return tokens


def load_extra_semantic_tokens(path: str | None) -> List[str]:
    if not path:
        return []
    with open(path, "r", encoding="utf-8") as f:
        data: Any = json.load(f)
    if isinstance(data, list):
        return [str(token) for token in data]
    if isinstance(data, dict):
        if isinstance(data.get("extra_semantic_tokens"), list):
            return [str(token) for token in data["extra_semantic_tokens"]]
        patterns = data.get("patterns")
        if isinstance(patterns, dict):
            return [str(token) for token in patterns]
    raise ValueError(f"Cannot read extra semantic tokens from {path}")


def create_semantic_id_tokenizer(
    codeword_nums: List[int],
    max_length: int,
    extra_semantic_tokens: List[str] | None = None,
    extra_semantic_tokens_path: str | None = None,
) -> PreTrainedTokenizerBase:
    if Qwen2Tokenizer is None:
        raise ImportError("Qwen2Tokenizer is required for the semantic ID tokenizer.")

    logging.info(
        "Building semantic ID tokenizer with Qwen2 core: codeword_nums=%s",
        codeword_nums,
    )

    dummy_vocab = {"[UNK]": 0}
    with tempfile.TemporaryDirectory() as temp_dir:
        vocab_file = os.path.join(temp_dir, "vocab.json")
        merges_file = os.path.join(temp_dir, "merges.txt")

        with open(vocab_file, "w", encoding="utf-8") as f:
            json.dump(dummy_vocab, f)
        with open(merges_file, "w", encoding="utf-8") as f:
            f.write("#version: 0.2\n")

        tokenizer = Qwen2Tokenizer(
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token="[UNK]",
            pad_token="[PAD]",
            bos_token="[BOS]",
            eos_token="[EOS]",
            padding_side="left",
            truncation_side="left",
        )

    special_tokens = [
        AddedToken("[PAD]", special=True, normalized=False),
        AddedToken("[UNK]", special=True, normalized=False),
        AddedToken("[BOS]", special=True, normalized=False),
        AddedToken("[EOS]", special=True, normalized=False),
    ]
    special_tokens.extend(build_semantic_tokens(codeword_nums=codeword_nums))
    loaded_extra_tokens = list(extra_semantic_tokens or [])
    loaded_extra_tokens.extend(load_extra_semantic_tokens(extra_semantic_tokens_path))
    loaded_extra_tokens = sorted(set(loaded_extra_tokens))
    special_tokens.extend(build_extra_semantic_tokens(loaded_extra_tokens))

    tokenizer.add_special_tokens(
        {"additional_special_tokens": special_tokens},
        replace_additional_special_tokens=False,
    )
    tokenizer.pad_token = "[PAD]"
    tokenizer.unk_token = "[UNK]"
    tokenizer.bos_token = "[BOS]"
    tokenizer.eos_token = "[EOS]"
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    tokenizer.model_max_length = max_length

    logging.info(
        "Tokenizer creation complete. vocab=%s extra_semantic_tokens=%s pad=%s bos=%s eos=%s",
        len(tokenizer),
        len(loaded_extra_tokens),
        tokenizer.pad_token_id,
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
    )
    return tokenizer
