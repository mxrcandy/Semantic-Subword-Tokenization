import json
import os
import pickle
from typing import Iterable, List, Sequence

import numpy as np
import torch


def load_embedding_matrix(path: str) -> np.ndarray:
    path = os.path.abspath(path)
    if path.endswith(".npy"):
        embeddings = np.load(path)
    elif path.endswith(".pt") or path.endswith(".pth"):
        tensor = torch.load(path, map_location="cpu")
        if isinstance(tensor, torch.Tensor):
            embeddings = tensor.detach().cpu().numpy()
        else:
            raise TypeError(f"Unsupported torch object type in {path}: {type(tensor)}")
    else:
        raise ValueError(f"Unsupported embedding file format: {path}")

    if embeddings.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape={embeddings.shape}")
    return embeddings.astype(np.float32, copy=False)


def code_to_tokens(code: Sequence[int]) -> List[str]:
    tokens = []
    for level, cluster_id in enumerate(code):
        prefix = chr(ord("a") + level)
        tokens.append(f"<{prefix}_{int(cluster_id)}>")
    return tokens


def dump_item_codes_json(codes: np.ndarray, output_path: str, tokenized: bool = True) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    payload = {}
    for item_id, code in enumerate(codes.tolist()):
        payload[str(item_id)] = code_to_tokens(code) if tokenized else [int(v) for v in code]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def save_quantizer(quantizer, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(quantizer, f)


def load_quantizer(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
