import torch
from typing import Dict, List, Optional, Union
from transformers import PreTrainedTokenizerBase
import re


ATOM_TOKEN_RE = re.compile(r"^<([a-z])_\d+>$")


def preprocess_train_dataset(
    examples: Dict[str, List[str]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, List[List[int]]]:
    texts = examples["text"]
    if "sid_ground_truth" in examples:
        texts = [
            f"{history_text}{ground_truth[0] if isinstance(ground_truth, list) else ground_truth}"
            for history_text, ground_truth in zip(examples["text"], examples["sid_ground_truth"])
        ]
    tokenized = tokenizer(
        texts,
        is_split_into_words=False,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    if "sid_ground_truth" in examples:
        tokenized["sid_ground_truth"] = examples["sid_ground_truth"]
    if "pid_ground_truth" in examples:
        tokenized["pid_ground_truth"] = examples["pid_ground_truth"]
    return tokenized


def preprocess_eval_dataset(
    examples: Dict[str, List[Union[str, List[str]]]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, List[Union[List[int], List[str]]]]:
    tokenized = tokenizer(
        examples["text"],
        is_split_into_words=False,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )
    tokenized["sid_ground_truth"] = examples["sid_ground_truth"]
    if "pid_ground_truth" in examples:
        tokenized["pid_ground_truth"] = examples["pid_ground_truth"]
    return tokenized


def preprocess_loss_eval_dataset(
    examples: Dict[str, List[Union[str, List[str]]]],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> Dict[str, List[Union[List[int], List[str]]]]:
    input_ids = []
    attention_masks = []
    labels = []

    for history_text, ground_truth in zip(examples["text"], examples["sid_ground_truth"]):
        target_text = ground_truth[0] if isinstance(ground_truth, list) else ground_truth
        full_text = f"{history_text}{target_text}"
        target_token_ids = tokenizer(
            target_text,
            is_split_into_words=False,
            padding=False,
            return_attention_mask=False,
        )["input_ids"]
        tokenized = tokenizer(
            full_text,
            is_split_into_words=False,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=True,
        )
        sample_labels = [-100] * len(tokenized["input_ids"])
        target_len = min(len(target_token_ids), len(sample_labels))
        if target_len:
            sample_labels[-target_len:] = tokenized["input_ids"][-target_len:]

        input_ids.append(tokenized["input_ids"])
        attention_masks.append(tokenized["attention_mask"])
        labels.append(sample_labels)

    result = {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
    }
    if "sid_ground_truth" in examples:
        result["sid_ground_truth"] = examples["sid_ground_truth"]
    if "pid_ground_truth" in examples:
        result["pid_ground_truth"] = examples["pid_ground_truth"]
    return result


def preprocess_seq2seq_dataset(
    examples: Dict[str, List[Union[str, List[str]]]],
    tokenizer: PreTrainedTokenizerBase,
    max_source_length: int,
    max_target_length: int,
    input_sid_pool_item_token_size: Optional[int] = None,
) -> Dict[str, List[Union[List[int], List[str]]]]:
    tokenized = tokenizer(
        examples["text"],
        is_split_into_words=False,
        truncation=True,
        max_length=max_source_length,
        padding=False,
        return_attention_mask=True,
    )
    targets = [
        ground_truth[0] if isinstance(ground_truth, list) else ground_truth
        for ground_truth in examples["sid_ground_truth"]
    ]
    labels = tokenizer(
        targets,
        is_split_into_words=False,
        truncation=True,
        max_length=max_target_length,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]
    tokenized["labels"] = labels
    tokenized["sid_ground_truth"] = examples["sid_ground_truth"]
    if "pid_ground_truth" in examples:
        tokenized["pid_ground_truth"] = examples["pid_ground_truth"]
    if input_sid_pool_item_token_size:
        id_to_slot = build_input_sid_pool_id_to_slot(tokenizer, input_sid_pool_item_token_size)
        tokenized["input_sid_pool_segments"] = [
            build_input_sid_pool_segments_from_ids(
                input_ids,
                item_token_size=input_sid_pool_item_token_size,
                id_to_slot=id_to_slot,
            )
            for input_ids in tokenized["input_ids"]
        ]
    return tokenized


def build_input_sid_pool_id_to_slot(
    tokenizer: PreTrainedTokenizerBase,
    item_token_size: int,
) -> Dict[int, int]:
    expected_slots = [chr(ord("a") + idx) for idx in range(item_token_size)]
    slot_to_index = {slot: idx for idx, slot in enumerate(expected_slots)}
    id_to_slot = {}
    for token, token_id in tokenizer.get_vocab().items():
        match = ATOM_TOKEN_RE.match(token)
        if match is None:
            continue
        slot = match.group(1)
        if slot in slot_to_index:
            id_to_slot[int(token_id)] = slot_to_index[slot]
    return id_to_slot


def build_input_sid_pool_segments_from_ids(
    input_ids: List[int],
    *,
    item_token_size: int,
    id_to_slot: Dict[int, int],
) -> List[List[int]]:
    slot_ids = [id_to_slot.get(int(token_id), -1) for token_id in input_ids]
    segments: List[List[int]] = []
    start: Optional[int] = None
    slots_seen = 0
    pos = 0
    while pos < len(slot_ids):
        slot = slot_ids[pos]
        if slot < 0:
            if start is not None:
                segments.extend([idx, idx + 1] for idx in range(start, pos))
                start = None
                slots_seen = 0
            segments.append([pos, pos + 1])
            pos += 1
            continue

        if start is None:
            if slot == 0:
                start = pos
                slots_seen = 1
            else:
                segments.append([pos, pos + 1])
            pos += 1
            continue

        expected_next = slots_seen if slots_seen < item_token_size else None
        if slot == expected_next:
            slots_seen += 1
            pos += 1
            if slots_seen == item_token_size:
                segments.append([start, pos])
                start = None
                slots_seen = 0
            continue

        segments.extend([idx, idx + 1] for idx in range(start, pos))
        start = None
        slots_seen = 0

    if start is not None:
        segments.extend([idx, idx + 1] for idx in range(start, len(slot_ids)))
    return segments


# 训练数据整理器
class TrainDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        """
        训练数据的整理器。

        Args:
            tokenizer (PreTrainedTokenizerFast): 使用的 tokenizer。
            max_length (int): 最大序列长度。
            item_token_length (int): 每个物品由多少个 token 组成 (例如: <a_id> <b_id> <c_id> 是 3 个 token)。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        features = [
            {
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
            }
            for example in examples
        ]
        batch_dict = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )

        if "labels" in examples[0]:
            max_length = batch_dict["input_ids"].shape[1]
            padded_labels = []
            for example in examples:
                label_ids = list(example["labels"])
                pad_len = max_length - len(label_ids)
                if self.tokenizer.padding_side == "left":
                    label_ids = [-100] * pad_len + label_ids
                else:
                    label_ids = label_ids + [-100] * pad_len
                padded_labels.append(label_ids)
            batch_dict["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        else:
            batch_dict['labels'] = batch_dict['input_ids'].clone()

            # label=-100会让loss忽略
            if self.tokenizer.pad_token_id is not None:
                batch_dict["labels"][batch_dict["labels"] == self.tokenizer.pad_token_id] = -100
        
        return batch_dict


class Seq2SeqDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        features = [
            {
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
            }
            for example in examples
        ]
        batch_dict = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )

        max_label_length = max(len(example["labels"]) for example in examples)
        padded_labels = []
        for example in examples:
            label_ids = list(example["labels"])
            pad_len = max_label_length - len(label_ids)
            label_ids = label_ids + [self.tokenizer.pad_token_id] * pad_len
            padded_labels.append(label_ids)
        labels = torch.tensor(padded_labels, dtype=torch.long)
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch_dict["labels"] = labels
        if "input_sid_pool_segments" in examples[0]:
            max_segments = max(len(example["input_sid_pool_segments"]) for example in examples)
            padded_segments = []
            for example in examples:
                segments = [list(segment) for segment in example["input_sid_pool_segments"]]
                segments.extend([[-1, -1]] * (max_segments - len(segments)))
                padded_segments.append(segments)
            batch_dict["input_sid_pool_segments"] = torch.tensor(padded_segments, dtype=torch.long)
        return batch_dict


# 评估数据整理器
class EvalDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int):
        """
        评估数据的整理器。

        Args:
            tokenizer (PreTrainedTokenizerFast): 使用的 tokenizer。
            max_length (int): 最大序列长度。
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __call__(self, examples: List[Dict[str, Union[str, List[int]]]]) -> Dict[str, torch.Tensor]:
        sid_ground_truths = [e["sid_ground_truth"] for e in examples]
        pid_ground_truths = [e["pid_ground_truth"] for e in examples] if "pid_ground_truth" in examples[0] else None
        features = [
            {
                "input_ids": example["input_ids"],
                "attention_mask": example["attention_mask"],
            }
            for example in examples
        ]
        batch_dict = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt"
        )
        
        batch_dict["sid_ground_truth"] = sid_ground_truths
        if pid_ground_truths is not None:
            batch_dict["pid_ground_truth"] = pid_ground_truths
        return batch_dict
