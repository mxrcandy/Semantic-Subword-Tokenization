# Semantic Subword Tokenization

This repository contains a minimal implementation for **Semantic Subword Tokenization (SST)** for generative recommendation. SST augments fixed-length Semantic IDs with:

- **Item-level Subword Tokenization (IST)**: learns merge rules over adjacent SID tokens and rewrites item histories with pattern tokens.
- **Behavior-induced Co-occurrence Augmentation (BCA)**: mines semantic-prefix transitions from user behavior and injects replay samples for training.

The repository intentionally excludes datasets, checkpoints, logs, and generated experiment outputs.

## Repository Structure

```text
.
├── train_single.py                         # Train seq2seq generative recommender
├── evaluate_single.py                      # Full-ranking generation evaluation
├── tools/
│   ├── build_varlen_sid_index.py           # IST: build variable-length SID index
│   ├── build_prefix_pair_augmentation.py   # BCA: behavior prefix-pair replay
│   └── build_seq2seq_sliding_dataset.py    # Build fixed-target seq2seq splits
├── util/                                   # Tokenizer, datacollators, evaluation/runtime helpers
├── llamarec/                               # Lightweight recommender backbones
├── quantization/                           # Minimal quantizer I/O helpers
└── pretrain_config/                        # Example training configs
```

## Data Format

Prepare a dataset directory under `data/<DATASET_NAME>/` with files such as:

```text
<DATASET_NAME>.index.json      # item_id -> ["<a_i>", "<b_j>", ...]
<DATASET_NAME>.sid2pid.json    # SID string -> item ids / candidates
<DATASET_NAME>.inter.json      # user_id -> chronological item_id sequence
train_data.json                # optional existing seq2seq train samples
val_data.json
test_data.json
```

Datasets are not included in this repository.

## IST: Build Variable-Length SID Index

```bash
python tools/build_varlen_sid_index.py \
  --input-dir data/Beauty_TIGER \
  --output-dir data/Beauty_TIGER_varlen_cond_entropy \
  --selection-strategy cond_entropy_bpe \
  --top-k 128 \
  --min-weighted-freq 20 \
  --min-item-support 10 \
  --min-pattern-usage 10 \
  --overwrite
```

Supported merge criteria include `bpe`, `wordpiece`, and `cond_entropy_bpe`.

## Build Fixed-Target Seq2Seq Splits

SST keeps target SIDs fixed-length while rewriting history-side SIDs.

```bash
python tools/build_seq2seq_sliding_dataset.py \
  --output-dir data/Beauty_TIGER_varlen_cond_entropy \
  --inter-path data/Beauty_TIGER/Beauty.inter.json \
  --history-index-path data/Beauty_TIGER_varlen_cond_entropy/Beauty.index.json \
  --target-index-path data/Beauty_TIGER/Beauty.index.json \
  --prefix seq2seq_letter_fixed_target \
  --max-history-items 20
```

## BCA: Build Behavior-Augmented Training Data

```bash
python tools/build_prefix_pair_augmentation.py \
  --source-dataset-dir data/Beauty_TIGER \
  --target-train-files data/Beauty_TIGER_varlen_cond_entropy/train_seq2seq_letter_fixed_target.json \
  --output-suffix bca \
  --augmentation-mode sequence_replay \
  --prefix-tokens 2 \
  --window-size 3 \
  --top-k 128 \
  --min-count 3 \
  --max-augmentations-per-pair 50
```

## Training

Edit a YAML file under `pretrain_config/` to point to your local dataset paths, then run:

```bash
python train_single.py \
  --dataset Beauty_TIGER_varlen_cond_entropy_bca \
  --model_name seq2seq_t5-rec
```

## Evaluation

```bash
python evaluate_single.py \
  --dataset Beauty_TIGER_varlen_cond_entropy_bca \
  --model_name seq2seq_t5-rec \
  --checkpoint experiment/<RUN_DIR>/best_model \
  --eval_split test \
  --generation_constraint full_trie
```

## Notes

- The public repository is code-only by design.
- Large files such as datasets, checkpoints, generated SIDs, and logs should remain outside Git.
- Example configs are templates; update paths before running experiments.
