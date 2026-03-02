# equation_evaluation

A character-level language model trained to evaluate simple single-digit arithmetic equations (e.g. `3*7=21`). Two architectures are supported.

## Architecture

Both models are trained as autoregressive next-character predictors on equations of the form `XopY=result\n`.

| Model | File | Notes |
|---|---|---|
| GRU | `src/gru.py` | Stateful recurrent model |
| Transformer | `src/transformer.py` | Decoder-only with causal self-attention |

## Requirements

- Python 3.9+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

## Setup

```bash
uv sync
```

## Training

```bash
uv run src/train.py gru
uv run src/train.py transformer
uv run src/train.py all
```

Checkpoints are saved to `./training_checkpoints/{model}/ckpt_{epoch}.weights.h5` after each epoch. Delete the model's checkpoint folder before switching architectures to avoid loading incompatible weights.

## Evaluation

```bash
uv run src/evaluate.py gru
uv run src/evaluate.py transformer
uv run src/evaluate.py all
```

Loads the latest checkpoint, generates 20 fresh equations, and compares the model's predicted result to the correct answer:

```
Loaded checkpoint: ./training_checkpoints/transformer/ckpt_1.weights.h5  (model: transformer)

[OK  ]  3*7=21        |  predicted: 21
[FAIL]  8/4=2.0       |  predicted: 1.0
[OK  ]  6+5=11        |  predicted: 11
...
Accuracy: 17/20  (85.0%)
```

## Tests

End-to-end overfitting tests verify that each architecture can learn a reduced dataset (digits 0 and 1 only, ~14 equations):

```bash
uv run pytest tests/ -v
```

Each model is trained until it can correctly complete every training equation from scratch (100% training accuracy), confirming the architecture is capable of learning the task.

## Project structure

```
src/
  gru.py          # GRU architecture + generate()
  transformer.py  # Transformer architecture + generate()
  dataset.py      # Shared dataset utilities (VOCAB, build_eq_table, etc.)
  train.py        # Training entry point (accepts gru|transformer)
  evaluate.py     # Inference and accuracy reporting
  eqgen.py        # Random equation generator
tests/
  test_overfit.py  # End-to-end overfitting tests
```
