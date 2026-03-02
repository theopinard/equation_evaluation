# CLAUDE.md

Developer notes for working on this codebase.

## File map

| File | Purpose |
|---|---|
| `src/gru.py` | GRU model: `NAME`, `build_model(vocab_size, batch_size)`, `generate(model, prefix_indices, idx2char, eos_idx)` |
| `src/transformer.py` | Transformer model: same interface as `gru.py` |
| `src/dataset.py` | Shared constants (`VOCAB`, `EOS`) and dataset utilities used by training and tests |
| `src/train.py` | CLI entry point — imports the requested model module by name (`importlib.import_module`) |
| `src/evaluate.py` | CLI entry point — loads latest checkpoint and reports accuracy on fresh equations |
| `src/eqgen.py` | Standalone random equation generator used by `evaluate.py` |
| `tests/test_overfit.py` | Parametrized overfit tests — imports `gru` and `transformer` modules directly |

## Model interface contract

Each model module must export:

```python
NAME: str                          # "gru" or "transformer"
build_model(vocab_size, batch_size) -> tf.keras.Model
generate(model, prefix_indices, idx2char, eos_idx, max_len=30) -> str
```

`prefix_indices` is a Python list of integer token indices (not a string).

## GRU statefulness

The GRU uses `stateful=True` and a fixed `batch_shape` in `tf.keras.Input`. This means:

- The model object is tied to a specific `batch_size` at build time — rebuild with `batch_size=1` for inference.
- Hidden state persists across `model()` calls within a session; call `layer.reset_states()` on every GRU layer before each new sequence.
- In the overfit test, `batch_size=1` + `_ResetStatePerSequence` callback ensure each training sequence starts from a zero hidden state, matching inference behaviour.

## Transformer loss floor

With a causal mask, the first two token positions are genuinely ambiguous — seeing `"0"` alone doesn't reveal which operator follows. The per-token cross-entropy therefore cannot fall below the dataset entropy floor (~0.22 nats). The overfit tests use **inference accuracy** (give "XopY=", check the generated result) rather than a loss threshold, which is a fair metric for both architectures.

## Checkpoint layout

```
training_checkpoints/
  gru/
    ckpt_1.weights.h5
    ckpt_2.weights.h5
    ...
  transformer/
    ckpt_1.weights.h5
    ...
```

`evaluate.py` and `train.py` both derive the checkpoint path from the model name argument, so there is no shared state between the two models.

## Vocabulary

The shared `VOCAB = "0123456789.+-*/=\n"` is defined in `dataset.py`. The overfit tests derive a smaller dynamic vocab from only the characters that appear in the tiny (digits 0–1) dataset, keeping training fast.
