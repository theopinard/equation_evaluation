"""End-to-end overfitting tests for each model architecture.

Uses a reduced vocabulary (digits 0 and 1 only) to build a tiny dataset of
~14 equations.  Training correctness is verified via inference accuracy:
given the formula prefix "XopY=", each model must generate the correct result
for every training equation (i.e. achieve 100 % training accuracy).

The loss threshold approach is intentionally avoided for the transformer: with
a causal mask the first two positions are genuinely ambiguous (seeing "0" alone
does not reveal which operator follows), so the per-token cross-entropy cannot
go below the dataset entropy floor (~0.22 nats).  The GRU escapes this floor
via its cross-sequence hidden state, but that is a form of leakage, not better
learning of the task itself.  Inference accuracy is a fair metric for both.
"""
import os
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import gru as gru_module
import transformer as transformer_module
from dataset import EOS, build_eq_table, pad_eq_table


# ── Tiny dataset ──────────────────────────────────────────────────────────────

def _make_tiny_dataset():
    """Return all equations using only digits 0 and 1.

    The vocab is derived dynamically from the characters that actually appear,
    keeping it minimal for fast training.

    Returns (seqs, char2idx, idx2char, eos_idx, vocab_size).
    """
    eq_strings = build_eq_table(range(2))
    chars = set("".join(eq_strings))
    vocab = sorted(chars - {EOS}) + [EOS]      # EOS always last
    char2idx = {c: i for i, c in enumerate(vocab)}
    idx2char = list(vocab)
    eos_idx = char2idx[EOS]
    seqs = [[char2idx[c] for c in eq] for eq in eq_strings]
    return seqs, char2idx, idx2char, eos_idx, len(vocab)


# ── Training ──────────────────────────────────────────────────────────────────

class _ResetStatePerSequence(tf.keras.callbacks.Callback):
    """Reset stateful RNN state before every training batch (= every sequence when batch_size=1).

    This makes stateful-GRU training equivalent to stateful=False: each
    sequence is always processed from a zero hidden state, exactly matching
    the reset performed in gru.generate before inference.
    """
    def on_train_batch_begin(self, batch, logs=None):
        for layer in self.model.layers:
            if hasattr(layer, "reset_states"):
                layer.reset_states()


def _train(module, seqs, eos_idx, vocab_size, epochs, lr):
    padded = pad_eq_table(seqs, eos_idx)
    # For stateful GRU: batch_size=1 + reset-per-sequence callback so each
    # sequence is always seen from a zero state, matching inference exactly.
    batch_size = 1 if module.NAME == "gru" else len(seqs)
    dataset = (
        tf.data.Dataset.from_tensor_slices((padded[:, :-1], padded[:, 1:]))
        .batch(batch_size, drop_remainder=True)
        .repeat()
    )
    model = module.build_model(vocab_size, batch_size=batch_size)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=lambda y, logits: tf.keras.losses.sparse_categorical_crossentropy(
            y, logits, from_logits=True
        ),
    )
    callbacks = [_ResetStatePerSequence()] if module.NAME == "gru" else []
    model.fit(
        dataset,
        steps_per_epoch=len(seqs) // batch_size,
        epochs=epochs,
        callbacks=callbacks,
        verbose=0,
    )
    # Rebuild with batch_size=1 for inference and transfer weights
    inf_model = module.build_model(vocab_size, batch_size=1)
    inf_model.set_weights(model.get_weights())
    return inf_model


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "module,epochs,lr",
    [
        pytest.param(gru_module,         300, 1e-3, id="gru"),
        pytest.param(transformer_module, 300, 1e-3, id="transformer"),
    ],
)
def test_overfit(module, epochs, lr):
    """Each model must correctly complete every training equation after overfitting."""
    tf.random.set_seed(42)
    np.random.seed(42)

    seqs, char2idx, idx2char, eos_idx, vocab_size = _make_tiny_dataset()
    inf_model = _train(module, seqs, eos_idx, vocab_size, epochs=epochs, lr=lr)

    # For each training equation, give "XopY=" and verify the generated result
    correct = 0
    for seq in seqs:
        eq_pos = next(i for i, c in enumerate(seq) if idx2char[c] == "=")
        prefix = seq[: eq_pos + 1]              # indices up to and including "="
        truth = "".join(
            idx2char[c] for c in seq[eq_pos + 1 :] if c != eos_idx
        )
        predicted = module.generate(inf_model, prefix, idx2char, eos_idx)
        correct += predicted == truth

    accuracy = correct / len(seqs)
    assert accuracy == 1.0, (
        f"{module.NAME} reached only {accuracy:.0%} training accuracy "
        f"({correct}/{len(seqs)} equations correct)"
    )
