"""GRU character-level language model."""
import numpy as np
import tensorflow as tf

NAME = "gru"

EMBEDDING_DIM = 8
UNITS = 64


def build_model(vocab_size, batch_size):
    inputs = tf.keras.Input(batch_shape=(batch_size, None), dtype=tf.int32)
    x = tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM)(inputs)
    x = tf.keras.layers.GRU(
        UNITS,
        return_sequences=True,
        stateful=True,
        recurrent_initializer="glorot_uniform",
    )(x)
    outputs = tf.keras.layers.Dense(vocab_size)(x)
    return tf.keras.Model(inputs, outputs)


def generate(model, prefix_indices, idx2char, eos_idx, max_len=30):
    """Stateful step-by-step generation.

    Resets hidden state, warms up on prefix, then decodes greedily until EOS.
    prefix_indices is a list of integer token indices.
    """
    for layer in model.layers:
        if hasattr(layer, "reset_states"):
            layer.reset_states()
    for c in prefix_indices[:-1]:
        model(np.array([[c]]))
    idx = np.array([[prefix_indices[-1]]])
    out = []
    for _ in range(max_len):
        pred = model(idx)                          # (1, 1, vocab_size)
        pi = int(tf.argmax(pred[0, 0]))
        if pi == eos_idx:
            break
        out.append(idx2char[pi])
        idx = np.array([[pi]])
    return "".join(out)
