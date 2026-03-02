"""Decoder-only Transformer character-level language model."""
import numpy as np
import tensorflow as tf

NAME = "transformer"

EMBED_DIM = 64
NUM_HEADS = 4
FF_DIM = 128
NUM_LAYERS = 2
MAX_SEQ_LEN = 32  # must be >= longest equation


class _TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, max_seq_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(max_seq_len, embed_dim)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        positions = tf.range(seq_len)
        return self.token_emb(x) + self.pos_emb(positions)


def build_model(vocab_size, batch_size):
    inputs = tf.keras.Input(batch_shape=(batch_size, None), dtype=tf.int32)
    x = _TokenAndPositionEmbedding(MAX_SEQ_LEN, vocab_size, EMBED_DIM)(inputs)
    for _ in range(NUM_LAYERS):
        attn = tf.keras.layers.MultiHeadAttention(
            num_heads=NUM_HEADS,
            key_dim=EMBED_DIM // NUM_HEADS,
        )(x, x, use_causal_mask=True)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn)
        ff = tf.keras.layers.Dense(FF_DIM, activation="relu")(x)
        ff = tf.keras.layers.Dense(EMBED_DIM)(ff)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)
    outputs = tf.keras.layers.Dense(vocab_size)(x)
    return tf.keras.Model(inputs, outputs)


def generate(model, prefix_indices, idx2char, eos_idx, max_len=30):
    """Full-context autoregressive generation.

    Feeds the growing context at each step, reads the prediction from the last
    position. prefix_indices is a list of integer token indices.
    """
    context = list(prefix_indices)
    out = []
    for _ in range(max_len):
        pred = model(np.array([context]))          # (1, seq_len, vocab_size)
        pi = int(tf.argmax(pred[0, -1]))
        if pi == eos_idx:
            break
        out.append(idx2char[pi])
        context.append(pi)
    return "".join(out)
