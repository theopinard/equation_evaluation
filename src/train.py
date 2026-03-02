"""Training entry point for GRU and Transformer models.

Usage:
    uv run src/train.py gru
    uv run src/train.py transformer
    uv run src/train.py all

Checkpoints are saved to ./training_checkpoints/{model}/ after each epoch.
Delete the checkpoint folder before switching architectures to avoid loading
incompatible weights.
"""
import importlib
import os
import sys

import tensorflow as tf

sys.path.insert(0, os.path.dirname(__file__))
from dataset import VOCAB, EOS, build_eq_table, pad_eq_table, make_streaming_dataset

TRAINING_EXAMPLES = 128 * 128 * 16
BATCH_SIZE = 128
BUFFER_SIZE = 10000
EPOCHS = 1
CHUNK_SIZE = 4096

CHECKPOINT_DIR = "./training_checkpoints"


def main(model_name):
    module = importlib.import_module(model_name)

    char2idx = {u: i for i, u in enumerate(VOCAB)}
    eos_idx = char2idx[EOS]
    vocab_size = len(VOCAB)

    eq_array = pad_eq_table(
        [[char2idx[c] for c in eq] for eq in build_eq_table(range(10))],
        eos_idx,
    )
    dataset = make_streaming_dataset(
        eq_array, TRAINING_EXAMPLES, CHUNK_SIZE, BATCH_SIZE, BUFFER_SIZE
    )

    model = module.build_model(vocab_size, batch_size=BATCH_SIZE)
    model.compile(
        optimizer="adam",
        loss=lambda y, logits: tf.keras.losses.sparse_categorical_crossentropy(
            y, logits, from_logits=True
        ),
    )

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, model_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, "ckpt_{epoch}.weights.h5"),
        save_weights_only=True,
    )

    model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("gru", "transformer", "all"):
        print("Usage: train.py [gru|transformer|all]")
        sys.exit(1)
    models = ("gru", "transformer") if sys.argv[1] == "all" else (sys.argv[1],)
    for model_name in models:
        main(model_name)
