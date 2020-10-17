import os

import tensorflow as tf
import numpy as np
from eqgen import simple_eq

vocab = "0123456789.+-*/= "

TRAINING_EXAMPLE = 128
BATCH_SIZE = 128
BUFFER_SIZE = 10000
EPOCHS = 100


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 8

# Number of RNN units
rnn_units = 64


# Creating a mapping from unique characters to indices
char2idx = {u: i for i, u in enumerate(vocab)}
idx2char = np.array(list(vocab))

eq_as_index = []
for _ in range(TRAINING_EXAMPLE):
    eq = [char2idx[char] for char in simple_eq()]
    eq_as_index.append(np.array(eq))

dataset = tf.data.Dataset.from_tensor_slices(eq_as_index)


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


dataset = dataset.map(split_input_target).shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Embedding(
                vocab_size, embedding_dim, batch_input_shape=[batch_size, None]
            ),
            tf.keras.layers.GRU(
                rnn_units,
                return_sequences=True,
                stateful=True,
                recurrent_initializer="glorot_uniform",
            ),
            tf.keras.layers.Dense(vocab_size),
        ]
    )
    return model


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE,
)


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True
    )


model.compile(optimizer="adam", loss=loss)


# Directory where the checkpoints will be saved
checkpoint_dir = "./training_checkpoints"
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix, save_weights_only=True
)


if __name__ == "__main__":
    print(dataset)
    for input_example, target_example in dataset.take(5):
        print("Input data: ", repr("".join(idx2char[input_example.numpy()[0]])))
        print("Target data:", repr("".join(idx2char[target_example.numpy()[0]])))

    print(dataset.take(1))
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        print(
            example_batch_predictions.shape,
            "# (batch_size, sequence_length, vocab_size)",
        )
        print(input_example_batch.shape, "input_example_batch")
        print(target_example_batch.shape, "target_example_batch")
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    print("Input: \n", repr("".join(idx2char[input_example_batch][0])))
    print()
    print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices])))
    history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])
