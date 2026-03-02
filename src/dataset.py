"""Shared dataset utilities used by training and tests."""
import numpy as np
import tensorflow as tf

VOCAB = "0123456789.+-*/=\n"  # \n is the end-of-equation token
EOS = "\n"


def build_eq_table(digits):
    """Return all valid equation strings for the given digit values.

    Args:
        digits: iterable of ints, e.g. range(10) for full dataset or range(2)
                for the tiny test dataset.

    Returns:
        List of strings like "3+4=7\\n".
    """
    eqs = []
    for a in digits:
        for op in "+-*/":
            for b in digits:
                formula = f"{a}{op}{b}"
                try:
                    result = str(round(eval(formula), 5))
                    eqs.append(f"{formula}={result}{EOS}")
                except ZeroDivisionError:
                    pass
    return eqs


def pad_eq_table(eq_table, eos_idx):
    """Pad a list of index sequences to uniform length.

    Args:
        eq_table: list of int lists (character indices)
        eos_idx:  index used for right-padding shorter sequences

    Returns:
        numpy int32 array of shape (n_equations, max_len)
    """
    max_len = max(len(s) for s in eq_table)
    return np.array(
        [s + [eos_idx] * (max_len - len(s)) for s in eq_table],
        dtype=np.int32,
    )


def make_streaming_dataset(eq_array, n_samples, chunk_size, batch_size, buffer_size):
    """Build a streaming TF dataset by randomly sampling from a pre-computed equation array.

    Equations are sampled in chunks to amortise Python call overhead, then
    flattened, split into (input, target) pairs, shuffled, and batched.

    Args:
        eq_array:    numpy array of shape (n_eq, seq_len) — padded equation indices
        n_samples:   total number of samples to draw per epoch
        chunk_size:  equations sampled per py_function call
        batch_size:  training batch size
        buffer_size: shuffle buffer size

    Returns:
        tf.data.Dataset yielding (input, target) batches of shape
        (batch_size, seq_len - 1).
    """
    max_eq_len = eq_array.shape[1]

    def generate_chunk(_):
        indices = np.random.randint(0, len(eq_array), size=chunk_size)
        return eq_array[indices]

    def split_input_target(seq):
        return seq[:-1], seq[1:]

    return (
        tf.data.Dataset.range(n_samples // chunk_size)
        .map(
            lambda i: tf.py_function(func=generate_chunk, inp=[i], Tout=tf.int32),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        .map(lambda chunk: tf.ensure_shape(chunk, [chunk_size, max_eq_len]))
        .flat_map(lambda chunk: tf.data.Dataset.from_tensor_slices(chunk))
        .map(split_input_target)
        .shuffle(buffer_size)
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )
