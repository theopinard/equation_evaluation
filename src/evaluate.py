"""Evaluate a trained model against freshly generated equations.

Usage:
    uv run src/evaluate.py gru
    uv run src/evaluate.py transformer
    uv run src/evaluate.py all
"""
import importlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from dataset import VOCAB, EOS
from eqgen import simple_eq

N_SAMPLES = 20
CHECKPOINT_DIR = "./training_checkpoints"


def latest_checkpoint(checkpoint_dir):
    files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".weights.h5")]
    if not files:
        return None

    def epoch_key(name):
        m = re.search(r"ckpt_(\d+)", name)
        return int(m.group(1)) if m else -1

    return os.path.join(checkpoint_dir, max(files, key=epoch_key))


def main(model_name):
    module = importlib.import_module(model_name)

    char2idx = {u: i for i, u in enumerate(VOCAB)}
    idx2char = list(VOCAB)
    eos_idx = char2idx[EOS]
    vocab_size = len(VOCAB)

    checkpoint_dir = os.path.join(CHECKPOINT_DIR, model_name)
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"Checkpoint directory '{checkpoint_dir}' not found. "
            f"Run `uv run src/train.py {model_name}` first."
        )
    checkpoint = latest_checkpoint(checkpoint_dir)
    if checkpoint is None:
        raise FileNotFoundError(
            f"No checkpoints in '{checkpoint_dir}'. "
            f"Run `uv run src/train.py {model_name}` first."
        )

    model = module.build_model(vocab_size, batch_size=1)
    model.load_weights(checkpoint)
    print(f"Loaded checkpoint: {checkpoint}  (model: {model_name})\n")

    correct = 0
    for _ in range(N_SAMPLES):
        eq = simple_eq()
        split = eq.index("=")
        prefix_str = eq[: split + 1]
        truth = eq[split + 1 :].rstrip(EOS)

        prefix_indices = [char2idx[c] for c in prefix_str]
        predicted = module.generate(model, prefix_indices, idx2char, eos_idx)

        match = truth == predicted
        correct += match

        status = "OK" if match else "FAIL"
        print(f"[{status:4}]  {prefix_str}{truth:<12}  |  predicted: {predicted}")

    print(f"\nAccuracy: {correct}/{N_SAMPLES}  ({100 * correct / N_SAMPLES:.1f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("gru", "transformer", "all"):
        print("Usage: evaluate.py [gru|transformer|all]")
        sys.exit(1)
    models = ("gru", "transformer") if sys.argv[1] == "all" else (sys.argv[1],)
    for model_name in models:
        main(model_name)
