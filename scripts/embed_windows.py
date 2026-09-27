"""Sentence embeddings of every feature window, streamed to disk.

    python3 scripts/embed_windows.py written 50

Writes data/processed/<name>_w<W>_minilm.npy (float16) in the row order used by
the experiments. Encoding streams in small batches into a memory-mapped array,
so peak memory stays near the size of one batch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.real_common import EMBEDDER  # noqa: E402
from safenest.data.registry import PROCESSED  # noqa: E402

BATCH = 512


def main(name: str, window: int) -> int:
    if name == "written":
        from experiments.exp13_real_estimation import _load_written

        _, wins, _ = _load_written(window)
    else:
        from experiments.real_common import load_corpus

        _, wins = load_corpus(name, window)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDER, device="cpu")
    texts = wins["text"].tolist()
    dim = model.get_sentence_embedding_dimension()
    path = PROCESSED / f"{name}_w{window}_minilm.npy"
    out = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=(len(texts), dim))
    for start in range(0, len(texts), BATCH):
        chunk = texts[start:start + BATCH]
        out[start:start + len(chunk)] = model.encode(
            chunk, batch_size=64, show_progress_bar=False, normalize_embeddings=True)
        if start // BATCH % 50 == 0:
            print(f"  {start + len(chunk)}/{len(texts)}", flush=True)
    out.flush()
    print(f"  wrote {path} ({len(texts)} x {dim})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 50))
