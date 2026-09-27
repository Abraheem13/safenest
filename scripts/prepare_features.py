"""Parse every available corpus once and write per-window feature tables.

    python3 scripts/prepare_features.py            # all corpora present locally
    python3 scripts/prepare_features.py persuade   # one corpus

Outputs, under data/processed/:
    <corpus>_docs.parquet          one row per document (metadata, no text)
    <corpus>_w<W>.parquet          one row per (document, window) for W in WINDOWS,
                                   with the five features and the window's words
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safenest.data import available, load  # noqa: E402
from safenest.data.registry import PROCESSED  # noqa: E402
from safenest.features import Featurizer  # noqa: E402

WINDOWS = (25, 50, 100)
N_PROCESS = 6


def prepare(name: str, featurizer: Featurizer) -> None:
    t0 = time.time()
    docs = load(name)
    docs = docs[docs["admissible"] != ""].reset_index(drop=True)
    spoken = docs["modality"].iloc[0] == "spoken"
    parsed = featurizer.parse(docs["text"].tolist(), spoken=spoken)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    docs.drop(columns=["text"]).to_parquet(PROCESSED / f"{name}_docs.parquet", index=False)
    for w in WINDOWS:
        feats = featurizer.featurise(docs, w, parsed)
        feats.to_parquet(PROCESSED / f"{name}_w{w}.parquet", index=False)
        print(f"  {name}: window {w:3d} words -> {len(feats):6d} windows "
              f"from {feats['doc_id'].nunique():5d} documents")
    print(f"  {name}: {len(docs)} documents in {time.time() - t0:.0f}s")


def main(names: list[str]) -> int:
    names = names or available()
    featurizer = Featurizer(n_process=N_PROCESS)
    for name in names:
        prepare(name, featurizer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
