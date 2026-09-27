"""Fetch the external corpora and verify them against recorded checksums.

    python3 scripts/download_data.py            # every corpus that can be fetched
    python3 scripts/download_data.py persuade   # one corpus

Nothing is redistributed with this repository. Files are written under
data/raw/ and checked against the SHA-256 recorded in
`safenest/data/registry.py`, which is the hash of the file the published results
were computed from. The CHILDES corpora require a (free) TalkBank login and
cannot be fetched by script; the command prints what to do instead.
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from safenest.data.registry import CORPORA, RAW  # noqa: E402

CHUNK = 1 << 20


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "safenest-data/2.0"})
    with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
        while block := r.read(CHUNK):
            f.write(block)
    tmp.rename(dest)


def main(names: list[str]) -> int:
    failures = 0
    for key in names or list(CORPORA):
        corpus = CORPORA[key]
        print(f"{corpus.name} ({corpus.licence})")
        if corpus.manual:
            for rf in corpus.files:
                dest = RAW / rf.path
                if not dest.exists():
                    print(f"  MISSING: data/raw/{rf.path}")
                    continue
                # TalkBank revises transcripts, so a later download can differ
                # from the file the reported results were computed from.
                ok = rf.sha256 is None or sha256(dest) == rf.sha256
                print(f"  {'ok' if ok else 'CHECKSUM MISMATCH'}: data/raw/{rf.path}")
                failures += not ok
            if not all((RAW / rf.path).exists() for rf in corpus.files):
                print(f"  {corpus.manual}")
            continue
        for rf in corpus.files:
            dest = RAW / rf.path
            if not dest.exists():
                print(f"  downloading {rf.path}")
                fetch(rf.url, dest)
            digest = sha256(dest)
            ok = rf.sha256 is None or digest == rf.sha256
            print(f"  {'ok' if ok else 'CHECKSUM MISMATCH'}: {rf.path}")
            failures += not ok
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
