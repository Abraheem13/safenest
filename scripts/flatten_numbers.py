"""Write a copy of a manuscript with every \\val{name} replaced by its value.

    python3 scripts/flatten_numbers.py path/to/main.tex path/to/numbers.json out.tex

Journals typeset from plain LaTeX, so the submitted source carries literal
numbers. The flattened copy is produced mechanically from the same
`numbers.json` the build used, so the two cannot differ.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path


def main(src: str, values: str, dst: str) -> int:
    text = Path(src).read_text()
    table = json.loads(Path(values).read_text())
    missing = sorted(set(re.findall(r"\\val\{([A-Za-z]+)\}", text)) - set(table))
    if missing:
        print(f"undefined values: {missing}")
        return 1
    text = re.sub(r"\\val\{([A-Za-z]+)\}", lambda m: table[m.group(1)], text)
    text = text.replace("\\input{tex/numbers}\n", "")
    Path(dst).write_text(text)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
