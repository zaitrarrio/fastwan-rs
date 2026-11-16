"""Print the gallery safetensors path for a prompt.

Exit 0 and print the path when the prompt is listed. Exit 1 when the gallery
or the prompt is absent. Exit 2 on a broken index. Matching ignores repeated
whitespace, same as fastwan's gallery lookup.
"""

import json
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: resolve-embed.py GALLERY PROMPT", file=sys.stderr)
        return 2
    gallery = Path(sys.argv[1])
    want = " ".join(sys.argv[2].split())
    index_path = gallery / "gallery.json"
    if not index_path.is_file():
        return 1
    try:
        index = json.loads(index_path.read_text())
    except json.JSONDecodeError as exc:
        print(f"{index_path}: {exc}", file=sys.stderr)
        return 2
    for entry in index.get("prompts", []):
        if " ".join(str(entry.get("prompt", "")).split()) != want:
            continue
        path = gallery / entry["file"]
        if not path.is_file():
            print(f"missing {path}", file=sys.stderr)
            return 2
        print(path)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
