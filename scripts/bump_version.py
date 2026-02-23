#!/usr/bin/env python3
import pathlib
import re
import sys


def main() -> int:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    version_file = repo_root / "VERSION"

    if not version_file.exists():
        print("ERROR: VERSION file not found", file=sys.stderr)
        return 1

    text = version_file.read_text(encoding="utf-8").strip()
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", text)
    if match is None:
        print(f"ERROR: Invalid VERSION format: '{text}'", file=sys.stderr)
        return 1

    major, minor, patch = (int(g) for g in match.groups())
    patch += 1
    bumped = f"{major}.{minor}.{patch}"
    version_file.write_text(f"{bumped}\n", encoding="utf-8")
    print(f"v{bumped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
