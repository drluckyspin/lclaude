#!/bin/bash

# -----------------------------------------------------------------------------------------------------------
# Script Name: bump-version.sh
#
# Description: Update the repository VERSION file and synchronize that value to
#              the __version__ variable in every top-level Python script.
#
# Usage:
#   make bump-version 0.4.0
# -----------------------------------------------------------------------------------------------------------

set -euo pipefail

# Require Make's positional version argument rather than silently choosing one.
if [[ "$#" -ne 1 ]]; then
    echo "Usage: make bump-version <version>" >&2
    exit 1
fi

# Resolve paths from this script, so the command works from any current directory.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
version_file="$repo_root/VERSION"
version="$1"

# Accept semantic versions, including prerelease or build suffixes.
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-.][0-9A-Za-z.-]+)?$ ]]; then
    echo "Version must use semantic versioning, for example: 0.4.0" >&2
    exit 1
fi

# VERSION is the checked-in release source of truth.
printf '%s\n' "$version" > "$version_file"

# Keep each executable script's runtime version in sync with VERSION. The
# embedded Python makes the multiline-safe replacement without external tools.
python3 - "$repo_root" "$version" <<'PY'
from pathlib import Path
import re
import sys

repo_root = Path(sys.argv[1])
version = sys.argv[2]

# Every top-level Python entry point must define exactly one __version__ value.
for path in sorted(repo_root.glob("*.py")):
    text = path.read_text()
    updated, count = re.subn(
        r'^__version__ = "[^"]+"$',
        f'__version__ = "{version}"',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit(f"Could not update __version__ in {path}")
    path.write_text(updated)
PY
