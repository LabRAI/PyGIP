#!/usr/bin/env bash
set -euo pipefail

SRC=${1:-origin/feature-genie-extraction-final}

# Compute files that differ between labrai/main and the source branch
FILES=$(git diff --name-only labrai/main.."$SRC")
if [ -z "$FILES" ]; then
  echo "No files to import from $SRC. Check that $SRC exists and has diffs vs labrai/main."
  exit 1
fi

echo "Importing these files from $SRC:"
printf '%s\n' "$FILES"

for f in $FILES; do
  echo " -> $f"
  mkdir -p "$(dirname "$f")" 2>/dev/null || true
  # Use git show to write the file from the source branch
  git show "$SRC:$f" > "$f" || { echo "Failed to extract $f from $SRC"; exit 2; }
done

echo "Done."
