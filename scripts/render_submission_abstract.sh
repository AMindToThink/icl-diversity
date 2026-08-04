#!/usr/bin/env bash
# Expand paper macros into the abstract for conference submission forms.
# Uses flachtex (--newcommand) to inline every \newcommand from
# results/tables/paper_macros.tex into paper/sections/abstract_workshop.tex,
# then strips the prelude (definitions + comment headers) and the leftover
# `{}` empty-group markers that follow expanded macros.
#
# Output: paper/build/abstract_expanded.tex

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/paper/build"
WRAPPER="$BUILD_DIR/abstract_wrapper.tex"
OUT="$BUILD_DIR/abstract_expanded.tex"

if [[ ! -f "$WRAPPER" ]]; then
  echo "Missing wrapper: $WRAPPER" >&2
  exit 1
fi

cd "$BUILD_DIR"

uv run --with flachtex flachtex --newcommand "$(basename "$WRAPPER")" 2>/dev/null \
  | grep -vE '^(\\newcommand|%)' \
  | sed '/./,$!d' \
  | sed -E 's/([0-9.])\{\}/\1/g' \
  > "$OUT"

echo "Wrote $OUT"
