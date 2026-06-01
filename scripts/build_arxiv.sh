#!/usr/bin/env bash
#
# Build a flat, arXiv-valid source bundle for the ICML workshop paper.
#
# Why this exists: arXiv compiles the *source* itself (pdflatex, and it does NOT
# run bibtex -- it uses the bundled .bbl), and it requires the main .tex at the
# archive root with NO parent-escaping ("../") member paths. Our paper keeps
# figures/tables in directories ABOVE paper/ (../figures, ../results,
# ../investigations), so a raw arxiv-collector bundle has ../ paths that arXiv
# rejects. This script:
#   1. runs arxiv-collector to gather exactly the files latexmk actually used
#      and to bundle the .bbl (comments kept -> preserves numbers-from-scripts
#      and bib-from-ids provenance comments),
#   2. normalizes the ../figures|results|investigations asset paths so the
#      assets sit under the bundle root and every .tex reference matches,
#   3. verifies the bundle compiles STANDALONE with the bundled .bbl (pdflatex
#      x3, no bibtex) to the expected page count, with no undefined refs/cites,
#      no missing files, real authors, and zero anonymization / concurrent-venue
#      ("NeurIPS") leak strings.
#
# Usage:  scripts/build_arxiv.sh
# Output: arxiv-build/icl-diversity-arxiv.tar.gz  (arxiv-build/ is gitignored)
#
# Fails loudly (set -euo pipefail) on any gathering, normalization, or
# verification failure -- never produces a silently-broken bundle.
set -euo pipefail

MAIN="main_icml_workshop"
EXPECTED_PAGES=28
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PAPER_DIR="$REPO_ROOT/paper"
OUT_DIR="$REPO_ROOT/arxiv-build"
TARBALL="$OUT_DIR/icl-diversity-arxiv.tar.gz"
STAGE="$(mktemp -d)"
VERIFY="$(mktemp -d)"
trap 'rm -rf "$STAGE" "$VERIFY"' EXIT

echo "==> 1/4 gathering with arxiv-collector (compiles to track deps)"
cd "$PAPER_DIR"
rm -f arxiv.tar.gz
uvx --from arxiv-collector arxiv-collector --no-strip-comments "$MAIN.tex"

echo "==> 2/4 normalizing ../ asset paths"
# GNU tar REFUSES to extract members containing '..' (security check fires before
# --transform), so use Python's tarfile to extract with the leading ../ and ./
# stripped from each member name -> assets land flat under STAGE.
python3 - "$PAPER_DIR/arxiv.tar.gz" "$STAGE" <<'PY'
import sys, tarfile
src, dest = sys.argv[1], sys.argv[2]
with tarfile.open(src) as t:
    for m in t.getmembers():
        name = m.name
        while name.startswith("../") or name.startswith("./"):
            name = name[3:] if name.startswith("../") else name[2:]
        name = name.lstrip("/")
        if not name:
            continue
        m.name = name
        t.extract(m, dest)
PY
rm -f "$PAPER_DIR/arxiv.tar.gz"
# Strip the matching ../ prefixes from every bundled .tex reference.
find "$STAGE" -name '*.tex' -print0 \
  | xargs -0 sed -i -E 's#\.\./(figures|results|investigations)/#\1/#g'
# Guard: no parent-escaping reference may survive.
if find "$STAGE" -name '*.tex' -print0 | xargs -0 grep -lE '\.\./(figures|results|investigations)/' 2>/dev/null; then
  echo "ERROR: ../ asset references survived normalization" >&2
  exit 1
fi
# Guard: main .tex and bundled .bbl must be at the archive root.
[ -f "$STAGE/$MAIN.tex" ] || { echo "ERROR: $MAIN.tex not at bundle root" >&2; exit 1; }
[ -f "$STAGE/$MAIN.bbl" ] || { echo "ERROR: bundled $MAIN.bbl missing" >&2; exit 1; }

echo "==> 3/4 packing flat tarball -> $TARBALL"
mkdir -p "$OUT_DIR"
tar czf "$TARBALL" -C "$STAGE" .
# Guard: no ../ members in the final archive.
if tar tzf "$TARBALL" | grep -qE '\.\./'; then
  echo "ERROR: tarball still contains ../ member paths" >&2
  exit 1
fi

echo "==> 4/4 verifying standalone compile (pdflatex x3, bundled .bbl, no bibtex)"
tar xzf "$TARBALL" -C "$VERIFY"
cd "$VERIFY"
for i in 1 2 3; do
  if ! pdflatex -interaction=nonstopmode -halt-on-error "$MAIN.tex" >/dev/null 2>&1; then
    echo "ERROR: pdflatex pass $i failed:" >&2
    tail -40 "$MAIN.log" >&2
    exit 1
  fi
done
# Undefined refs/cites or missing files in the log are hard failures.
if grep -qiE 'undefined (reference|citation|control)|^! .*not found|File .* not found' "$MAIN.log"; then
  echo "ERROR: undefined refs/cites or missing files in standalone compile:" >&2
  grep -iE 'undefined (reference|citation|control)|not found' "$MAIN.log" | sort -u >&2
  exit 1
fi
PAGES="$(pdfinfo "$MAIN.pdf" | awk '/^Pages:/ {print $2}')"
[ "$PAGES" = "$EXPECTED_PAGES" ] || { echo "ERROR: expected $EXPECTED_PAGES pages, got $PAGES" >&2; exit 1; }
pdftotext "$MAIN.pdf" - > body.txt 2>/dev/null
for pat in "Anonymous" "anonymous@" "4open.science" "under review" "do not distribute" "NeurIPS"; do
  if grep -qi "$pat" body.txt; then
    echo "ERROR: leak string '$pat' present in bundled PDF" >&2
    exit 1
  fi
done
grep -q "Khoriaty" body.txt || { echo "ERROR: real author block did not render" >&2; exit 1; }

SIZE="$(du -h "$TARBALL" | cut -f1)"
NFILES="$(tar tzf "$TARBALL" | grep -vc '/$' || true)"
echo
echo "PASS: $TARBALL ($SIZE, $NFILES files)"
echo "      standalone compile = $PAGES pages, 0 undefined refs/cites, 0 missing files,"
echo "      real authors render, 0 anonymization/NeurIPS leak strings."
