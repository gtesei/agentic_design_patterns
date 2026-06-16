#!/usr/bin/env bash
# Regenerate assets/hero.png — 2x2 montage of 4 representative pattern diagrams
# in the Da Vinci sepia/parchment palette.
#
# Requires: bun (for `bunx`), ImageMagick (`magick`).
# Mermaid CLI is fetched on-demand by bunx; no global install needed.

set -euo pipefail

cd "$(dirname "$0")"
OUT="../hero.png"
BG="#f5ecd9"
BORDER="#c5b393"

for f in multi-agent rag react tot; do
  bunx --bun @mermaid-js/mermaid-cli -i "$f.mmd" -o "$f.png" -s 2 -b "$BG"
done

magick montage multi-agent.png rag.png react.png tot.png \
  -tile 2x2 -geometry +20+20 \
  -background "$BG" -bordercolor "$BORDER" -border 8 \
  "$OUT"

echo "Wrote $OUT"
