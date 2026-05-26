#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
OUT_DIR="${OUT_DIR:-edge_concat_test}"
WORDS="${WORDS:-go,stop,bath,food,yes,no,pain,help}"
VOICES="${VOICES:-woman,man}"
LEVELS="${LEVELS:-8}"
MAX_CANDIDATES="${MAX_CANDIDATES:-64}"
EDGE_RATE="${EDGE_RATE:-+0%}"
EDGE_PITCH="${EDGE_PITCH:-+0Hz}"
CARRIER="${CARRIER:-Say {word}. {candidates}.}"

"$PYTHON_BIN" run_edge_concat_blabber_test.py \
  --python-bin "$PYTHON_BIN" \
  --words "$WORDS" \
  --voices "$VOICES" \
  --levels "$LEVELS" \
  --max-candidates "$MAX_CANDIDATES" \
  --out-dir "$OUT_DIR" \
  --edge-rate "$EDGE_RATE" \
  --edge-pitch "$EDGE_PITCH" \
  --carrier "$CARRIER" \
  "$@"
