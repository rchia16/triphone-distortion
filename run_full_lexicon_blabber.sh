#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON:-python}"
PROCESSOR="${PROCESSOR:-deep_phone_candidate_stack_blabber_triphone.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/raqchia/audio-assets/speech-assets-triphone}"
LEVELS="${LEVELS:-8}"
PIPER_LENGTH_SCALE="${PIPER_LENGTH_SCALE:-1.2}"

WORDS=(go stop bath food yes no pain help)
VOICES=(man woman)

extra_args=("$@")

model_for_voice() {
  case "$1" in
    woman) printf '%s\n' "piper-models/en_GB-jenny_dioco-medium.onnx" ;;
    man) printf '%s\n' "piper-models/en_GB-northern_english_male-medium.onnx" ;;
    *) printf 'Unknown voice: %s\n' "$1" >&2; return 1 ;;
  esac
}

config_for_voice() {
  case "$1" in
    woman) printf '%s\n' "piper-models/en_GB-jenny_dioco-medium.onnx.json" ;;
    man) printf '%s\n' "piper-models/en_GB-northern_english_male-medium.onnx.json" ;;
    *) printf 'Unknown voice: %s\n' "$1" >&2; return 1 ;;
  esac
}

reference_wav_for() {
  local voice="$1"
  local word="$2"
  printf 'aus-en_%s/aus-en_%s_%s.wav\n' "$voice" "$voice" "$word"
}

preflight() {
  local missing=()
  local voice word model config ref_wav

  if ! "$PYTHON_BIN" --version >/dev/null 2>&1; then
    missing+=("python executable: $PYTHON_BIN")
  fi

  [[ -f "$PROCESSOR" ]] || missing+=("$PROCESSOR")

  for voice in "${VOICES[@]}"; do
    model="$(model_for_voice "$voice")"
    config="$(config_for_voice "$voice")"
    [[ -f "$model" ]] || missing+=("$model")
    [[ -f "$config" ]] || missing+=("$config")

    for word in "${WORDS[@]}"; do
      ref_wav="$(reference_wav_for "$voice" "$word")"
      [[ -f "$ref_wav" ]] || missing+=("$ref_wav")
    done
  done

  if ((${#missing[@]} > 0)); then
    printf 'Missing required inputs; no processing was started:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
  fi
}

run_word_voice() {
  local voice="$1"
  local word="$2"
  local model config ref_wav word_output_dir json_out

  model="$(model_for_voice "$voice")"
  config="$(config_for_voice "$voice")"
  ref_wav="$(reference_wav_for "$voice" "$word")"
  word_output_dir="$OUTPUT_ROOT/$voice/triphone/$word"
  json_out="$word_output_dir/${word}_blabber_scored.json"

  mkdir -p "$word_output_dir"

  printf '\n[%s/%s] reference=%s\n' "$voice" "$word" "$ref_wav"
  "$PYTHON_BIN" "$PROCESSOR" \
    --word "$word" \
    --levels "$LEVELS" \
    --reference-wav "$ref_wav" \
    --render-dir "$word_output_dir" \
    --json-out "$json_out" \
    --tts-engine piper \
    --piper-model "$model" \
    --piper-config "$config" \
    --piper-length-scale "$PIPER_LENGTH_SCALE" \
    --voice-mode "$voice" \
    --piper-use-python-module \
    "${extra_args[@]}"
}

main() {
  local voice word

  preflight

  for voice in "${VOICES[@]}"; do
    for word in "${WORDS[@]}"; do
      run_word_voice "$voice" "$word"
    done
  done

  printf '\nCompleted full lexicon generation under %s\n' "$OUTPUT_ROOT"
}

main "$@"
