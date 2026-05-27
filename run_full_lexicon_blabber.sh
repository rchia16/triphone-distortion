#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON:-python}"
PROCESSOR="${PROCESSOR:-deep_phone_candidate_stack_blabber_triphone.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/raqchia/audio-assets/speech-assets-triphone}"
LEVELS="${LEVELS:-8}"
PIPER_LENGTH_SCALE="${PIPER_LENGTH_SCALE:-1.2}"
TTS_ENGINE="${TTS_ENGINE:-piper}"
EDGE_RATE="${EDGE_RATE:-+0%}"
EDGE_PITCH="${EDGE_PITCH:-+0Hz}"
EDGE_FEMALE_VOICE="${EDGE_FEMALE_VOICE:-en-AU-NatashaNeural}"
EDGE_MALE_VOICE="${EDGE_MALE_VOICE:-en-AU-WilliamNeural}"
TEXT_ONLY="${TEXT_ONLY:-0}"

WORDS=(go stop bath food yes no pain help)
VOICES=(man woman)

extra_args=()
EXTRA_ARGS_COUNT=0
for arg in "$@"; do
  case "$arg" in
    --text-only)
      TEXT_ONLY=1
      ;;
    *)
      extra_args[EXTRA_ARGS_COUNT]="$arg"
      EXTRA_ARGS_COUNT=$((EXTRA_ARGS_COUNT + 1))
      ;;
  esac
done

run_processor() {
  local argv=("$PYTHON_BIN" "$PROCESSOR" "$@")
  if ((EXTRA_ARGS_COUNT > 0)); then
    argv+=("${extra_args[@]}")
  fi
  "${argv[@]}"
}

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

edge_voice_for() {
  case "$1" in
    woman) printf '%s\n' "$EDGE_FEMALE_VOICE" ;;
    man) printf '%s\n' "$EDGE_MALE_VOICE" ;;
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
  local voice word model config ref_wav edge_voice

  if ! "$PYTHON_BIN" --version >/dev/null 2>&1; then
    missing+=("python executable: $PYTHON_BIN")
  fi

  [[ -f "$PROCESSOR" ]] || missing+=("$PROCESSOR")

  if ((TEXT_ONLY)); then
    if ! command -v jq >/dev/null 2>&1; then
      missing+=("jq")
    fi
    if ((${#missing[@]} > 0)); then
      printf 'Missing required inputs; no processing was started:\n' >&2
      printf '  %s\n' "${missing[@]}" >&2
      exit 1
    fi
    return 0
  fi

  case "$TTS_ENGINE" in
    piper)
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
      ;;
    edge)
      if ! command -v ffmpeg >/dev/null 2>&1; then
        missing+=("ffmpeg")
      fi
      if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import edge_tts
PY
      then
        missing+=("python module: edge_tts")
      fi
      for voice in "${VOICES[@]}"; do
        edge_voice="$(edge_voice_for "$voice")"
        if [[ -z "$edge_voice" ]]; then
          missing+=("edge voice for $voice")
        fi
      done
      ;;
    *)
      printf 'Unknown TTS_ENGINE: %s\n' "$TTS_ENGINE" >&2
      exit 1
      ;;
  esac

  if ((${#missing[@]} > 0)); then
    printf 'Missing required inputs; no processing was started:\n' >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
  fi
}

run_word_text_only() {
  local word="$1"
  local word_output_dir json_tmp text_out

  word_output_dir="$OUTPUT_ROOT/text/triphone/$word"
  text_out="$word_output_dir/${word}_combinations.txt"

  mkdir -p "$word_output_dir"
  json_tmp="$word_output_dir/.${word}_blabber_tmp.json"

  printf '\n[text/%s] combinations=%s\n' "$word" "$text_out"
  run_processor \
    --word "$word" \
    --level 1.0 \
    --render-dir "$word_output_dir" \
    --json-out "$json_tmp" \
    --skip-neural

  jq -r '
    [.assets[]?]
    | sort_by((.phone_distance // 0), (.candidate_text // ""))
    | .[] .candidate_text // empty
  ' "$json_tmp" > "$text_out"
  rm -f "$json_tmp"
}

run_word_voice() {
  local voice="$1"
  local word="$2"
  local model config ref_wav word_output_dir json_out edge_voice

  word_output_dir="$OUTPUT_ROOT/$voice/triphone/$word"
  json_out="$word_output_dir/${word}_blabber_scored.json"

  mkdir -p "$word_output_dir"

  case "$TTS_ENGINE" in
    piper)
      model="$(model_for_voice "$voice")"
      config="$(config_for_voice "$voice")"
      ref_wav="$(reference_wav_for "$voice" "$word")"

      printf '\n[%s/%s] reference=%s\n' "$voice" "$word" "$ref_wav"
      run_processor \
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
        --piper-use-python-module
      ;;
    edge)
      edge_voice="$(edge_voice_for "$voice")"

      printf '\n[%s/%s] edge-voice=%s mode=render-only\n' "$voice" "$word" "$edge_voice"
      run_processor \
        --word "$word" \
        --levels "$LEVELS" \
        --render-dir "$word_output_dir" \
        --json-out "$json_out" \
        --tts-engine edge \
        --edge-voice "$edge_voice" \
        --edge-rate "$EDGE_RATE" \
        --edge-pitch "$EDGE_PITCH" \
        --voice-mode "$voice" \
        --skip-neural \
        --render-audio
      ;;
    *)
      printf 'Unknown TTS_ENGINE: %s\n' "$TTS_ENGINE" >&2
      exit 1
      ;;
  esac
}

main() {
  local voice word

  preflight

  if ((TEXT_ONLY)); then
    for word in "${WORDS[@]}"; do
      run_word_text_only "$word"
    done
    printf '\nCompleted text-only lexicon generation under %s/text/triphone\n' "$OUTPUT_ROOT"
    return 0
  fi

  for voice in "${VOICES[@]}"; do
    for word in "${WORDS[@]}"; do
      run_word_voice "$voice" "$word"
    done
  done

  printf '\nCompleted full lexicon generation under %s\n' "$OUTPUT_ROOT"
}

main "$@"
