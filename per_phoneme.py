"""
Per-phoneme phone stitching and EEG/preview-driven audio selection.

This script has two jobs:
  1. Build stitched word-audio libraries from isolated phoneme assets.
  2. Select the closest stitched WAV from a manual preview signal or EEG signal.

No TTS is used. Audio is stitched from a flat phoneme inventory, e.g.:
  /data/raqchia/audio-assets/aus-phones/cleaned/j.wav
  /data/raqchia/audio-assets/aus-phones/cleaned/ɛ.wav
  /data/raqchia/audio-assets/aus-phones/cleaned/s.wav

Words are resolved to ARPABET phonemes:
  yes -> Y EH S
  no  -> N OW

Each phoneme receives a distortion target in [0.0, 1.0].
Example:
  yes target_signal = [0.10, 0.72, 0.30]
means:
  Y  = low distortion
  EH = high distortion
  S  = moderate distortion

Granularity controls the available bins per phoneme:
  granularity=2 -> 0.000, 1.000
  granularity=4 -> 0.000, 0.333, 0.667, 1.000
  granularity=8 -> 0.000, 0.143, 0.286, 0.429, 0.571, 0.714, 0.857, 1.000

Generated libraries are stored by granularity:
  <output_root>/granularity_<N>/<voice>/<word>/

Example:
  per-phoneme/granularity_8/woman/yes/
    yes_per_phoneme_assets.json
    0.0_0.0_0.0.wav
    0.143_0.714_0.286.wav
    ...

Pregenerate stitched libraries:
  python per_phoneme.py --pregenerate-assets \
    --phone-inventory-root /data/raqchia/audio-assets/aus-phones/cleaned \
    --output-root per-phoneme \
    --granularity 8 \
    --words yes,no,bath,pain,help,food,go,stop \
    --voices man

Manual preview selection:
  python per_phoneme.py --word yes \
    --preview-signal 0.1,0.7,0.3,1.0 \
    --output-root per-phoneme \
    --granularity 8 \

EEG selection:
  python per_phoneme.py --eeg \
    --output-root per-phoneme \
    --granularity 8 \
    --requested-voice man \

EEG mode creates:
  EEG trial -> template comparison -> per_time_l2 -> normalized [0,1] signal
  -> one target per phoneme -> closest bin tuple -> selected stitched WAV

If the required granularity/voice/word library is missing during selection,
it is created automatically before choosing the closest WAV.

Important output fields:
  audio_path                 final selected stitched WAV
  target_signal              desired per-phoneme distortion values
  selected_phoneme_grades    closest granularity-bin tuple
  actual_phone_distances     normalized CMVN auditory distances
  generated_phones           phonemes stitched into the WAV
  cmvn_pairwise_distances    cleaned-phone CMVN/MFCC/DTW distance cache
"""

from __future__ import annotations

from pathlib import Path
import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import tempfile
import wave
from typing import Any

import numpy as np

from my_test import (
    _cmvn,
    _extract_mfcc,
    _resample_time_axis,
    _dtw_distance,
    CMVN_DTW_MAX_FRAMES,
)

from deep_phone_candidate_stack_blabber_triphone import (
    PHONE_FEATURES,
    variants_for_phone,
)
from select_blabber_asset import normalize_voice_mode


TOP_K_DEBUG = 10

PREVIEW_WORDS = ("yes", "no", "bath", "pain", "help", "food", "go", "stop")
PREVIEW_LEXICON: dict[str, list[str]] = {
    "go": ["G", "OW"],
    "stop": ["S", "T", "OW", "P"],
    "bath": ["B", "AE", "TH"],
    "food": ["F", "UW", "D"],
    "yes": ["Y", "EH", "S"],
    "no": ["N", "OW"],
    "pain": ["P", "EY", "N"],
    "help": ["HH", "EH", "L", "P"],
}

DEFAULT_PHONE_INVENTORY_ROOT = "/projects/SSNFB/Ray/audio-assets/aus-phones/cleaned"
DEFAULT_OUTPUT_ROOT = "/projects/SSNFB/Ray/audio-assets/per-phoneme"
DEFAULT_REQUESTED_VOICE = "man"
DEFAULT_MAX_PHONE_RANK = 4


ARPABET_TO_AUS_IPA_CANDIDATES: dict[str, list[str]] = {
    "P": ["p"],
    "B": ["b"],
    "T": ["t"],
    "D": ["d"],
    "K": ["k"],
    "G": ["got", "g", "ɡ"],
    "S": ["s"],
    "Z": ["z"],
    "SH": ["ʃ"],
    "ZH": ["ʒ"],
    "CH": ["tʃ", "ʧ"],
    "JH": ["dʒ", "ʤ"],
    "TH": ["θ"],
    "DH": ["ð"],
    "F": ["f"],
    "V": ["v"],
    "HH": ["h"],
    "M": ["m"],
    "N": ["n"],
    "NG": ["ŋ"],
    "L": ["l"],
    "R": ["ɹ", "r"],
    "Y": ["j"],
    "W": ["w"],
    "IH": ["ɪ"],
    "IY": ["i"],
    "EH": ["ɛ", "e"],
    "AE": ["a", "æ"],
    "AH": ["ʌ", "ə"],
    "AX": ["ə", "ʌ"],
    "AA": ["a", "ɒ"],
    "AO": ["ɔ", "ɒ"],
    "UH": ["ʊ"],
    "UW": ["u"],
    "ER": ["ɜ"],
    "OW": ["oʊ", "əʊ"],
    "EY": ["eɪ"],
    "AY": ["aɪ"],
    "AW": ["aʊ"],
    "OY": ["ɔɪ"],
}


def strip_cmudict_stress(phone: str) -> str:
    return re.sub(r"\d+$", "", str(phone).strip().upper())


def resolve_cmudict_phones(word: str, pronunciation_index: int = 0) -> list[str]:
    normalized_word = str(word or "").strip().lower()
    if not normalized_word:
        raise ValueError("Cannot resolve phones for an empty word.")

    try:
        import cmudict
    except ImportError as exc:
        raise ImportError("cmudict is required. Install with: pip install cmudict") from exc

    pronunciations = cmudict.dict().get(normalized_word)
    if not pronunciations:
        raise ValueError(f"Word {normalized_word!r} was not found in cmudict.")

    index = max(0, min(int(pronunciation_index), len(pronunciations) - 1))
    phones = [strip_cmudict_stress(phone) for phone in pronunciations[index]]
    phones = [phone for phone in phones if phone]
    if not phones:
        raise ValueError(f"cmudict returned an empty pronunciation for word {normalized_word!r}.")
    return phones


def resolve_preview_phones(word: str) -> list[str]:
    normalized = str(word).strip().lower()
    if normalized in PREVIEW_LEXICON:
        return list(PREVIEW_LEXICON[normalized])

    try:
        return resolve_cmudict_phones(normalized)
    except (ImportError, ValueError):
        raise


def clamp_unit(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, float(value)))


def normalized_time_distance_profile(comparison: dict[str, Any]) -> list[float]:
    values = comparison.get("per_time_l2")
    if not isinstance(values, list) or not values:
        raise ValueError("comparison JSON must contain a non-empty per_time_l2 list.")

    numeric = np.asarray([float(v) for v in values], dtype=np.float32)
    numeric = np.nan_to_num(numeric, nan=0.0, posinf=0.0, neginf=0.0)

    lo = float(np.min(numeric))
    hi = float(np.max(numeric))
    if hi <= lo + 1e-12:
        return [0.0 for _ in numeric]

    return [clamp_unit((float(v) - lo) / (hi - lo)) for v in numeric]


def phoneme_bins(values: list[float], target_count: int) -> list[float]:
    target_count = max(1, int(target_count))
    if not values:
        return [0.0] * target_count
    chunks = np.array_split(np.asarray([clamp_unit(v) for v in values], dtype=np.float32), target_count)
    return [
        clamp_unit(float(np.max(chunk))) if chunk.size else clamp_unit(values[min(index, len(values) - 1)])
        for index, chunk in enumerate(chunks)
    ]


def per_phoneme_eeg_targets(comparison: dict[str, Any], phone_count: int) -> list[float]:
    return phoneme_bins(normalized_time_distance_profile(comparison), phone_count)


def preview_targets_for_phone_count(signal: list[float], phone_count: int) -> list[float]:
    return phoneme_bins(signal, phone_count)


def parse_distance_signal(value: str) -> list[float]:
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("Preview distance signal cannot be empty.")
    return [float(part) for part in parts]


def granularity_values(granularity: int) -> list[float]:
    granularity = int(granularity)
    if granularity < 2:
        raise ValueError("--granularity must be at least 2.")
    return [i / float(granularity - 1) for i in range(granularity)]


def distance_label(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def distance_key(values: tuple[float, ...] | list[float]) -> str:
    return "_".join(distance_label(value) for value in values)


def output_distance_key(values: tuple[float, ...] | list[float], min_components: int = 2) -> str:
    padded = list(values)
    while len(padded) < int(min_components):
        padded.append(0.0)
    return distance_key(padded)


def effective_output_root(output_root: str | Path, granularity: int) -> Path:
    root = Path(output_root)
    suffix = f"granularity_{int(granularity)}"
    if root.name == suffix:
        return root
    return root / suffix


def phone_options_by_distance(
    phone: str,
    pairwise_distances: dict[str, Any],
) -> list[tuple[str, float, str]]:
    source = strip_cmudict_stress(phone)
    row = (pairwise_distances.get("distances") or {}).get(source) or {}
    options: dict[str, tuple[str, float, str]] = {source: (source, 0.0, f"canonical:{source}")}
    labels: dict[str, str] = {}
    for candidate, _declared_distance, label in variants_for_phone(source, 1.0):
        labels[strip_cmudict_stress(candidate)] = label
    for candidate, distance in row.items():
        candidate_key = strip_cmudict_stress(candidate)
        label = labels.get(candidate_key, f"cmvn:{source}->{candidate_key}")
        options[candidate_key] = (candidate_key, float(distance), label)
    return sorted(options.values(), key=lambda item: (item[1], item[0]))

def select_phones_for_distance_tuple(
    canonical: list[str],
    distances: tuple[float, ...] | list[float],
    pairwise_distances: dict[str, Any],
    max_phone_rank: int = DEFAULT_MAX_PHONE_RANK,
) -> tuple[list[str], list[float], list[str]]:
    """Select one generated phone per slot using normalized CMVN auditory distance."""
    phones: list[str] = []
    actual_distances: list[float] = []
    operations: list[str] = []

    for canonical_phone, target in zip(canonical, distances):
        options = phone_options_by_distance(canonical_phone, pairwise_distances)
        max_index = max(0, len(options) - 1)
        if int(max_phone_rank) > 0:
            max_index = min(max_index, int(max_phone_rank))
        index = round(clamp_unit(float(target)) * max_index)
        selected_phone, selected_distance, operation = options[index]
        phones.append(selected_phone)
        actual_distances.append(float(selected_distance))
        operations.append(operation)

    return phones, actual_distances, operations

def parse_words(value: str | None) -> tuple[str, ...]:
    if not value:
        return PREVIEW_WORDS
    words = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not words:
        raise ValueError("Word list cannot be empty.")
    return words


def parse_voices(value: str | None) -> tuple[str, ...]:
    raw = value or "man,woman"
    voices = tuple(normalize_voice_mode(part.strip()) for part in raw.split(",") if part.strip())
    if not voices:
        raise ValueError("Voice list cannot be empty.")
    return voices


def parse_int_values(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_s, end_s = token.split("-", 1)
            start = int(start_s.strip())
            end = int(end_s.strip())
            step = 1 if end >= start else -1
            out.extend(range(start, end + step, step))
        else:
            out.append(int(token))

    if not out:
        raise ValueError(f"No integer values parsed from {value!r}")
    return out


def phone_inventory_names(phone: str) -> list[str]:
    normalized = strip_cmudict_stress(phone)
    names = list(ARPABET_TO_AUS_IPA_CANDIDATES.get(normalized, []))
    names.extend([normalized, normalized.lower()])

    deduped: list[str] = []
    seen: set[str] = set()
    for name in names:
        if name and name not in seen:
            deduped.append(name)
            seen.add(name)
    return deduped


def resolve_phone_inventory_asset(phone: str, inventory_root: Path) -> Path | None:
    for basename in phone_inventory_names(phone):
        for suffix in (".wav", ".mp3"):
            path = inventory_root / f"{basename}{suffix}"
            if path.is_file():
                return path
    return None


def resolve_phone_inventory_assets(
    phones: list[str],
    inventory_root: Path,
) -> tuple[list[Path], list[str]]:
    assets: list[Path] = []
    missing: list[str] = []

    for phone in phones:
        path = resolve_phone_inventory_asset(phone, inventory_root)
        if path is None:
            missing.append(phone)
        else:
            assets.append(path)

    return assets, missing


def _read_wav_int16(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(f"Expected 16-bit PCM WAV, got sample width {sample_width}: {path}")

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)

    return sample_rate, audio.astype(np.float32) / 32768.0


def _write_wav_int16(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)

    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(int(sample_rate))
        wav_file.writeframes(pcm.tobytes())


def resample_audio_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    source_rate = int(source_rate)
    target_rate = int(target_rate)

    if source_rate == target_rate or len(audio) == 0:
        return audio.astype(np.float32)

    duration = len(audio) / float(source_rate)
    target_len = max(1, int(round(duration * target_rate)))

    source_x = np.linspace(0.0, 1.0, len(audio), endpoint=False)
    target_x = np.linspace(0.0, 1.0, target_len, endpoint=False)

    return np.interp(target_x, source_x, audio).astype(np.float32)


def decode_phone_asset_to_audio(path: Path, *, sample_rate: int = 24000) -> tuple[int, np.ndarray]:
    if path.suffix.lower() == ".wav":
        source_rate, audio = _read_wav_int16(path)
        return sample_rate, resample_audio_linear(audio, source_rate, sample_rate)

    try:
        import soundfile as sf

        audio, source_rate = sf.read(str(path), dtype="float32", always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        audio = np.nan_to_num(audio)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak

        return sample_rate, resample_audio_linear(audio, int(source_rate), sample_rate)

    except Exception:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("Could not decode phone asset with soundfile, and ffmpeg is not installed.")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)

    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", str(path), "-ar", str(sample_rate), "-ac", "1", str(tmp_path)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return _read_wav_int16(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def crossfade_concat(
    segments: list[np.ndarray],
    *,
    sample_rate: int,
    gap_ms: float,
    crossfade_ms: float,
) -> np.ndarray:
    if not segments:
        return np.zeros(0, dtype=np.float32)

    gap_samples = max(0, int(round((float(gap_ms) / 1000.0) * sample_rate)))
    crossfade_samples = max(0, int(round((float(crossfade_ms) / 1000.0) * sample_rate)))

    output = segments[0].astype(np.float32)

    for segment in segments[1:]:
        segment = segment.astype(np.float32)
        fade = min(crossfade_samples, len(output), len(segment))

        if fade > 0:
            ramp = np.linspace(0.0, 1.0, fade, endpoint=False, dtype=np.float32)
            blended = output[-fade:] * (1.0 - ramp) + segment[:fade] * ramp
            output = np.concatenate([output[:-fade], blended, segment[fade:]])
        else:
            spacer = np.zeros(gap_samples, dtype=np.float32) if gap_samples else np.zeros(0, dtype=np.float32)
            output = np.concatenate([output, spacer, segment])

    return output.astype(np.float32)


def stitch_phone_assets(
    *,
    source_paths: list[Path],
    output_wav: Path,
    overwrite: bool,
    gap_ms: float,
    crossfade_ms: float,
    sample_rate: int = 24000,
) -> dict[str, Any]:
    if output_wav.exists() and not overwrite:
        return {
            "status": "exists",
            "source_phone_assets": [str(path) for path in source_paths],
        }

    decoded_segments = [
        decode_phone_asset_to_audio(path, sample_rate=sample_rate)[1]
        for path in source_paths
    ]
    audio = crossfade_concat(
        decoded_segments,
        sample_rate=sample_rate,
        gap_ms=gap_ms,
        crossfade_ms=crossfade_ms,
    )

    _write_wav_int16(output_wav, sample_rate, audio)

    return {
        "status": "ok",
        "source_phone_assets": [str(path) for path in source_paths],
        "duration_sec": float(len(audio) / sample_rate) if sample_rate else 0.0,
    }



def cmvn_pairwise_cache_path(inventory_root: Path) -> Path:
    """Cache path for phone-audio pairwise CMVN/MFCC/DTW distances."""
    return inventory_root / "cmvn_pairwise_distances.json"


def available_inventory_phone_assets(inventory_root: Path) -> dict[str, Path]:
    """Return ARPABET phones that can be resolved to cleaned inventory audio."""
    phones = sorted(set(ARPABET_TO_AUS_IPA_CANDIDATES) | set(PHONE_FEATURES))
    assets: dict[str, Path] = {}

    for phone in phones:
        path = resolve_phone_inventory_asset(phone, inventory_root)
        if path is not None:
            assets[strip_cmudict_stress(phone)] = path

    return assets


def _cmvn_features_from_phone_asset(path: Path) -> np.ndarray:
    """Extract CMVN-normalized MFCC features for one isolated phone asset."""
    sample_rate, audio = decode_phone_asset_to_audio(path, sample_rate=24000)
    features = _cmvn(_extract_mfcc(audio, sample_rate))
    if CMVN_DTW_MAX_FRAMES > 0 and features.size:
        features = _resample_time_axis(features, CMVN_DTW_MAX_FRAMES)
    return features


def normalize_pairwise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Add normalized [0,1] auditory distances while preserving raw CMVN-DTW.
    """
    raw_distances = payload.get("raw_distances") or payload.get("distances") or {}

    raw_max = 0.0
    for row in raw_distances.values():
        for value in row.values():
            raw_max = max(raw_max, float(value))

    # 1.0 maps to the furthest raw CMVN-DTW phone-pair distance in this cache.
    effective_max = raw_max

    normalized: dict[str, dict[str, float]] = {}
    for source, row in raw_distances.items():
        source_key = strip_cmudict_stress(source)
        normalized[source_key] = {}
        for target, value in row.items():
            target_key = strip_cmudict_stress(target)
            if effective_max <= 0.0:
                normalized[source_key][target_key] = 0.0
            else:
                normalized[source_key][target_key] = clamp_unit(float(value) / effective_max)

    payload["raw_distances"] = raw_distances
    payload["distances"] = normalized
    payload["raw_distance_max"] = raw_max
    payload["distance_scale"] = "normalized_0_1"
    payload["normalization"] = {
        "method": "raw_max",
        "raw_distance_max": raw_max,
        "effective_max_distance": effective_max,
        "comment": f"1.0 maps to raw_distance_max={raw_max}",
    }
    return payload


def build_cmvn_pairwise_distances(inventory_root: Path) -> dict[str, Any]:
    """
    Build pairwise auditory distances from the cleaned phoneme asset library.

    Distance = CMVN-normalized MFCC + DTW. Raw distances are kept and normalized
    to [0,1] using normalize_pairwise_payload().
    """
    phone_assets = available_inventory_phone_assets(inventory_root)
    if not phone_assets:
        raise FileNotFoundError(f"No cleaned phone assets found in inventory root: {inventory_root}")

    phone_features: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}

    for phone, path in sorted(phone_assets.items()):
        try:
            features = _cmvn_features_from_phone_asset(path)
            if features.size == 0:
                skipped[phone] = "empty_features"
            else:
                phone_features[phone] = features
        except Exception as exc:
            skipped[phone] = str(exc)

    phones = sorted(phone_features)
    raw_distances: dict[str, dict[str, float]] = {phone: {} for phone in phones}

    for idx, source in enumerate(phones):
        for target in phones[idx:]:
            if source == target:
                distance = 0.0
            else:
                distance = float(_dtw_distance(phone_features[source], phone_features[target]))
            raw_distances[source][target] = distance
            raw_distances[target][source] = distance

    payload = {
        "schema_version": "1.0-cmvn-phone-pairwise",
        "inventory_root": str(inventory_root),
        "distance_metric": "cmvn_mfcc_dtw",
        "distance_scale": "normalized_0_1",
        "phones": phones,
        "phone_assets": {phone: str(phone_assets[phone]) for phone in phones},
        "raw_distances": raw_distances,
        "skipped": skipped,
    }
    return normalize_pairwise_payload(payload)


def load_or_create_cmvn_pairwise_distances(
    inventory_root: Path,
    *,
    rebuild: bool = False,
) -> dict[str, Any]:
    """
    Load the pairwise CMVN auditory-distance cache, creating it if missing.

    The raw pairwise distances are computed from comparisons among cleaned
    phoneme assets in phone_inventory_root. If the cache already exists, raw
    distances are reused and normalized by the raw maximum pairwise distance.
    """
    cache_path = cmvn_pairwise_cache_path(inventory_root)

    if cache_path.is_file() and not rebuild:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return normalize_pairwise_payload(payload)

    payload = build_cmvn_pairwise_distances(inventory_root)
    inventory_root.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return payload


def pairwise_cmvn_distance(
    pairwise: dict[str, Any],
    source_phone: str,
    generated_phone: str,
) -> float | None:
    """Return normalized [0,1] auditory distance for source -> generated."""
    source = strip_cmudict_stress(source_phone)
    generated = strip_cmudict_stress(generated_phone)
    value = ((pairwise.get("distances") or {}).get(source) or {}).get(generated)
    return None if value is None else float(value)


def pairwise_signature(pairwise: dict[str, Any]) -> dict[str, Any]:
    """Small metadata signature used to invalidate stale generated libraries."""
    return {
        "distance_metric": pairwise.get("distance_metric"),
        "distance_scale": pairwise.get("distance_scale"),
        "phones": list(pairwise.get("phones") or []),
        "normalization": dict(pairwise.get("normalization") or {}),
    }


def asset_audio_path(asset: dict[str, Any]) -> str | None:
    return asset.get("audio_path") or asset.get("wav_path") or asset.get("output_audio_path")


def asset_audio_exists(asset: dict[str, Any]) -> bool:
    audio = asset_audio_path(asset)
    return bool(audio) and Path(str(audio)).is_file()


def per_phoneme_library_json_path(output_root: str | Path, voice_mode: str, word: str) -> Path:
    word = word.strip().lower()
    return Path(output_root) / normalize_voice_mode(voice_mode) / word / f"{word}_per_phoneme_assets.json"


def make_per_phoneme_asset(
    *,
    word: str,
    voice_mode: str,
    canonical: list[str],
    generated: list[str],
    requested_distances: tuple[float, ...],
    actual_distances: list[float],
    operations: list[str],
    wav_path: Path,
    source_paths: list[Path],
    stitch_info: dict[str, Any],
) -> dict[str, Any]:
    key = output_distance_key(requested_distances)
    render_text = " ".join(generated)

    return {
        "schema_version": "1.0-per-phoneme-grid",
        "asset_id": f"{word}__{voice_mode}__{key}",
        "word": word,
        "target_word": word,
        "voice_mode": voice_mode,
        "generation_mode": "per_phoneme_grid",
        "audio_source": "phone_stitch",
        "candidate_text": render_text,
        "render_text": render_text,
        "phones": list(generated),
        "canonical_phones": list(canonical),
        "distance_tuple": [float(value) for value in requested_distances],
        "actual_phone_distances": list(actual_distances),
        "actual_phone_distance_metric": "cmvn_mfcc_dtw_normalized",
        "distance_key": key,
        "raw_distance_key": distance_key(requested_distances),
        "operations": list(operations),
        "audio_path": str(wav_path),
        "wav_path": str(wav_path),
        "output_audio_path": str(wav_path),
        "source_phone_assets": [str(path) for path in source_paths],
        "render_status": f"stitched_{stitch_info.get('status', 'unknown')}",
        "stitched_duration_sec": stitch_info.get("duration_sec"),
        "synthetic_asset": False,
    }


def pregenerate_word_assets(
    *,
    word: str,
    voice_mode: str,
    output_root: Path,
    distance_values: list[float],
    phone_inventory_root: Path,
    pairwise_distances: dict[str, Any],
    max_phone_rank: int,
    overwrite: bool,
    gap_ms: float,
    crossfade_ms: float,
) -> dict[str, Any]:
    word = word.strip().lower()
    voice_mode = normalize_voice_mode(voice_mode)
    canonical = resolve_preview_phones(word)

    word_dir = output_root / voice_mode / word
    word_dir.mkdir(parents=True, exist_ok=True)

    assets: list[dict[str, Any]] = []
    skipped_missing_phone = 0
    skipped_missing_phone_details: list[dict[str, Any]] = []
    stitched_count = 0
    existing_count = 0

    for requested_distances in itertools.product(distance_values, repeat=len(canonical)):
        generated, actual_distances, operations = select_phones_for_distance_tuple(
            canonical,
            requested_distances,
            pairwise_distances,
            max_phone_rank=max_phone_rank,
        )
        key = output_distance_key(requested_distances)
        wav_path = word_dir / f"{key}.wav"

        source_phone_assets, missing_phones = resolve_phone_inventory_assets(
            generated,
            phone_inventory_root,
        )
        if missing_phones:
            skipped_missing_phone += 1
            if len(skipped_missing_phone_details) < TOP_K_DEBUG:
                skipped_missing_phone_details.append(
                    {
                        "distance_key": key,
                        "phones": list(generated),
                        "missing_phones": list(missing_phones),
                    }
                )
            continue

        stitch_info = stitch_phone_assets(
            source_paths=source_phone_assets,
            output_wav=wav_path,
            overwrite=overwrite,
            gap_ms=gap_ms,
            crossfade_ms=crossfade_ms,
        )

        if stitch_info["status"] == "exists":
            existing_count += 1
        else:
            stitched_count += 1

        assets.append(
            make_per_phoneme_asset(
                word=word,
                voice_mode=voice_mode,
                canonical=canonical,
                generated=generated,
                requested_distances=requested_distances,
                actual_distances=actual_distances,
                operations=operations,
                wav_path=wav_path,
                source_paths=source_phone_assets,
                stitch_info=stitch_info,
            )
        )

    cmvn_signature = pairwise_signature(pairwise_distances)
    payload = {
        "schema_version": "1.0-per-phoneme-grid",
        "generation_mode": "per_phoneme_grid",
        "audio_source": "phone_stitch",
        "distance_metric": "cmvn_mfcc_dtw",
        "word": word,
        "voice_mode": voice_mode,
        "canonical_phones": canonical,
        "granularity": len(distance_values),
        "filename_distance_components": max(2, len(canonical)),
        "distance_values": list(distance_values),
        "max_phone_rank": int(max_phone_rank),
        "asset_count": len(assets),
        "phone_inventory_root": str(phone_inventory_root),
        "cmvn_pairwise_distances_path": str(cmvn_pairwise_cache_path(phone_inventory_root)),
        "cmvn_pairwise_signature": cmvn_signature,
        "skipped_missing_phone_count": skipped_missing_phone,
        "skipped_missing_phone_examples": skipped_missing_phone_details,
        "stitched_count": stitched_count,
        "existing_audio_count": existing_count,
        "assets": assets,
        "library": {
            "schema_version": "1.0-per-phoneme-grid",
            "word": word,
            "voice_mode": voice_mode,
            "canonical_phones": canonical,
            "assets": assets,
        },
    }

    json_path = word_dir / f"{word}_per_phoneme_assets.json"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return payload

def pregenerate_assets(args: argparse.Namespace) -> list[dict[str, Any]]:
    values = granularity_values(args.granularity)
    output_root = Path(args.output_root)
    phone_inventory_root = Path(args.phone_inventory_root)
    pairwise_distances = load_or_create_cmvn_pairwise_distances(
        phone_inventory_root,
        rebuild=bool(args.rebuild_cmvn_pairwise),
    )
    expected_pairwise_signature = pairwise_signature(pairwise_distances)
    expected_max_phone_rank = int(args.max_phone_rank)

    payloads: list[dict[str, Any]] = []

    for voice_mode in parse_voices(args.voices):
        for word in parse_words(args.words):
            library_path = per_phoneme_library_json_path(output_root, voice_mode, word)
            expected_canonical = resolve_preview_phones(word)
            force_overwrite = bool(args.overwrite or args.rebuild_cmvn_pairwise)
            try:
                assets, payload = load_per_phoneme_assets(library_path)
                force_overwrite = force_overwrite or library_canonical_phones(payload) != expected_canonical
                force_overwrite = force_overwrite or payload.get("cmvn_pairwise_signature") != expected_pairwise_signature
                force_overwrite = force_overwrite or int(payload.get("max_phone_rank") or 0) != expected_max_phone_rank
                for asset in assets:
                    phones = list(asset.get("phones") or [])
                    source_paths = [str(path) for path in asset.get("source_phone_assets") or []]
                    current_paths, missing = resolve_phone_inventory_assets(phones, phone_inventory_root)
                    force_overwrite = force_overwrite or bool(missing)
                    force_overwrite = force_overwrite or source_paths != [str(path) for path in current_paths]
            except Exception:
                pass

            payloads.append(
                pregenerate_word_assets(
                    word=word,
                    voice_mode=voice_mode,
                    output_root=output_root,
                    distance_values=values,
                    phone_inventory_root=phone_inventory_root,
                    pairwise_distances=pairwise_distances,
                    max_phone_rank=expected_max_phone_rank,
                    overwrite=force_overwrite,
                    gap_ms=float(args.phone_gap_ms),
                    crossfade_ms=float(args.phone_crossfade_ms),
                )
            )

    return payloads

def load_per_phoneme_assets(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assets = payload.get("assets")

    if not isinstance(assets, list) and isinstance(payload.get("library"), dict):
        assets = payload["library"].get("assets")

    if not isinstance(assets, list):
        raise ValueError(f"Per-phoneme library does not contain an assets list: {path}")

    return assets, payload


def library_canonical_phones(payload: dict[str, Any]) -> list[str]:
    canonical = payload.get("canonical_phones")
    if canonical is None and isinstance(payload.get("library"), dict):
        canonical = payload["library"].get("canonical_phones")
    return list(canonical or [])


def per_phoneme_library_needs_generation(
    path: Path,
    expected_canonical: list[str] | None = None,
    phone_inventory_root: Path | None = None,
    expected_pairwise_signature: dict[str, Any] | None = None,
    expected_max_phone_rank: int | None = None,
) -> bool:
    if not path.is_file():
        return True

    try:
        assets, payload = load_per_phoneme_assets(path)
    except Exception:
        return True

    if expected_pairwise_signature is not None:
        if payload.get("cmvn_pairwise_signature") != expected_pairwise_signature:
            return True
    if expected_max_phone_rank is not None and int(payload.get("max_phone_rank") or 0) != int(expected_max_phone_rank):
        return True

    if expected_canonical is not None:
        if library_canonical_phones(payload) != list(expected_canonical):
            return True
        if int(payload.get("filename_distance_components") or 0) < max(2, len(expected_canonical)):
            return True

        for asset in assets:
            distance_tuple = asset.get("distance_tuple")
            audio_path = asset_audio_path(asset)
            if isinstance(distance_tuple, list) and audio_path:
                if Path(str(audio_path)).stem != output_distance_key(distance_tuple):
                    return True

            if phone_inventory_root is not None:
                phones = list(asset.get("phones") or [])
                source_paths = [str(path) for path in asset.get("source_phone_assets") or []]
                current_paths, missing = resolve_phone_inventory_assets(phones, phone_inventory_root)
                if missing or source_paths != [str(path) for path in current_paths]:
                    return True

    return not assets or any(not asset_audio_exists(asset) for asset in assets)

def ensure_word_library(
    *,
    word: str,
    voice_mode: str,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    library_path = per_phoneme_library_json_path(args.output_root, voice_mode, word)
    expected_canonical = resolve_preview_phones(word)
    expected_max_phone_rank = int(args.max_phone_rank)
    phone_inventory_root = Path(args.phone_inventory_root)
    pairwise_distances = load_or_create_cmvn_pairwise_distances(
        phone_inventory_root,
        rebuild=bool(args.rebuild_cmvn_pairwise),
    )
    expected_pairwise_signature = pairwise_signature(pairwise_distances)
    force_overwrite = bool(args.overwrite or args.rebuild_cmvn_pairwise)

    if not per_phoneme_library_needs_generation(
        library_path,
        expected_canonical,
        phone_inventory_root,
        expected_pairwise_signature,
        expected_max_phone_rank,
    ):
        return None

    try:
        assets, payload = load_per_phoneme_assets(library_path)
        force_overwrite = force_overwrite or library_canonical_phones(payload) != expected_canonical
        force_overwrite = force_overwrite or payload.get("cmvn_pairwise_signature") != expected_pairwise_signature
        force_overwrite = force_overwrite or int(payload.get("max_phone_rank") or 0) != expected_max_phone_rank
        for asset in assets:
            phones = list(asset.get("phones") or [])
            source_paths = [str(path) for path in asset.get("source_phone_assets") or []]
            current_paths, missing = resolve_phone_inventory_assets(phones, phone_inventory_root)
            force_overwrite = force_overwrite or bool(missing)
            force_overwrite = force_overwrite or source_paths != [str(path) for path in current_paths]
    except Exception:
        pass

    print(f"Creating missing phone-stitch library: {library_path}")

    return pregenerate_word_assets(
        word=word,
        voice_mode=voice_mode,
        output_root=Path(args.output_root),
        distance_values=granularity_values(args.granularity),
        phone_inventory_root=phone_inventory_root,
        pairwise_distances=pairwise_distances,
        max_phone_rank=expected_max_phone_rank,
        overwrite=force_overwrite,
        gap_ms=float(args.phone_gap_ms),
        crossfade_ms=float(args.phone_crossfade_ms),
    )

def vector_mse(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    if not left:
        return 0.0
    return sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)) / len(left)


def per_phoneme_target_signal(
    *,
    comparison: dict[str, Any] | None,
    preview_signal: str | None,
    phone_count: int,
) -> tuple[list[float], str]:
    if preview_signal is not None:
        return (
            preview_targets_for_phone_count(parse_distance_signal(preview_signal), phone_count),
            "preview_signal",
        )

    if comparison is None:
        raise ValueError("comparison is required when --preview-signal is not provided.")

    return per_phoneme_eeg_targets(comparison, phone_count), "template_probe_comparison"


def select_per_phoneme_asset_from_library(
    *,
    word: str,
    requested_voice: str,
    output_root: str | Path,
    comparison: dict[str, Any] | None = None,
    preview_signal: str | None = None,
    top_k: int = TOP_K_DEBUG,
) -> dict[str, Any]:
    word = word.strip().lower()
    voice_mode = normalize_voice_mode(requested_voice)
    canonical = resolve_preview_phones(word)

    target_signal, target_source = per_phoneme_target_signal(
        comparison=comparison,
        preview_signal=preview_signal,
        phone_count=len(canonical),
    )

    library_path = per_phoneme_library_json_path(output_root, voice_mode, word)
    assets, _payload = load_per_phoneme_assets(library_path)

    ranked: list[dict[str, Any]] = []

    for asset in assets:
        if not asset_audio_exists(asset):
            continue

        distance_tuple = [float(v) for v in asset.get("distance_tuple") or []]
        actual_distances = [float(v) for v in asset.get("actual_phone_distances") or []]

        if len(distance_tuple) != len(target_signal):
            continue

        tuple_mse = vector_mse(distance_tuple, target_signal)
        actual_mse = vector_mse(actual_distances, target_signal)

        ranked.append(
            {
                "asset": asset,
                "score": -actual_mse,
                "score_details": {
                    "selection_metric": "actual_phone_distance_mse",
                    "distance_tuple_mse": tuple_mse,
                    "actual_phone_distance_mse": actual_mse,
                    "target_signal_source": target_source,
                },
            }
        )

    if not ranked:
        raise ValueError(
            f"No stitched per-phoneme assets matched word={word!r}, "
            f"voice={voice_mode!r}, target length={len(target_signal)}."
        )

    ranked.sort(
        key=lambda row: (
            row["score_details"]["actual_phone_distance_mse"],
            row["score_details"]["distance_tuple_mse"],
            str(row["asset"].get("asset_id") or ""),
        )
    )

    best = ranked[0]
    asset = best["asset"]

    return {
        "word": word,
        "voice_mode": voice_mode,
        "generation_mode": "per_phoneme_grid",
        "audio_source": "phone_stitch",
        "library_json": str(library_path),
        "audio_path": asset_audio_path(asset),
        "asset_id": asset.get("asset_id"),
        "target_signal": target_signal,
        "target_signal_source": target_source,
        "selected_phoneme_grades": asset.get("distance_tuple"),
        "actual_phone_distances": asset.get("actual_phone_distances"),
        "generated_phones": asset.get("phones"),
        "canonical_phones": asset.get("canonical_phones"),
        "score": best["score"],
        "score_details": best["score_details"],
        "asset_metadata": asset,
        "top_candidates": [
            {
                "asset_id": row["asset"].get("asset_id"),
                "candidate_text": row["asset"].get("candidate_text"),
                "distance_tuple": row["asset"].get("distance_tuple"),
                "actual_phone_distances": row["asset"].get("actual_phone_distances"),
                "audio_path": asset_audio_path(row["asset"]),
                "score": row["score"],
                "score_details": row["score_details"],
            }
            for row in ranked[: max(1, int(top_k))]
        ],
        "times": list(comparison.get("times", [])) if comparison else [],
        "per_time_l2": list(comparison.get("per_time_l2", [])) if comparison else [],
    }


def preview_rows(
    *,
    words: tuple[str, ...] = PREVIEW_WORDS,
    distance_signal: list[float] | None = None,
    phone_inventory_root: Path = Path(DEFAULT_PHONE_INVENTORY_ROOT),
    rebuild_cmvn_pairwise: bool = False,
    max_phone_rank: int = DEFAULT_MAX_PHONE_RANK,
) -> list[dict[str, Any]]:
    signal = distance_signal if distance_signal is not None else [0.1, 0.3, 0.6, 0.9]
    pairwise_distances = load_or_create_cmvn_pairwise_distances(
        phone_inventory_root,
        rebuild=rebuild_cmvn_pairwise,
    )
    rows: list[dict[str, Any]] = []

    for word in words:
        canonical = resolve_preview_phones(word)
        targets = preview_targets_for_phone_count(signal, len(canonical))
        generated, distances, operations = select_phones_for_distance_tuple(
            canonical,
            targets,
            pairwise_distances,
            max_phone_rank=max_phone_rank,
        )

        rows.append(
            {
                "word": word,
                "canonical_phones": canonical,
                "preview_targets": targets,
                "generated_phones": generated,
                "operations": operations,
                "distance_metric": "cmvn_mfcc_dtw",
                "max_phone_rank": int(max_phone_rank),
                "phoneme_map": [
                    {
                        "source": source,
                        "generated": gen,
                        "target": targets[idx],
                        "cmvn_distance": distances[idx],
                    }
                    for idx, (source, gen) in enumerate(zip(canonical, generated))
                ],
            }
        )

    return rows

def print_preview(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        canonical = " ".join(row["canonical_phones"])
        generated = " ".join(row["generated_phones"])
        targets = ",".join(f"{value:.3f}" for value in row["preview_targets"])
        print(f"{row['word']}: {canonical} -> {generated}  targets={targets}")


def build_default_comparison(args: argparse.Namespace) -> dict[str, Any]:
    from Template_l2_compare_v2 import (
        build_template_from_dataset,
        compare_signal_to_prebuilt_template,
        get_one_trial_from_dataset,
    )
    from select_blabber_asset import adapt_comparison_result

    template_object = build_template_from_dataset(
        template_days=parse_int_values(args.template_days),
        template_sessions=parse_int_values(args.template_sessions),
        nperseg=args.nperseg,
        noverlap=args.noverlap,
        eps=args.eps,
        fmax=args.fmax,
    )

    trial, label_name, dataset = get_one_trial_from_dataset(
        day=args.probe_day,
        sess=args.probe_session,
        label=args.label,
        trial_index_within_label=args.trial_index,
    )

    print("Loaded trial label:", label_name)

    result = compare_signal_to_prebuilt_template(
        template_object=template_object,
        input_signal=trial,
        input_label=args.label,
        output_mode="both",
        sr=getattr(dataset, "final_sampling_rate", args.sr),
    )

    return adapt_comparison_result(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phone-stitch per-phoneme grid + EEG/template-distance selection."
    )

    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--pregenerate-assets", action="store_true")
    parser.add_argument("--eeg", action="store_true")
    parser.add_argument("--preview-signal", default=None)

    parser.add_argument("--granularity", type=int, default=4)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--words", default=",".join(PREVIEW_WORDS))
    parser.add_argument("--word", default=None)

    parser.add_argument("--phone-inventory-root", default=DEFAULT_PHONE_INVENTORY_ROOT)
    parser.add_argument("--phone-gap-ms", type=float, default=20.0)
    parser.add_argument("--phone-crossfade-ms", type=float, default=5.0)
    parser.add_argument(
        "--max-phone-rank",
        type=int,
        default=DEFAULT_MAX_PHONE_RANK,
        help="Highest sorted pairwise phone rank reachable at target 1.0; 0 means no cap.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--rebuild-cmvn-pairwise",
        action="store_true",
        help="Rebuild <phone-inventory-root>/cmvn_pairwise_distances.json.",
    )


    parser.add_argument(
        "--voices", default="man", choices=['man'],
        help="man support only for now, woman will be later if female AU "\
        "phonemes are found"
    )
    parser.add_argument(
        "--requested-voice",
        default=DEFAULT_REQUESTED_VOICE,
        choices=["male", "man"],
        help="man support only for now, woman will be later "\
        "if female AU phonemes are found. female,woman",
    )
    parser.add_argument("--json-out", default=None)

    parser.add_argument("--template-days", default="1")
    parser.add_argument("--template-sessions", default="1-2")
    parser.add_argument("--probe-day", type=int, default=1)
    parser.add_argument("--probe-session", type=int, default=8)
    parser.add_argument("--label", type=int, default=4)
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--sr", type=int, default=250)
    parser.add_argument("--nperseg", type=int, default=128)
    parser.add_argument("--noverlap", type=int, default=96)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--fmax", type=float, default=50)

    # Backward-compatible no-op flags.
    parser.add_argument("--render-audio", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--render-audio-source", default="phone-stitch", help=argparse.SUPPRESS)
    parser.add_argument("--metadata-only", action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()

    args.output_root = str(effective_output_root(args.output_root, args.granularity))

    if args.preview:
        rows = preview_rows(
            distance_signal=parse_distance_signal(args.preview_signal or "0.1,0.3,0.6,0.9"),
            phone_inventory_root=Path(args.phone_inventory_root),
            rebuild_cmvn_pairwise=bool(args.rebuild_cmvn_pairwise),
            max_phone_rank=int(args.max_phone_rank),
        )

        payload = json.dumps(rows, indent=2, ensure_ascii=False, default=str)
        if args.json_out:
            Path(args.json_out).write_text(payload, encoding="utf-8")
        else:
            print_preview(rows)
        return

    if args.pregenerate_assets:
        payloads = pregenerate_assets(args)
        summary = {
            "output_root": args.output_root,
            "granularity": args.granularity,
            "voice_count": len(parse_voices(args.voices)),
            "word_count": len(parse_words(args.words)),
            "payload_count": len(payloads),
            "asset_count": sum(int(payload["asset_count"]) for payload in payloads),
            "audio_source": "phone_stitch",
            "cmvn_pairwise_distances_path": str(cmvn_pairwise_cache_path(Path(args.phone_inventory_root))),
            "max_phone_rank": int(args.max_phone_rank),
            "skipped_missing_phone_count": sum(
                int(payload.get("skipped_missing_phone_count") or 0)
                for payload in payloads
            ),
            "stitched_count": sum(int(payload.get("stitched_count") or 0) for payload in payloads),
            "existing_audio_count": sum(
                int(payload.get("existing_audio_count") or 0)
                for payload in payloads
            ),
        }

        payload = json.dumps(summary, indent=2, ensure_ascii=False, default=str)
        if args.json_out:
            Path(args.json_out).write_text(payload, encoding="utf-8")
        else:
            print(payload)
        return

    comparison: dict[str, Any] | None = None
    preview_signal = args.preview_signal or "0.1,0.3,0.6,0.9"
    word = str(args.word or "").strip().lower()

    if args.eeg:
        comparison = build_default_comparison(args)
        word = str(args.word or comparison.get("label_name") or "").strip().lower()
        if not word:
            parser.error("Could not infer word from default comparison. Pass --word.")
        preview_signal = None
    elif not word:
        parser.error("--word is required unless --eeg is used.")

    # optional: only if not man or woman
    voice_mode = normalize_voice_mode(args.requested_voice)

    # generate the assets if it doesn't exist
    ensure_word_library(
        word=word,
        voice_mode=voice_mode,
        args=args,
    )

    # key part: takes the word, voice, where to search the assets and the
    # distance signal for binning before returning the result dict. The key you
    # want is 'audio_path'. The rest are for diagnostics
    result = select_per_phoneme_asset_from_library(
        word=word,
        requested_voice=voice_mode,
        output_root=args.output_root,
        comparison=comparison,
        preview_signal=preview_signal,
        top_k=TOP_K_DEBUG,
    )

    result["output_root"] = args.output_root
    result["granularity"] = args.granularity
    result["cmvn_pairwise_distances_path"] = str(cmvn_pairwise_cache_path(Path(args.phone_inventory_root)))
    result["max_phone_rank"] = int(args.max_phone_rank)

    payload = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    if args.json_out:
        Path(args.json_out).write_text(payload, encoding="utf-8")
    else:
        print(payload)
        print("recommended asset: ", result['audio_path'])


if __name__ == "__main__":
    main()
