#!/usr/bin/env python3
"""
Chunk manually spoken blabber-batch WAV files into the same asset layout used by
run_full_lexicon_blabber.sh / deep_phone_candidate_stack_blabber_triphone.py.

Input convention:
  {word}_01.wav, {word}_02.wav, ...
Each file contains up to N spoken candidate words separated by silence. Files and
chunks are consumed in ascending batch/chunk order for each source word.

Default scored output convention:
  {out_root}/{word}/level_{level}/{index:04d}_L{level}_{candidate}_{suffix}.wav

Example output:
  yes/level_1.00/0058_L1.00_yehz_en-AU-Ray.wav

The script reads scored.json for each word, orders the scored candidates by the
same distance/order metadata, assigns each detected chunk to the next scored
candidate, and writes it under the candidate's level directory. With --bins 8,
levels are:
  0.00, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.00

Typical use:
  python3 chunk_spoken_blabber_assets_v2.py /path/to/spoken_wavs \
    --out-root /path/to/lexicon_audio \
    --scored-root /path/to/run_full_lexicon_output \
    --suffix en-AU-Ray \
    --bins 8 \
    --threshold-db -40
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import os
import sys
import wave
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from glob import glob
import requests
import shutil
import io

import numpy as np
import pandas as pd

INPUT_RE = re.compile(r"^(?P<word>[A-Za-z][A-Za-z0-9-]*)_(?P<batch>\d+)\.wav$", re.IGNORECASE)

CANDIDATE_TEXT_KEYS = (
    "grapheme", "candidate_text", "candidate", "surface", "text", "label", "name",
    "variant", "utterance", "blabber", "blabber_word", "render", "pronunciation",
    "token", "output_word", "word",
)
LEVEL_KEYS = ("level", "level_value", "target_level", "level_bin", "bin", "lev")
INDEX_KEYS = ("index", "idx", "rank", "candidate_index", "level_index", "i")
DISTANCE_KEYS = (
    "distance", "dist", "distortion", "score", "total_distance", "weighted_distance",
    "normalized_distance", "norm_distance", "norm_dist", "d",
)
PHONE_DISTANCE_KEYS = ("phone_distance", "phoneme_distance", "mean_phone_distance")
LIST_KEYS = ("candidates", "scored", "results", "items", "entries", "variants", "rows", "data")

PARENT_DIR = "/data/raqchia/audio-assets/"
ORIGINAL_ASSETS = f"{PARENT_DIR}/speech-assets-triphone/man/triphone/"
# RAY_ASSET_DIR = Path(f"{PARENT_DIR}/RayAssets/")
RAY_ASSET_DIR = Path(f"{PARENT_DIR}/RayAssets/")
ORIGINAL_VOICE = "piper"
TRANSCRIPT_DIR = str(RAY_ASSET_DIR) + "/text-only/text/triphone/" 
TRANSCRIPT_MAN_ASSET_DIR = RAY_ASSET_DIR / Path("man")
TRANSCRIPT_WOMAN_ASSET_DIR = RAY_ASSET_DIR / Path("woman")

ELEVENLABS_MODEL_ID = "eleven_multilingual_sts_v2"
ELEVENLABS_VOICE_STRING = "Charlotte - Meditation and Relaxation Australian"
ELEVENLABS_OUTPUT_FMT = "mp3_44100_128"
_API_KEY = "sk_4fd5d4c0da1af329be0bc4b93901b6cd6ee4d720a2310133"

@dataclass
class Segment:
    start_s: float
    end_s: float
    start_frame: int
    end_frame: int
    peak_dbfs: float
    rms_dbfs: float


@dataclass
class InputWav:
    word: str
    batch: int
    path: Path


@dataclass
class Candidate:
    source_word: str
    candidate: str
    level: str
    distance: Optional[float]
    json_order: int
    original_index: Optional[int]
    output_index: Optional[int] = None
    output_level: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None

@dataclass
class SignalConfig:
    max_chunks_per_file:int=10
    threshold_db:int=-35
    noise_percentile:float=20.0
    above_noise_db:float=18.0
    min_auto_db:float=-45.0
    max_auto_db:float=-35
    frame_ms:float=25.0
    hop_ms:float=10.0
    merge_gap_ms:float=180.0
    min_speech_ms:float=250.0
    pad_ms:float=120.0
    peak_normalize_dbfs:float=None

def dbfs(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(x, eps))


def sanitize_token(token: str) -> str:
    token = str(token).strip()
    token = re.sub(r"\s+", "-", token)
    token = re.sub(r"[^A-Za-z0-9_+.-]", "", token)
    return token or "candidate"


def read_wav_pcm(path: Path) -> Tuple[int, int, int, np.ndarray]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        sr = wf.getframerate()
        sampwidth = wf.getsampwidth()
        nframes = wf.getnframes()
        raw = wf.readframes(nframes)

    if sampwidth == 1:
        data = np.frombuffer(raw, dtype=np.uint8).astype(np.int16) - 128
    elif sampwidth == 2:
        data = np.frombuffer(raw, dtype=np.int16)
    elif sampwidth == 4:
        data = np.frombuffer(raw, dtype=np.int32)
    else:
        raise ValueError(f"Unsupported WAV sample width {sampwidth} bytes in {path}")

    if data.size % channels != 0:
        raise ValueError(f"Corrupt WAV: sample count is not divisible by channels in {path}")
    return sr, channels, sampwidth, data.reshape(-1, channels)


def write_wav_pcm(path: Path, sr: int, channels: int, sampwidth: int, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if sampwidth == 1:
        out = np.clip(data + 128, 0, 255).astype(np.uint8)
    elif sampwidth == 2:
        out = np.clip(data, -32768, 32767).astype(np.int16)
    elif sampwidth == 4:
        out = np.clip(data, -2147483648, 2147483647).astype(np.int32)
    else:
        raise ValueError(f"Unsupported sample width {sampwidth}")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(out.tobytes())

def wav_bytes_from_pcm(
    sr: int, channels: int, sampwidth: int, data: np.ndarray
) -> bytes:
    buf = io.BytesIO()

    if sampwidth == 1:
        out = np.clip(data + 128, 0, 255).astype(np.uint8)
    elif sampwidth == 2:
        out = np.clip(data, -32768, 32767).astype(np.int16)
    elif sampwidth == 4:
        out = np.clip(data, -2147483648, 2147483647).astype(np.int32)
    else:
        raise ValueError(f"Unsupported sample width {sampwidth}")

    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sr)
        wf.writeframes(out.tobytes())

    return buf.getvalue()


def resolve_elevenlabs_voice_id(
    voice_search: str,
    api_key:str=_API_KEY,
) -> str:
    resp = requests.get(
        "https://api.elevenlabs.io/v2/voices",
        headers={"xi-api-key": api_key},
        params={"search": voice_search, "page_size": 100},
        timeout=30,
    )
    resp.raise_for_status()

    voices = resp.json().get("voices", [])
    if not voices:
        raise SystemExit(f"No ElevenLabs voice found for search: {voice_search!r}")

    exact = [
        v for v in voices
        if v.get("name", "").strip().lower() == voice_search.strip().lower()
    ]
    chosen = (exact or voices)[0]
    print(f"Using ElevenLabs voice: {chosen.get('name')} ({chosen.get('voice_id')})")
    return chosen["voice_id"]


def elevenlabs_voice_change(
    chunk: np.ndarray,
    sr: int,
    channels: int,
    sampwidth: int,
    api_key: str,
    voice_id: str,
    model_id: str,
    output_format: str,
    remove_background_noise: bool,
) -> bytes:

    audio_bytes = wav_bytes_from_pcm(sr, channels, sampwidth, chunk)

    resp = requests.post(
        f"https://api.elevenlabs.io/v1/speech-to-speech/{voice_id}",
        headers={"xi-api-key": api_key},
        params={"output_format": output_format},
        data={
            "model_id": model_id,
            "remove_background_noise": str(remove_background_noise).lower(),
        },
        files={"audio": ("chunk.wav", audio_bytes, "audio/wav")},
        timeout=(10, 180),
    )
    if not resp.ok:
        raise RuntimeError(f"ElevenLabs voice changer failed {resp.status_code}: {resp.text[:1000]}")
    return resp.content


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def to_float_mono(data: np.ndarray, sampwidth: int) -> np.ndarray:
    if sampwidth == 1:
        denom = 128.0
    elif sampwidth == 2:
        denom = 32768.0
    elif sampwidth == 4:
        denom = 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {sampwidth}")
    x = data.astype(np.float32) / denom
    return x.mean(axis=1) if x.ndim == 2 else x


def frame_rms(mono: np.ndarray, sr: int, frame_ms: float, hop_ms: float) -> Tuple[np.ndarray, np.ndarray]:
    frame = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    if len(mono) < frame:
        mono = np.pad(mono, (0, frame - len(mono)))
    starts = np.arange(0, len(mono) - frame + 1, hop, dtype=np.int64)
    rms = np.empty(len(starts), dtype=np.float32)
    for i, s in enumerate(starts):
        chunk = mono[s:s + frame]
        rms[i] = math.sqrt(float(np.mean(chunk * chunk)))
    centers_s = (starts + frame / 2.0) / sr
    return rms, centers_s


def raw_segments_from_mask(mask: np.ndarray, times_s: np.ndarray, duration_s: float, frame_ms: float) -> List[Tuple[float, float]]:
    segs: List[Tuple[float, float]] = []
    in_seg = False
    start = 0.0
    half_frame_s = frame_ms / 2000.0
    for i, active in enumerate(mask):
        if active and not in_seg:
            in_seg = True
            start = max(0.0, float(times_s[i]) - half_frame_s)
        if in_seg and ((not active) or i == len(mask) - 1):
            end = min(duration_s, float(times_s[i]) + half_frame_s)
            segs.append((start, end))
            in_seg = False
    return segs


def merge_and_filter_segments(segs: Sequence[Tuple[float, float]], merge_gap_s: float, min_speech_s: float) -> List[Tuple[float, float]]:
    merged: List[List[float]] = []
    for s, e in segs:
        if e <= s:
            continue
        if not merged or s - merged[-1][1] > merge_gap_s:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged if e - s >= min_speech_s]


def choose_threshold_db(
    rms_db: np.ndarray,
    threshold_db: Optional[float],
    noise_percentile: float,
    above_noise_db: float,
    min_auto_db: float,
    max_auto_db: float,
) -> float:
    if threshold_db is not None:
        return float(threshold_db)
    noise = float(np.percentile(rms_db, noise_percentile))
    return min(max(noise + above_noise_db, min_auto_db), max_auto_db)


def detect_segments(
    data: np.ndarray,
    sr: int,
    sampwidth: int,
    frame_ms: float,
    hop_ms: float,
    threshold_db: Optional[float],
    noise_percentile: float,
    above_noise_db: float,
    min_auto_db: float,
    max_auto_db: float,
    merge_gap_ms: float,
    min_speech_ms: float,
    pad_ms: float,
    max_chunks: int,
) -> Tuple[List[Segment], float, float]:
    mono = to_float_mono(data, sampwidth)
    duration_s = len(mono) / sr
    rms, times_s = frame_rms(mono, sr, frame_ms, hop_ms)
    rms_db = dbfs(rms)
    threshold = choose_threshold_db(rms_db, threshold_db, noise_percentile, above_noise_db, min_auto_db, max_auto_db)

    mask = rms_db > threshold
    raw = raw_segments_from_mask(mask, times_s, duration_s, frame_ms)
    merged = merge_and_filter_segments(raw, merge_gap_ms / 1000.0, min_speech_ms / 1000.0)

    # If the threshold is too permissive, words can bridge together. Try stricter
    # thresholds until we get closer to max_chunks without exceeding it.
    if 0 < len(merged) < max_chunks and threshold < -32.0:
        best = (len(merged), threshold, merged)
        for th in np.arange(threshold + 1.0, -24.0, 1.0):
            m = rms_db > th
            cand = merge_and_filter_segments(
                raw_segments_from_mask(m, times_s, duration_s, frame_ms),
                merge_gap_ms / 1000.0,
                min_speech_ms / 1000.0,
            )
            if len(cand) <= max_chunks and len(cand) > best[0]:
                best = (len(cand), float(th), cand)
            if len(cand) == max_chunks:
                break
        _, threshold, merged = best

    if len(merged) > max_chunks:
        raise ValueError(
            f"Detected {len(merged)} chunks, which exceeds --max-chunks-per-file={max_chunks}. "
            f"Raise --threshold-db, raise --merge-gap-ms, or inspect the recording."
        )

    pad_s = pad_ms / 1000.0
    out: List[Segment] = []
    for s, e in merged:
        ps = max(0.0, s - pad_s)
        pe = min(duration_s, e + pad_s)
        sf = max(0, int(round(ps * sr)))
        ef = min(len(data), int(round(pe * sr)))
        chunk_mono = mono[sf:ef]
        peak = float(np.max(np.abs(chunk_mono))) if len(chunk_mono) else 0.0
        rms_val = float(math.sqrt(np.mean(chunk_mono * chunk_mono))) if len(chunk_mono) else 0.0
        out.append(Segment(ps, pe, sf, ef, float(dbfs(np.array([peak]))[0]), float(dbfs(np.array([rms_val]))[0])))
    return out, threshold, float(np.percentile(rms_db, noise_percentile))


def peak_normalize(data: np.ndarray, sampwidth: int, target_dbfs: Optional[float]) -> np.ndarray:
    if target_dbfs is None:
        return data
    if sampwidth == 1:
        max_int = 127.0
    elif sampwidth == 2:
        max_int = 32767.0
    elif sampwidth == 4:
        max_int = 2147483647.0
    else:
        return data
    peak = float(np.max(np.abs(data.astype(np.float64))))
    if peak <= 0:
        return data
    target = max_int * (10.0 ** (target_dbfs / 20.0))
    gain = target / peak
    return np.round(data.astype(np.float64) * gain).astype(data.dtype)


def iter_input_wavs(input_paths: Sequence[Path]) -> List[InputWav]:
    found: List[Path] = []

    if not isinstance(input_paths, list):
        input_paths = [input_paths]

    for p in input_paths:
        if p.is_dir():
            found.extend(sorted(p.glob("*.wav")))
        else:
            found.append(p)
    parsed: List[InputWav] = []
    for p in found:
        m = INPUT_RE.match(p.name)
        if not m:
            print(f"Skipping {p}: expected filename like word_01.wav", file=sys.stderr)
            continue
        parsed.append(InputWav(m.group("word").lower(), int(m.group("batch")), p))
    parsed.sort(key=lambda t: (t.word, t.batch, str(t.path)))
    return parsed


def level_labels(bins: int) -> List[str]:
    if bins < 2:
        raise ValueError("--bins must be at least 2")
    return [f"{i / (bins - 1):.2f}" for i in range(bins)]


def normalize_level(value: Any, bins: int) -> str:
    labels = level_labels(bins)
    if value is None:
        raise ValueError("Cannot normalize missing level")
    if isinstance(value, str):
        m = re.search(r"(?:level_|L)?([0-9]+(?:\.[0-9]+)?)", value)
        if not m:
            raise ValueError(f"Cannot parse level from {value!r}")
        x = float(m.group(1))
    else:
        x = float(value)
    return min(labels, key=lambda s: abs(float(s) - x))


def distance_to_level(distance: float, bins: int, mode: str) -> str:
    labels = level_labels(bins)
    d = min(max(float(distance), 0.0), 1.0)
    step = 1.0 / (bins - 1)
    if mode == "floor":
        idx = int(math.floor(d / step + 1e-12))
    elif mode == "ceil":
        idx = int(math.ceil(d / step - 1e-12))
    else:
        idx = int(round(d / step))
    idx = min(max(idx, 0), bins - 1)
    return labels[idx]


def first_present(row: Dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            m = re.search(r"-?\d+(?:\.\d+)?", value)
            if m:
                return float(m.group(0))
        return None


def coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        if isinstance(value, str):
            m = re.search(r"\d+", value)
            if m:
                return int(m.group(0))
        return None


def parse_asset_path_fields(row: Dict[str, Any], bins: int) -> Tuple[Optional[str], Optional[int]]:
    value = first_present(row, ("wav_path", "audio_path", "output_audio_path", "file", "filename", "path", "output"))
    if value is None:
        return None, None
    path = Path(str(value))
    level: Optional[str] = None
    for part in path.parts:
        if part.startswith("level_"):
            try:
                level = normalize_level(part, bins)
            except ValueError:
                level = None
            break
    index: Optional[int] = None
    m = re.match(r"(\d+)_", path.name)
    if m:
        index = int(m.group(1))
    return level, index


def extract_list_from_json(obj: Any) -> List[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        for key in LIST_KEYS:
            value = obj.get(key)
            if isinstance(value, list):
                return value
        # Common shape: {"yes": [...], "no": [...]} or nested dicts.
        list_values = [v for v in obj.values() if isinstance(v, list)]
        if len(list_values) == 1:
            return list_values[0]
        if list_values:
            out: List[Any] = []
            for v in list_values:
                out.extend(v)
            return out
    raise ValueError("scored.json does not contain a recognizable candidate list")


def row_from_any(item: Any) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item)
    if isinstance(item, (list, tuple)):
        row: Dict[str, Any] = {"_raw_list": list(item)}
        # Heuristic support for simple list rows such as [candidate, distance]
        # or [index, level, candidate, distance].
        if len(item) == 2:
            if isinstance(item[0], str):
                row["candidate"] = item[0]
                row["distance"] = item[1]
            else:
                row["distance"] = item[0]
                row["candidate"] = item[1]
        elif len(item) >= 3:
            row["index"] = item[0]
            row["level"] = item[1]
            row["candidate"] = item[2]
            if len(item) >= 4:
                row["distance"] = item[3]
        return row
    return {"candidate": str(item)}


def find_scored_json(word: str, inputs: Sequence[InputWav], out_root: Path, scored_root: Optional[Path], scored_json_name: str) -> Optional[Path]:
    dirs: List[Path] = []
    if scored_root is not None:
        dirs.extend([scored_root / word, scored_root])
    dirs.extend([out_root / word, out_root])
    for inp in inputs:
        if inp.word == word:
            dirs.extend([inp.path.parent / word, inp.path.parent])
    checked: List[Path] = []
    for d in dirs:
        checked.extend([
            d / scored_json_name,
            d / word / scored_json_name,
            d / f"{word}_{scored_json_name}",
        ])
    seen = set()
    for p in checked:
        if p in seen:
            continue
        seen.add(p)
        if p.exists() and p.is_file():
            return p
    # Last resort: one-level recursive search under scored_root/out_root.
    for root in [scored_root, out_root]:
        if root and root.exists():
            matches = list(root.glob(f"**/{scored_json_name}"))
            word_matches = [m for m in matches if m.parent.name.lower() == word or word in str(m.parent).lower().split("/")]
            if word_matches:
                return sorted(word_matches, key=lambda p: len(str(p)))[0]
    return None


def load_candidates(
    word: str,
    scored_path: Path,
    bins: int,
    order: str,
    bin_assignment: str,
) -> List[Candidate]:
    with scored_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    items = extract_list_from_json(obj)
    rows = [row_from_any(item) for item in items]

    # Normalize distances if they look raw rather than 0..1.
    raw_distances = [coerce_float(first_present(r, DISTANCE_KEYS)) for r in rows]
    known_distances = [d for d in raw_distances if d is not None]
    use_minmax = bool(known_distances) and (min(known_distances) < 0.0 or max(known_distances) > 1.0)
    d_min = min(known_distances) if known_distances else 0.0
    d_max = max(known_distances) if known_distances else 1.0

    candidates: List[Candidate] = []
    for j, row in enumerate(rows):
        text = first_present(row, CANDIDATE_TEXT_KEYS)
        if text is None:
            # Try filename-like fields before failing.
            file_value = first_present(row, ("file", "filename", "path", "output"))
            if file_value is not None:
                stem = Path(str(file_value)).stem
                m = re.search(r"_L[0-9.]+_([^_]+)_", stem)
                text = m.group(1) if m else stem
        if text is None:
            raise ValueError(f"Candidate row {j} in {scored_path} does not expose a candidate text field")
        text_s = sanitize_token(str(text))

        dist = coerce_float(first_present(row, DISTANCE_KEYS))
        norm_dist = dist
        if dist is not None and use_minmax and d_max > d_min:
            norm_dist = (dist - d_min) / (d_max - d_min)

        lvl_value = first_present(row, LEVEL_KEYS)
        if lvl_value is not None:
            lvl = normalize_level(lvl_value, bins)
        elif norm_dist is not None:
            lvl = distance_to_level(norm_dist, bins, bin_assignment)
        else:
            raise ValueError(f"Candidate row {j} in {scored_path} has neither level nor distance")
        path_level, path_index = parse_asset_path_fields(row, bins)

        candidates.append(Candidate(
            source_word=word,
            candidate=text_s,
            level=lvl,
            distance=norm_dist,
            json_order=j,
            original_index=coerce_int(first_present(row, INDEX_KEYS)),
            output_index=path_index,
            output_level=path_level,
            raw=row,
        ))

    if order == "distance":
        candidates.sort(key=lambda c: (float("inf") if c.distance is None else c.distance, c.json_order))
    elif order == "level_distance":
        candidates.sort(key=lambda c: (float(c.level), float("inf") if c.distance is None else c.distance, c.json_order))
    elif order == "json":
        candidates.sort(key=lambda c: c.json_order)
    else:
        raise ValueError(f"Unsupported --candidate-order {order}")

    # run_full_lexicon style names use a zero-based sequence within each level folder.
    counters: Dict[str, int] = {}
    for c in candidates:
        if c.output_index is not None:
            pass
        elif c.original_index is not None and order == "json_keep_index":
            c.output_index = c.original_index
        else:
            output_level = c.output_level or c.level
            c.output_index = counters.get(output_level, 0)
            counters[output_level] = c.output_index + 1
        if c.output_level is None:
            c.output_level = c.level
    return candidates


def format_output_path(template: str, out_root: Path, word: str, candidate: Candidate, suffix: str) -> Path:
    idx = 0 if candidate.output_index is None else candidate.output_index
    level = candidate.output_level or candidate.level
    fields = {
        "out_root": str(out_root),
        "word": word,
        "source_word": word,
        "candidate": candidate.candidate,
        "variant": candidate.candidate,
        "level": level,
        "index": idx,
        "idx": idx,
        "suffix": suffix,
    }
    return Path(template.format(**fields))


def create_all_level_dirs(out_root: Path, words: Iterable[str], bins: int) -> None:
    for word in sorted(set(words)):
        for lvl in level_labels(bins):
            (out_root / word / f"level_{lvl}").mkdir(parents=True, exist_ok=True)


def candidate_key(candidate: Candidate) -> Tuple[int, str, int]:
    idx = -1 if candidate.output_index is None else candidate.output_index
    return candidate.json_order, candidate.output_level or candidate.level, idx


def raw_float(candidate: Candidate, keys: Iterable[str]) -> Optional[float]:
    if candidate.raw is None:
        return None
    return coerce_float(first_present(candidate.raw, keys))


def wav_index(candidate: Candidate) -> Optional[int]:
    if candidate.raw is None:
        return None
    value = first_present(candidate.raw, ("wav_path", "audio_path", "output_audio_path", "file", "filename", "path", "output"))
    if value is None:
        return None
    m = re.match(r"(\d+)_", Path(str(value)).name)
    return int(m.group(1)) if m else None


def representative_sort_key(candidate: Candidate, order: str) -> Tuple[float, int, str, int]:
    phone_distance = raw_float(candidate, PHONE_DISTANCE_KEYS)
    idx = wav_index(candidate)
    if order == "augmentation":
        primary = 0.0 if phone_distance is None else phone_distance
        secondary = 10**9 if idx is None else idx
    elif order == "wav_index":
        primary = 10**9 if idx is None else float(idx)
        secondary = candidate.json_order
    elif order == "json":
        primary = float(candidate.json_order)
        secondary = 0
    else:
        raise ValueError(f"Unsupported representative order: {order}")
    return primary, secondary, candidate.candidate, candidate.json_order


def representative_candidates(candidates: Sequence[Candidate], level: str, order: str) -> List[Candidate]:
    reps = [c for c in candidates if (c.output_level or c.level) == level]
    reps = reps or list(candidates)
    return sorted(reps, key=lambda c: representative_sort_key(c, order))


def candidates_by_text(candidates: Sequence[Candidate]) -> Dict[str, List[Candidate]]:
    grouped: Dict[str, List[Candidate]] = {}
    for cand in candidates:
        grouped.setdefault(cand.candidate, []).append(cand)
    return grouped

def set_transcript_triphone_path(gender, word):
    if gender == 'man' or gender == 'male':
        path = TRANSCRIPT_MAN_ASSET_DIR / Path('triphone') / Path(word)
    else:
        path = TRANSCRIPT_WOMAN_ASSET_DIR / Path('triphone') / Path(word)
    return path

def copy_json_scores(word, new_dir):
    json_fname = new_dir / f"{word}_blabber_scored.json"
    if not os.path.exists(json_fname):
        original_json = Path(ORIGINAL_ASSETS) / \
                word / f"{word}_blabber_scored.json"
        shutil.copy(original_json, json_fname)

def write_by_transcript_and_style(inp, chunk, aug_word, dry_run=False,
                                  man_suffix='en-AU-Ray', sr=48000, channels=2,
                                  sampwidth=100, woman_suffix='en-AU-Raychel'):
    # This is where it separates to a new asset directory
    voice_id = resolve_elevenlabs_voice_id(ELEVENLABS_VOICE_STRING)
    man_dir = set_transcript_triphone_path('man', inp.word)
    woman_dir = set_transcript_triphone_path('woman', inp.word)

    copy_json_scores(inp.word, man_dir)

    man_chunk = chunk
    try:
        woman_chunk = elevenlabs_voice_change(
            chunk, sr, channels, sampwidth, _API_KEY, voice_id,
            ELEVENLABS_MODEL_ID, ELEVENLABS_OUTPUT_FMT, False,
        )
        copy_json_scores(inp.word, woman_dir)
        chunk_info = ([man_dir, man_suffix, man_chunk],
                      [woman_dir, woman_suffix, woman_chunk])
    except RuntimeError:
        chunk_info = [[man_dir, man_suffix, man_chunk]]
        warnings.warn("API is not a paid subscription. ElevenLabs voice "\
                      "changer will not work")

    # for all of these words: save
    for m_dir, m_suffix, m_chunk in chunk_info:
        m_fname = m_dir /  Path(aug_word + "_" + m_suffix + '.wav')
        print("New asset path: ", m_fname)

        if not dry_run:
            write_wav_pcm(m_fname, sr, channels, sampwidth, chunk)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Chunk spoken blabber batches into run_full_lexicon-style WAV assets.")
    ap.add_argument(
        "--out-root", type=Path,
        default=ORIGINAL_ASSETS,
        help="Output root directory"
    )
    ap.add_argument(
        "--scored-root", type=Path,
        default=Path(f"{PARENT_DIR}/speech-assets-triphone/man/triphone/"),
        help="Root that contains {word}/scored.json from run_full_lexicon_blabber.sh"
    )
    # ap.add_argument("--scored-json-name", default="blabber_scored.json")
    ap.add_argument("--bins", type=int, default=8, help="Number of level bins. 8 gives 0.00, 0.14, ..., 1.00")
    ap.add_argument("--candidate-order", choices=("distance", "level_distance", "json"), default="distance")
    ap.add_argument(
        "--candidate-match-mode",
        choices=("text", "sequence"),
        default="text",
        help=(
            "Use 'text' when spoken chunks represent the highest level and should "
            "also populate duplicate lower-level candidate strings. Use 'sequence' "
            "for the legacy one-chunk-per-scored-row behavior."
        ),
    )
    ap.add_argument(
        "--representative-level",
        default=None,
        help="Level whose candidates correspond to spoken chunks in text mode. Default: highest bin, e.g. 1.00.",
    )
    ap.add_argument(
        "--representative-order",
        choices=("augmentation", "wav_index", "json"),
        default="wav_index",
        help=(
            "Order used to consume representative-level chunks in text mode. "
            "'wav_index' follows the original generated asset filenames, e.g. 0021_L1.00_nuh."
        ),
    )
    ap.add_argument("--suffix", type=str, default="en-AU-Ray")
    ap.add_argument("--bin-assignment", choices=("nearest", "floor", "ceil"), default="nearest")
    ap.add_argument("--allow-missing-scored", action="store_true", help="Fallback to legacy naming when scored.json is absent")
    ap.add_argument("--output-template", default="{out_root}/{word}/level_{level}/{index:04d}_L{level}_{candidate}_{suffix}.wav")
    ap.add_argument("--legacy-output-template", default="{out_root}/{word}/level_1.00/{word}_{index:03d}_{suffix}.wav")
    ap.add_argument("--dry-run", action="store_true", help="Detect and print chunks without writing WAVs")
    ap.add_argument("--metadata", type=Path, default=None, help="CSV metadata path. Default: {out_root}/chunk_metadata.csv")
    args = ap.parse_args(argv)

    wavs = iter_input_wavs(RAY_ASSET_DIR)
    if not wavs:
        raise SystemExit("No matching input WAV files found.")

    signal_cfg = SignalConfig()

    words = sorted(set(w.word for w in wavs))
    if not args.dry_run:
        create_all_level_dirs(args.out_root, words, args.bins)

    # from the previously output triphone dataset, get the names across levels
    # with transcript and index
    word_cfg_dicts = []
    for word in words:
        word_glob = glob(ORIGINAL_ASSETS + f"/{word}/**/*{ORIGINAL_VOICE}.wav")

        for wav_fname in word_glob:
            # split by level, index, and word
            fname_path = Path(wav_fname)
            level_dir = fname_path.parts[-2]
            fname = fname_path.parts[-1]
            index, level, aug_word, speaker = fname.split("_")
            tmp = "_".join([index, level, aug_word, args.suffix]) + '.wav'
            new_fname = args.out_root / Path(word) / Path(level_dir) / Path(tmp)
            word_cfg_dict = {
                'word': word,
                'aug_word': aug_word,
                'level_dir': level_dir,
                'level': level,
                'index': index,
                'original_fname': fname_path,
                'new_audio': np.nan,
                'new_fname': new_fname,
            }
            word_cfg_dicts.append(word_cfg_dict)

    all_aug_df = pd.DataFrame(word_cfg_dicts)
    lvl_standard = 'L1.00'

    transcript_dict = {}
    for word in words:
        df_lvl = all_aug_df[
            (
                (all_aug_df['level']==lvl_standard) & \
                (all_aug_df['word']==word) 
            )
        ].copy()
        df0 = all_aug_df[(all_aug_df['level']=='L0.00') & \
                         (all_aug_df['word']==word)].copy()
        df_lvl['idx_int'] = df_lvl['index'].map(lambda x: int(x))
        df0['idx_int'] = df0['index'].map(lambda x: int(x))
        df_lvl.sort_values(by='idx_int', ascending=False, inplace=True)
        df = pd.concat([df0, df_lvl])
        transcript_dict[word] = df[
            ['aug_word', 'index', 'new_audio', 'new_fname']
        ].reset_index(drop=True)
        print(df)

    metadata_path = args.metadata or (args.out_root / "chunk_metadata.csv")

    transcript_chunk_dict = {}
    for inp in wavs:
        sr, channels, sampwidth, data = read_wav_pcm(inp.path)
        print("processing ", inp.path)
        segments, threshold, noise_floor = detect_segments(
            data=data,
            sr=sr,
            sampwidth=sampwidth,
            frame_ms=signal_cfg.frame_ms,
            hop_ms=signal_cfg.hop_ms,
            threshold_db=signal_cfg.threshold_db,
            noise_percentile=signal_cfg.noise_percentile,
            above_noise_db=signal_cfg.above_noise_db,
            min_auto_db=signal_cfg.min_auto_db,
            max_auto_db=signal_cfg.max_auto_db,
            merge_gap_ms=signal_cfg.merge_gap_ms,
            min_speech_ms=signal_cfg.min_speech_ms,
            pad_ms=signal_cfg.pad_ms,
            max_chunks=signal_cfg.max_chunks_per_file,
        )

        file_idx = int(str(inp.path).split("_")[-1][:2])-1

        print(f"{inp.path.name}: {len(segments)} chunks,"\
              f"threshold={threshold:.1f} dBFS, "\
              "noise_p{signal_cfg.noise_percentile:g}={noise_floor:.1f} dBFS")

        # Patched to read the string and the line is the
        # index, search by string rather than by index
        with open(f"{TRANSCRIPT_DIR}/{inp.word}/{inp.word}_combinations.txt", 'r') as f:
            transcript = f.read()
        transcript = [text for text in transcript.split("\n") if text!=""]

        for local_i, seg in enumerate(segments, start=1):
            local_idx = local_i - 1
            aug_idx = file_idx*10 + local_idx

            chunk = data[seg.start_frame:seg.end_frame].copy()
            chunk = peak_normalize(chunk, sampwidth, signal_cfg.peak_normalize_dbfs)

            # This chunk is for this word and this augmentation, set to all
            # augmentation vocab and save to all new filenames
            aug_word = transcript[aug_idx]

            # This is where it separates to a new asset directory
            write_by_transcript_and_style(
                inp, chunk, aug_word, dry_run=args.dry_run,
                man_suffix=args.suffix, sr=sr, channels=channels,
                sampwidth=sampwidth, woman_suffix=args.suffix+"chel"
            )

            # paired by string, the rest follows
            new_fnames = all_aug_df[
                (all_aug_df['aug_word']==aug_word) & \
                (all_aug_df['word']==inp.word)
            ]['new_fname']

            # for all of these words: save
            for new_fname in new_fnames:
                print(f"{inp.word} \t {inp.path} \t {aug_word}")
                print(f"  {aug_idx:02d} -> {new_fname}  [{seg.start_s:.3f}, {seg.end_s:.3f}] {seg.end_s-seg.start_s:.3f}s")
                if not args.dry_run:
                    write_wav_pcm(new_fname, sr, channels, sampwidth, chunk)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
