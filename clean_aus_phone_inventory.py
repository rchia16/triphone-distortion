#!/usr/bin/env python3
"""
Temporary utility: crop each Australian phone asset to its first audio segment.

Default input:
  /data/raqchia/audio-assets/aus-phones

Default output:
  /data/raqchia/audio-assets/aus-phones/cleaned

The script assumes the desired isolated phone is the first non-silent audio
segment in each file. It writes WAV outputs by default so downstream stitching
does not depend on MP3 decoding.

Example:
  python clean_aus_phone_inventory.py --overwrite
"""
from __future__ import annotations

from pathlib import Path
import argparse
import json

import numpy as np
import soundfile as sf


DEFAULT_INPUT_DIR = Path("/data/raqchia/audio-assets/aus-phones")
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "cleaned"
AUDIO_SUFFIXES = {".mp3", ".wav", ".flac", ".ogg", ".m4a"}


def load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio)
    return audio, int(sample_rate)


def smooth_envelope(audio: np.ndarray, sample_rate: int, window_ms: float) -> np.ndarray:
    window = max(1, int(round(sample_rate * window_ms / 1000.0)))
    envelope = np.abs(audio).astype(np.float32)
    if window <= 1 or envelope.size == 0:
        return envelope
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(envelope, kernel, mode="same")


def first_segment_bounds(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_ratio: float,
    min_segment_ms: float,
    max_internal_silence_ms: float,
    pad_ms: float,
    envelope_window_ms: float,
) -> tuple[int, int]:
    if audio.size == 0:
        return 0, 0

    envelope = smooth_envelope(audio, sample_rate, envelope_window_ms)
    peak = float(np.max(envelope)) if envelope.size else 0.0
    if peak <= 0.0:
        return 0, min(audio.size, max(1, int(round(sample_rate * min_segment_ms / 1000.0))))

    threshold = peak * float(threshold_ratio)
    active = envelope >= threshold
    min_len = max(1, int(round(sample_rate * min_segment_ms / 1000.0)))
    silence_bridge = max(0, int(round(sample_rate * max_internal_silence_ms / 1000.0)))
    pad = max(0, int(round(sample_rate * pad_ms / 1000.0)))

    indices = np.flatnonzero(active)
    if indices.size == 0:
        return 0, min(audio.size, min_len)

    start = int(indices[0])
    end = start
    last_active = start
    for index in indices[1:]:
        index = int(index)
        if index - last_active > silence_bridge:
            break
        end = index
        last_active = index

    if end <= start:
        end = start + min_len
    if end - start < min_len:
        end = start + min_len

    return max(0, start - pad), min(audio.size, end + pad)


def clean_file(path: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    audio, sample_rate = load_mono(path)
    start, end = first_segment_bounds(
        audio,
        sample_rate,
        threshold_ratio=args.threshold_ratio,
        min_segment_ms=args.min_segment_ms,
        max_internal_silence_ms=args.max_internal_silence_ms,
        pad_ms=args.pad_ms,
        envelope_window_ms=args.envelope_window_ms,
    )

    output_path = output_dir / f"{path.stem}.wav"
    if output_path.exists() and not args.overwrite:
        status = "exists"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), audio[start:end], sample_rate, subtype="PCM_16")
        status = "ok"

    return {
        "input": str(path),
        "output": str(output_path),
        "status": status,
        "sample_rate": sample_rate,
        "start_sec": start / sample_rate if sample_rate else 0.0,
        "end_sec": end / sample_rate if sample_rate else 0.0,
        "duration_sec": (end - start) / sample_rate if sample_rate else 0.0,
    }


def iter_audio_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_SUFFIXES
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Crop Australian phone assets to their first non-silent segment.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--threshold-ratio", type=float, default=0.08)
    parser.add_argument("--min-segment-ms", type=float, default=80.0)
    parser.add_argument("--max-internal-silence-ms", type=float, default=120.0)
    parser.add_argument("--pad-ms", type=float, default=25.0)
    parser.add_argument("--envelope-window-ms", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest-name", default="cleaned_manifest.json")
    args = parser.parse_args()

    files = iter_audio_files(args.input_dir)
    results = [clean_file(path, args.output_dir, args) for path in files]
    manifest = {
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "asset_count": len(results),
        "settings": {
            "threshold_ratio": args.threshold_ratio,
            "min_segment_ms": args.min_segment_ms,
            "max_internal_silence_ms": args.max_internal_silence_ms,
            "pad_ms": args.pad_ms,
            "envelope_window_ms": args.envelope_window_ms,
        },
        "assets": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / args.manifest_name
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"asset_count": len(results), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
