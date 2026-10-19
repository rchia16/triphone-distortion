#!/usr/bin/env python3
"""
Create JSON-driven PanPhon substitution demo videos.

For each requested word and direction, this script:
  1. reads {transcript_root}/{word}_substitution_transcripts.json
  2. selects the entry whose distance tuple best matches lo2hi or hi2lo
  3. loads the selected substitution WAV from entry['path'][gender]
  4. slows the pronunciation to 0.6x speed using ffmpeg atempo
  5. renders a distance graph timed to the slowed pronunciation segments
  6. writes:
       asset-demo/{direction}/{word}_panphon_sub.wav
       asset-demo/{direction}/{word}_panphon_sub.mp4

Example:
  python make_asset_demo_videos.py --transcript-root tests --asset-root . --gender man
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import read, write


WORDS = ["yes", "stop"]
DIRECTIONS = ["hi2lo", "lo2hi"]
DEFAULT_OUTPUT_ROOT = Path("asset-demo")
DEFAULT_GENDER = "man"
DEFAULT_TEMPO = 0.6
DEFAULT_FPS = 30

VOCAB_PHONE_DICT = {
    "yes": ["Y", "EH", "S"],
    "stop": ["S", "T", "O", "P"],
}

# Source-word timing proportions reused from temporal_feedback_test.py and scaled
# to the duration of the selected slowed TTS pronunciation.
VOCAB_TIMESTAMP_DICT = {
    "yes": [
        (0.12, 0.23),  # Y
        (0.23, 0.33),  # EH
        (0.33, 0.75),  # S
    ],
    "stop": [
        (0.12, 0.31),  # S
        (0.31, 0.43),  # T
        (0.43, 0.54),  # O
        (0.54, 0.73),  # P
    ],
}

PHONE_TO_GRAPHEME = {
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "g",
    "S": "s", "Z": "z", "SH": "sh", "ZH": "zh", "CH": "ch", "JH": "j",
    "TH": "th", "DH": "dh", "F": "f", "V": "v", "HH": "h",
    "M": "m", "N": "n", "NG": "ng", "L": "l", "R": "r", "Y": "y", "W": "w",
    "IH": "ih", "IY": "ee", "EH": "eh", "AE": "a", "AH": "uh", "AX": "uhx",
    "AA": "ah", "AO": "aw", "UH": "u", "UW": "oo", "ER": "er",
    "OW": "oh", "EY": "ay", "AY": "eye", "AW": "ow", "OY": "oy", "O": "o",
}


def load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    sr, x = read(path)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.max(np.abs(x)) > 1.5:
        x /= 32768.0
    return sr, np.clip(x, -1.0, 1.0)


def write_wav_float(path: Path, sr: int, x: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(x, dtype=np.float32)
    y = y / (np.max(np.abs(y)) + 1e-9) * 0.95
    write(path, sr, (np.clip(y, -1.0, 1.0) * 32767).astype(np.int16))


def ffmpeg_atempo_chain(tempo: float) -> str:
    if tempo <= 0:
        raise ValueError("tempo must be positive")
    parts: list[str] = []
    while tempo < 0.5:
        parts.append("atempo=0.5")
        tempo /= 0.5
    while tempo > 2.0:
        parts.append("atempo=2.0")
        tempo /= 2.0
    parts.append(f"atempo={tempo:.8f}")
    return ",".join(parts)


def slow_wav_with_ffmpeg(input_wav: Path, output_wav: Path, tempo: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required")
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(input_wav),
            "-filter:a",
            ffmpeg_atempo_chain(tempo),
            str(output_wav),
        ],
        check=True,
    )


def load_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # New format: list of records.
    if isinstance(data, list):
        return data

    # Legacy format: {distance: new_word}. Convert enough to still demo.
    if isinstance(data, dict):
        return [
            {
                "distance": distance,
                "word": path.name.replace("_substitution_transcripts.json", ""),
                "new_word": new_word,
                "path": {},
            }
            for distance, new_word in data.items()
        ]

    raise ValueError(f"Unsupported JSON manifest format: {path}")


def parse_distance_tuple(distance: str) -> np.ndarray:
    return np.array([float(part) for part in distance.split("_")], dtype=float)


def target_profile(n: int, direction: str) -> np.ndarray:
    if direction == "lo2hi":
        return np.linspace(0.0, 1.0, n)
    if direction == "hi2lo":
        return np.linspace(1.0, 0.0, n)
    raise ValueError(f"Unknown direction: {direction}")


def select_manifest_record(manifest: list[dict], word: str, direction: str) -> dict:
    best_record = None
    best_score = math.inf

    for record in manifest:
        values = parse_distance_tuple(record["distance"])
        target = target_profile(len(values), direction)
        score = float(np.mean((values - target) ** 2))

        # Prefer non-original when tied so the demo visibly changes.
        if record.get("new_word") == word:
            score += 1e-6

        if score < best_score:
            best_score = score
            best_record = record

    if best_record is None:
        raise ValueError(f"No valid entries found for {word} {direction}")

    return best_record


def load_debug_ladders(debug_path: Path) -> dict:
    if not debug_path.exists():
        return {}
    with open(debug_path, "r", encoding="utf-8") as f:
        return json.load(f)


def distance_label(x: float) -> str:
    x = round(float(x), 2)
    if abs(x) < 1e-9:
        return "0.0"
    if abs(x - 1.0) < 1e-9:
        return "1.0"
    return f"{x:.2f}".rstrip("0").rstrip(".")


def bin_labels_from_debug(word: str, distances: np.ndarray, debug: dict) -> list[str]:
    phones = VOCAB_PHONE_DICT[word]
    labels: list[str] = []

    for i, (source_phone, dist) in enumerate(zip(phones, distances)):
        prefix = f"{word}:{i}:{source_phone}:"
        rows = []
        for key, value in debug.items():
            if key.startswith(prefix):
                rows = value
                break

        matched_phone = None
        if rows:
            for row in rows:
                if abs(float(row.get("rank_distance", -99)) - float(dist)) <= 0.011:
                    matched_phone = row.get("phone")
                    break

        if matched_phone is None:
            matched_phone = source_phone

        labels.append(PHONE_TO_GRAPHEME.get(matched_phone, matched_phone.lower()))

    return labels


def normalized_segment_times(word: str) -> list[tuple[float, float]]:
    raw = VOCAB_TIMESTAMP_DICT[word]
    start0 = raw[0][0]
    end_last = raw[-1][1]
    duration = end_last - start0
    if duration <= 0:
        raise ValueError(f"Invalid timestamps for {word}")
    return [((a - start0) / duration, (b - start0) / duration) for a, b in raw]


def segment_windows(word: str, duration_s: float) -> list[tuple[float, float]]:
    norm = normalized_segment_times(word)
    return [(a * duration_s, b * duration_s) for a, b in norm]


def expand_distance_signal(values: np.ndarray, windows: list[tuple[float, float]], points_per_segment: int = 24) -> tuple[np.ndarray, np.ndarray]:
    repeated_values = values
    xs: list[float] = []
    ys: list[float] = []
    for value, (a, b) in zip(repeated_values, windows):
        segment_x = np.linspace(a, b, points_per_segment)
        xs.extend(segment_x.tolist())
        ys.extend([float(value)] * len(segment_x))
    return np.array(xs), np.array(ys)


def render_distance_video(
    output_mp4: Path,
    audio_wav: Path,
    word: str,
    new_word: str,
    direction: str,
    distances: np.ndarray,
    labels: list[str],
    duration_s_for_segments: float,
    fps: int,
) -> None:
    sr, audio = load_wav_mono(audio_wav)
    duration_s = len(audio) / sr
    n_frames = max(1, int(np.ceil(duration_s * fps)))

    windows = segment_windows(word, duration_s_for_segments)
    plot_x, plot_y = expand_distance_signal(distances, windows)
    repeated_labels = labels
    repeated_distances = distances

    y_min = -0.05
    y_max = 1.10

    temp_dir = Path(tempfile.mkdtemp(prefix="asset_demo_frames_"))
    silent_video = temp_dir / "silent.mp4"
    try:
        frame_paths: list[Path] = []
        colors = plt.cm.Set3(np.linspace(0, 1, len(distances)))

        for frame_i in range(n_frames):
            current_t = min(duration_s, frame_i / fps)
            reveal_idx = np.searchsorted(plot_x, current_t, side="right")

            fig, ax = plt.subplots(figsize=(11, 5), dpi=120)

            for j, ((a, b), label, value) in enumerate(zip(windows, repeated_labels, repeated_distances)):
                color = colors[j % len(distances)]
                active = current_t >= a
                ax.axvspan(a, b, color=color, alpha=0.32 if active else 0.08, zorder=0)
                ax.text(
                    (a + b) / 2,
                    1.055,
                    f"{label}\n{value:.2g}",
                    ha="center",
                    va="top",
                    fontsize=12,
                    fontweight="bold",
                    alpha=1.0 if active else 0.35,
                    zorder=5,
                )


            ax.plot(plot_x, plot_y, alpha=0.18, linewidth=3)
            ax.plot(plot_x[:reveal_idx], plot_y[:reveal_idx], linewidth=3)
            ax.axvline(current_t, linestyle="--", linewidth=1.2)

            ax.set_xlim(0, duration_s)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Distance bin")
            ax.set_title(f"{word} -> {new_word}")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()

            frame_path = temp_dir / f"frame_{frame_i:05d}.png"
            fig.savefig(frame_path)
            plt.close(fig)
            frame_paths.append(frame_path)

        writer = imageio.get_writer(str(silent_video), fps=fps, codec="libx264", quality=8)
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))
        writer.close()

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("ffmpeg is required to mux audio")

        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(silent_video),
                "-i",
                str(audio_wav),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output_mp4),
            ],
            check=True,
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def build_slow_audio(source_wav: Path, output_wav: Path, tempo: float) -> float:
    with tempfile.TemporaryDirectory(prefix="asset_demo_audio_") as tmp:
        tmp = Path(tmp)
        slowed_wav = tmp / "slowed_once.wav"
        slow_wav_with_ffmpeg(source_wav, slowed_wav, tempo=tempo)
        sr, slowed = load_wav_mono(slowed_wav)
        write_wav_float(output_wav, sr, slowed)
        return len(slowed) / sr


def resolve_audio_path(record: dict, gender: str, asset_root: Path) -> Path:
    paths = record.get("path") or {}
    if gender not in paths:
        raise KeyError(
            f"Manifest entry for {record.get('word')} {record.get('new_word')} has no path for gender={gender!r}. "
            "Regenerate the JSON with path records, or choose a gender present in the JSON."
        )
    path = Path(paths[gender])
    if not path.is_absolute():
        path = asset_root / path
    if not path.exists():
        raise FileNotFoundError(f"Audio asset not found: {path}")
    return path


def create_demo_for_word_direction(
    word: str,
    direction: str,
    transcript_root: Path,
    asset_root: Path,
    output_root: Path,
    gender: str,
    tempo: float,
    fps: int,
    overwrite: bool,
) -> None:
    manifest_path = transcript_root / f"{word}_substitution_transcripts.json"
    manifest = load_manifest(manifest_path)
    record = select_manifest_record(manifest, word, direction)
    distances = parse_distance_tuple(record["distance"])

    source_wav = resolve_audio_path(record, gender=gender, asset_root=asset_root)
    out_dir = output_root / direction
    out_wav = out_dir / f"{word}_panphon_sub.wav"
    out_mp4 = out_dir / f"{word}_panphon_sub.mp4"

    if out_wav.exists() and out_mp4.exists() and not overwrite:
        print(f"Exists, skipped {out_wav} and {out_mp4}")
        return

    debug = load_debug_ladders(transcript_root / "panphon_phone_ladders_debug.json")
    labels = bin_labels_from_debug(word, distances, debug)

    duration_s_for_segments = build_slow_audio(
        source_wav=source_wav,
        output_wav=out_wav,
        tempo=tempo,
    )

    render_distance_video(
        output_mp4=out_mp4,
        audio_wav=out_wav,
        word=word,
        new_word=record["new_word"],
        direction=direction,
        distances=distances,
        labels=labels,
        duration_s_for_segments=duration_s_for_segments,
        fps=fps,
    )

    print(f"Saved {out_wav}")
    print(f"Saved {out_mp4}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--transcript-root", default="tests", type=Path)
    p.add_argument("--asset-root", default=".", type=Path)
    p.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, type=Path)
    p.add_argument("--gender", default=DEFAULT_GENDER)
    p.add_argument("--tempo", default=DEFAULT_TEMPO, type=float, help="0.6 means 60% speed")
    p.add_argument("--fps", default=DEFAULT_FPS, type=int)
    p.add_argument("--words", nargs="+", default=WORDS)
    p.add_argument("--directions", nargs="+", default=DIRECTIONS)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    for direction in args.directions:
        for word in args.words:
            create_demo_for_word_direction(
                word=word,
                direction=direction,
                transcript_root=args.transcript_root,
                asset_root=args.asset_root,
                output_root=args.output_root,
                gender=args.gender,
                tempo=args.tempo,
                fps=args.fps,
                overwrite=args.overwrite,
            )


if __name__ == "__main__":
    main()
