
"""
mfcc_pronunciation_distance_morph_continuous.py

Continuous distance-driven pronunciation morphing for short command words.

Why this exists
---------------
The earlier script selected the nearest candidate label from a candidate path:
    level 0.45 -> "nuh"
    level 0.60 -> "naw"
    level 0.75 -> "nyoh"

That creates audible jumps, because most of the signal still uses the same
candidate until a threshold is crossed.

This version makes the shift gradual by:
  1. interpolating between adjacent candidate levels,
  2. applying frame-level vowel/MFCC target shifts continuously,
  3. gradually introducing glide/rhotic coloration,
  4. using a continuous original-vs-morphed mix,
  5. keeping candidate labels only for trace/debugging.

Scale:
  morph_strength = 0.0 -> original-like
  morph_strength = 1.0 -> full distance-driven morph

Distance mapping:
  linear, log, inverse_log, sigmoid are supported.

Distance normalization:
  absolute is the default so scores are treated as real 0..1 distances.
  This means scores = np.ones_like(times) produces maximum morphing instead
  of being min-max normalized into zeros.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
from os.path import splitext

import numpy as np
from scipy.io import wavfile
from scipy.signal import stft, istft, resample
from scipy.fftpack import dct, idct


# -----------------------------
# Audio + MFCC utilities
# -----------------------------

def hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 80, fmin: float = 40, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center > left:
            fb[i - 1, left:center] = (np.arange(left, center) - left) / max(1, center - left)
        if right > center:
            fb[i - 1, center:right] = (right - np.arange(center, right)) / max(1, right - center)
    return fb


def pinv_mel_filterbank(sr: int, n_fft: int, n_mels: int = 80) -> np.ndarray:
    return np.linalg.pinv(mel_filterbank(sr, n_fft, n_mels))


def load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(path)
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    if np.issubdtype(x.dtype, np.integer):
        denom = max(abs(np.iinfo(x.dtype).min), np.iinfo(x.dtype).max)
        x = x.astype(np.float32) / denom
    else:
        x = x.astype(np.float32)
    x = np.nan_to_num(x)
    x /= np.max(np.abs(x)) + 1e-9
    return sr, x


def save_wav(path: str | Path, sr: int, x: np.ndarray) -> None:
    x = np.nan_to_num(np.asarray(x, dtype=np.float32))
    x /= np.max(np.abs(x)) + 1e-9
    wavfile.write(path, sr, (x * 32767).astype(np.int16))


def extract_mfcc(x: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256, n_mels: int = 80, n_mfcc: int = 32):
    _, _, Z = stft(x, fs=sr, nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft, boundary="zeros")
    mag = np.abs(Z) + 1e-8
    fb = mel_filterbank(sr, n_fft, n_mels)
    mel = fb @ mag
    logmel = np.log(np.maximum(mel, 1e-8))
    mfcc = dct(logmel, axis=0, norm="ortho")[:n_mfcc]
    phase = np.angle(Z)
    return mfcc.astype(np.float32), phase.astype(np.float32)


def mfcc_to_audio(mfcc: np.ndarray, phase: np.ndarray, sr: int, n_fft: int = 1024, hop: int = 256, n_mels: int = 80) -> np.ndarray:
    n_mfcc, frames = mfcc.shape
    padded = np.zeros((n_mels, frames), dtype=np.float32)
    padded[:n_mfcc] = mfcc
    logmel = idct(padded, axis=0, norm="ortho")
    mel = np.exp(np.clip(logmel, -20, 8))
    invfb = pinv_mel_filterbank(sr, n_fft, n_mels)
    mag = np.maximum(invfb @ mel, 1e-8)
    Z = mag * np.exp(1j * phase[:, :frames])
    _, y = istft(Z, fs=sr, nperseg=n_fft, noverlap=n_fft-hop, nfft=n_fft, input_onesided=True)
    return y.astype(np.float32)


def normalize_01(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-8:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def normalize_distance_scores(
    scores: np.ndarray,
    mode: str = "absolute",
    constant_policy: str = "preserve",
) -> np.ndarray:
    """
    Normalize template-distance scores.

    mode:
        absolute:
            Treat scores as already meaningful 0..1 distances.
            This is the right default for synced template-distance signals.
            np.ones_like(times) stays all ones.

        minmax:
            Treat scores as relative values and min-max normalize them.
            This is useful for arbitrary raw distance magnitudes.

    constant_policy for minmax when all scores are equal:
        preserve:
            Keep clipped constant value. ones -> ones, zeros -> zeros.
        high:
            Constant curve means maximum distance.
        low:
            Constant curve means minimum distance.
    """
    scores = np.asarray(scores, dtype=np.float32)

    if mode == "absolute":
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    if mode == "minmax":
        lo, hi = float(np.min(scores)), float(np.max(scores))
        if hi - lo < 1e-8:
            if constant_policy == "high":
                return np.ones_like(scores, dtype=np.float32)
            if constant_policy == "low":
                return np.zeros_like(scores, dtype=np.float32)
            if constant_policy == "preserve":
                return np.clip(scores, 0.0, 1.0).astype(np.float32)
            raise ValueError(f"Unknown constant policy: {constant_policy}")

        return ((scores - lo) / (hi - lo)).astype(np.float32)

    raise ValueError(f"Unknown distance normalization mode: {mode}")


def smooth_1d(x: np.ndarray, window: int = 5) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    window = int(max(1, window))
    if window <= 1 or len(x) <= 2:
        return x
    k = np.ones(window, dtype=np.float32) / window
    return np.convolve(x, k, mode="same").astype(np.float32)


def smoothstep(x: np.ndarray | float) -> np.ndarray | float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# -----------------------------
# Candidate path
# -----------------------------

NO_CANDIDATE_PATH = [
    {"level": 0.00, "grapheme": "no", "approximate_phones": ["N", "OW"]},
    {"level": 0.15, "grapheme": "noh", "approximate_phones": ["N", "OW"]},
    {"level": 0.30, "grapheme": "noo", "approximate_phones": ["N", "UW"]},
    {"level": 0.45, "grapheme": "nuh", "approximate_phones": ["N", "AH"]},
    {"level": 0.60, "grapheme": "naw", "approximate_phones": ["N", "AO"]},
    {"level": 0.75, "grapheme": "nyoh", "approximate_phones": ["N", "Y", "OW"]},
    {"level": 0.90, "grapheme": "nuh-woh", "approximate_phones": ["N", "AH", "W", "OW"]},
    {"level": 1.00, "grapheme": "nyer-oh", "approximate_phones": ["N", "Y", "ER", "OW"]},
]

YES_CANDIDATE_PATH = [
    {"level": 0.00, "grapheme": "yes", "approximate_phones": ["Y", "EH", "S"]},
    {"level": 0.15, "grapheme": "yehs", "approximate_phones": ["Y", "EH", "S"]},
    {"level": 0.30, "grapheme": "yihs", "approximate_phones": ["Y", "IH", "S"]},
    {"level": 0.45, "grapheme": "yuhs", "approximate_phones": ["Y", "AH", "S"]},
    {"level": 0.60, "grapheme": "yeh-zuh", "approximate_phones": ["Y", "EH", "Z", "AH"]},
    {"level": 0.75, "grapheme": "yuh-yes", "approximate_phones": ["Y", "AH", "Y", "EH", "S"]},
    {"level": 0.90, "grapheme": "yer-yesh", "approximate_phones": ["Y", "ER", "Y", "EH", "SH"]},
    {"level": 1.00, "grapheme": "yuh-yer-esh", "approximate_phones": ["Y", "AH", "Y", "ER", "EH", "SH"]},
]

GO_CANDIDATE_PATH = [
    {"level": 0.00, "grapheme": "go", "approximate_phones": ["G", "OW"]},
    {"level": 0.15, "grapheme": "goh", "approximate_phones": ["G", "OW"]},
    {"level": 0.30, "grapheme": "goo", "approximate_phones": ["G", "UW"]},
    {"level": 0.45, "grapheme": "guh", "approximate_phones": ["G", "AH"]},
    {"level": 0.60, "grapheme": "gaw", "approximate_phones": ["G", "AO"]},
    {"level": 0.75, "grapheme": "gyoh", "approximate_phones": ["G", "Y", "OW"]},
    {"level": 0.90, "grapheme": "guh-woh", "approximate_phones": ["G", "AH", "W", "OW"]},
    {"level": 1.00, "grapheme": "gyer-oh", "approximate_phones": ["G", "Y", "ER", "OW"]},
]

STOP_CANDIDATE_PATH = [
    {"level": 0.00, "grapheme": "stop", "approximate_phones": ["S", "T", "AA", "P"]},
    {"level": 0.15, "grapheme": "stahp", "approximate_phones": ["S", "T", "AA", "P"]},
    {"level": 0.30, "grapheme": "stuhp", "approximate_phones": ["S", "T", "AH", "P"]},
    {"level": 0.45, "grapheme": "shtop", "approximate_phones": ["SH", "T", "AA", "P"]},
    {"level": 0.60, "grapheme": "suh-top", "approximate_phones": ["S", "AH", "T", "AA", "P"]},
    {"level": 0.75, "grapheme": "shtuh-pah", "approximate_phones": ["SH", "T", "AH", "P", "AH"]},
    {"level": 0.90, "grapheme": "suh-taw-puh", "approximate_phones": ["S", "AH", "T", "AO", "P", "AH"]},
    {"level": 1.00, "grapheme": "shter-aw-puh", "approximate_phones": ["SH", "T", "ER", "AO", "P", "AH"]},
]

PAIN_CANDIDATE_PATH = [
    {"level": 0.00, "grapheme": "pain", "approximate_phones": ["P", "EY", "N"]},
    {"level": 0.15, "grapheme": "payn", "approximate_phones": ["P", "EY", "N"]},
    {"level": 0.30, "grapheme": "pehn", "approximate_phones": ["P", "EH", "N"]},
    {"level": 0.45, "grapheme": "puhn", "approximate_phones": ["P", "AH", "N"]},
    {"level": 0.60, "grapheme": "pawn", "approximate_phones": ["P", "AO", "N"]},
    {"level": 0.75, "grapheme": "pyayn", "approximate_phones": ["P", "Y", "EY", "N"]},
    {"level": 0.90, "grapheme": "puh-wayn", "approximate_phones": ["P", "AH", "W", "EY", "N"]},
    {"level": 1.00, "grapheme": "pyer-ayn", "approximate_phones": ["P", "Y", "ER", "EY", "N"]},
]

HELP_CANDIDATE_PATH = [
    {"level": 0.00, "grapheme": "help", "approximate_phones": ["HH", "EH", "L", "P"]},
    {"level": 0.15, "grapheme": "hehp", "approximate_phones": ["HH", "EH", "P"]},
    {"level": 0.30, "grapheme": "hilp", "approximate_phones": ["HH", "IH", "L", "P"]},
    {"level": 0.45, "grapheme": "hulp", "approximate_phones": ["HH", "AH", "L", "P"]},
    {"level": 0.60, "grapheme": "heh-luhp", "approximate_phones": ["HH", "EH", "L", "AH", "P"]},
    {"level": 0.75, "grapheme": "hyelp", "approximate_phones": ["HH", "Y", "EH", "L", "P"]},
    {"level": 0.90, "grapheme": "hyer-lup", "approximate_phones": ["HH", "Y", "ER", "L", "AH", "P"]},
    {"level": 1.00, "grapheme": "hyuh-ler-puh", "approximate_phones": ["HH", "Y", "AH", "L", "ER", "P", "AH"]},
]

BATH_CANDIDATE_PATH = [
    {"level": 0.00, "grapheme": "bath", "approximate_phones": ["B", "AE", "TH"]},
    {"level": 0.15, "grapheme": "bahth", "approximate_phones": ["B", "AE", "TH"]},
    {"level": 0.30, "grapheme": "beth", "approximate_phones": ["B", "EH", "TH"]},
    {"level": 0.45, "grapheme": "buhth", "approximate_phones": ["B", "AH", "TH"]},
    {"level": 0.60, "grapheme": "bawth", "approximate_phones": ["B", "AO", "TH"]},
    {"level": 0.75, "grapheme": "byath", "approximate_phones": ["B", "Y", "AE", "TH"]},
    {"level": 0.90, "grapheme": "buh-wath", "approximate_phones": ["B", "AH", "W", "AE", "TH"]},
    {"level": 1.00, "grapheme": "byer-ath", "approximate_phones": ["B", "Y", "ER", "AE", "TH"]},
]

FOOD_CANDIDATE_PATH = [
    {"level": 0.00, "grapheme": "food", "approximate_phones": ["F", "UW", "D"]},
    {"level": 0.15, "grapheme": "foood", "approximate_phones": ["F", "UW", "D"]},
    {"level": 0.30, "grapheme": "fuwd", "approximate_phones": ["F", "UH", "D"]},
    {"level": 0.45, "grapheme": "fuhd", "approximate_phones": ["F", "AH", "D"]},
    {"level": 0.60, "grapheme": "fawd", "approximate_phones": ["F", "AO", "D"]},
    {"level": 0.75, "grapheme": "fyood", "approximate_phones": ["F", "Y", "UW", "D"]},
    {"level": 0.90, "grapheme": "fuh-wood", "approximate_phones": ["F", "AH", "W", "UW", "D"]},
    {"level": 1.00, "grapheme": "fyer-ood", "approximate_phones": ["F", "Y", "ER", "UW", "D"]},
]

CANDIDATE_PATHS = {
    "no": NO_CANDIDATE_PATH,
    "yes": YES_CANDIDATE_PATH,
    "go": GO_CANDIDATE_PATH,
    "stop": STOP_CANDIDATE_PATH,
    "pain": PAIN_CANDIDATE_PATH,
    "help": HELP_CANDIDATE_PATH,
    "bath": BATH_CANDIDATE_PATH,
    "food": FOOD_CANDIDATE_PATH,
}


VOWEL_SHIFT_TEMPLATES = {
    # Back/rounded vowels
    "OW": np.array([0.00, 0.00, 0.00, 0.00, 0.00, 0.00], dtype=np.float32),
    "UW": np.array([-0.45, -0.20, 0.25, 0.15, -0.08, 0.04], dtype=np.float32),
    "UH": np.array([-0.32, -0.12, 0.18, 0.10, -0.05, 0.03], dtype=np.float32),
    "AO": np.array([-0.25, 0.30, -0.15, 0.12, -0.05, 0.03], dtype=np.float32),
    "AA": np.array([-0.05, 0.42, -0.22, 0.14, -0.06, 0.02], dtype=np.float32),

    # Central/rhotic vowels
    "AH": np.array([0.35, -0.25, 0.18, -0.10, 0.05, -0.04], dtype=np.float32),
    "ER": np.array([0.40, -0.35, 0.35, -0.18, 0.10, -0.06], dtype=np.float32),

    # Front vowels
    "EH": np.array([0.18, 0.28, -0.18, 0.08, -0.03, 0.02], dtype=np.float32),
    "IH": np.array([0.30, 0.22, -0.14, 0.05, -0.02, 0.01], dtype=np.float32),
    "IY": np.array([0.42, 0.32, -0.20, 0.08, -0.04, 0.02], dtype=np.float32),
    "EY": np.array([0.25, 0.24, -0.16, 0.07, -0.03, 0.01], dtype=np.float32),
    "AE": np.array([0.10, 0.48, -0.24, 0.12, -0.05, 0.02], dtype=np.float32),

    # Diphthongs
    "AY": np.array([0.20, 0.12, 0.12, -0.08, 0.03, -0.02], dtype=np.float32),
}


def candidate_path_for_word(word: str) -> list[dict[str, Any]]:
    w = word.lower().strip()
    if w in CANDIDATE_PATHS:
        return CANDIDATE_PATHS[w]
    supported = ", ".join(sorted(CANDIDATE_PATHS))
    raise ValueError(f"Supported words: {supported}")


def interpolate_candidate(path: list[dict[str, Any]], level: float) -> dict[str, Any]:
    """
    Return a continuous state between adjacent candidates.
    This is only for trace and for finding the left/right phones.
    """
    level = float(np.clip(level, 0.0, 1.0))
    path = sorted(path, key=lambda c: c["level"])

    if level <= path[0]["level"]:
        return {"left": path[0], "right": path[0], "alpha": 0.0, "level": level}
    if level >= path[-1]["level"]:
        return {"left": path[-1], "right": path[-1], "alpha": 1.0, "level": level}

    for left, right in zip(path[:-1], path[1:]):
        l0, l1 = float(left["level"]), float(right["level"])
        if l0 <= level <= l1:
            alpha = (level - l0) / max(1e-8, l1 - l0)
            return {"left": left, "right": right, "alpha": float(alpha), "level": level}

    return {"left": path[-1], "right": path[-1], "alpha": 1.0, "level": level}


def target_vowel_for_candidate(candidate: dict[str, Any]) -> str:
    for p in candidate["approximate_phones"]:
        if p in VOWEL_SHIFT_TEMPLATES:
            return p
    return "OW"


def vector_for_vowel(phone: str, C: int) -> np.ndarray:
    n = min(7, C) - 1
    base = VOWEL_SHIFT_TEMPLATES.get(phone, VOWEL_SHIFT_TEMPLATES["OW"])
    return base[:n]


def continuous_vowel_shift(path: list[dict[str, Any]], level: float, C: int) -> np.ndarray:
    state = interpolate_candidate(path, level)
    left_v = target_vowel_for_candidate(state["left"])
    right_v = target_vowel_for_candidate(state["right"])
    alpha = smoothstep(state["alpha"])
    return (1.0 - alpha) * vector_for_vowel(left_v, C) + alpha * vector_for_vowel(right_v, C)


def continuous_insert_weights(word: str, level: float) -> dict[str, float]:
    """
    Gradually introduces glide/rhotic/coloration before candidate endpoints.

    These are generic enough for the supported command words while still keeping
    the word core intact: original onset/coda phones remain in the candidate path,
    and added Y/W/ER/AH coloring grows smoothly with level.
    """
    level = float(np.clip(level, 0.0, 1.0))
    w = word.lower().strip()

    # Defaults: mid/high levels gradually add palatal, labial, rhotic, and weak-vowel color.
    weights = {
        "Y": float(smoothstep((level - 0.55) / 0.25)),
        "W": float(smoothstep((level - 0.65) / 0.25)),
        "ER": float(smoothstep((level - 0.72) / 0.25)),
        "AH": float(smoothstep((level - 0.42) / 0.30)),
    }

    # Short open words can tolerate earlier glide/color.
    if w in {"no", "go"}:
        weights["Y"] = float(smoothstep((level - 0.50) / 0.25))
        weights["W"] = float(smoothstep((level - 0.62) / 0.28))
        weights["ER"] = float(smoothstep((level - 0.72) / 0.28))

    # Words with strong codas should introduce extra material slightly later.
    if w in {"stop", "help", "bath", "food", "pain", "yes"}:
        weights["Y"] = float(smoothstep((level - 0.60) / 0.25))
        weights["W"] = float(smoothstep((level - 0.70) / 0.25))
        weights["ER"] = float(smoothstep((level - 0.76) / 0.24))

    return weights


# -----------------------------
# Distance curve
# -----------------------------

def test_distance_curve():
    times = np.array([0.00, 0.08, 0.16, 0.25, 0.34, 0.42, 0.50, 0.58, 0.68, 0.75, 0.82, 0.88, 0.94, 1.00], dtype=np.float32)
    scores = np.array([0.5, 0.6, 0.7, 0.78, 0.48, 0.6, 0.4, 0.2, 0.20, 0.34, 0.24, 0.84, 0.80, 0.8], dtype=np.float32)
    return times, scores


def load_distance_csv(path: str | Path):
    times, scores = [], []
    with open(path, "r", newline="") as f:
        sample = f.read(1024)
        f.seek(0)
        has_header = "time" in sample.lower() or "score" in sample.lower() or "distance" in sample.lower()
        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                times.append(float(row.get("time", row.get("t"))))
                scores.append(float(row.get("score", row.get("distance"))))
        else:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    times.append(float(row[0]))
                    scores.append(float(row[1]))
    return np.asarray(times, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def perceptual_map_01(x: np.ndarray, mode: str = "log", curve: float = 6.0) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    curve = max(1e-6, float(curve))

    if mode == "linear":
        return x.astype(np.float32)

    if mode == "log":
        return (np.log1p(curve * x) / np.log1p(curve)).astype(np.float32)

    if mode == "inverse_log":
        return (1.0 - (np.log1p(curve * (1.0 - x)) / np.log1p(curve))).astype(np.float32)

    if mode == "sigmoid":
        steepness = max(0.1, curve)
        y = 1.0 / (1.0 + np.exp(-steepness * (x - 0.5)))
        y0 = 1.0 / (1.0 + np.exp(-steepness * (0.0 - 0.5)))
        y1 = 1.0 / (1.0 + np.exp(-steepness * (1.0 - 0.5)))
        return ((y - y0) / max(1e-8, y1 - y0)).astype(np.float32)

    raise ValueError(f"Unknown perceptual mapping mode: {mode}")


def distance_to_frames(
    n_frames,
    audio_len,
    sr,
    hop,
    times,
    scores,
    time_units="normalized",
    gamma=1.0,
    perceptual_mode="log",
    perceptual_curve=6.0,
    level_floor=0.0,
    level_gain=1.0,
    distance_normalization="absolute",
    constant_policy="preserve",
):
    if times is None or scores is None:
        return np.zeros(n_frames, dtype=np.float32)

    times = np.asarray(times, dtype=np.float32)
    scores = normalize_distance_scores(
        scores,
        mode=distance_normalization,
        constant_policy=constant_policy,
    )

    if time_units == "seconds":
        duration = audio_len / sr
        times = times / max(duration, 1e-8)

    times = np.clip(times, 0.0, 1.0)
    order = np.argsort(times)
    times = times[order]
    scores = scores[order]

    frame_pos = np.clip((np.arange(n_frames) * hop) / max(1, audio_len), 0.0, 1.0)
    d_linear = np.interp(frame_pos, times, scores)
    d_linear = smooth_1d(d_linear, 3)

    # Do not min-max normalize an absolute distance curve after interpolation.
    # If scores are all ones, this must remain all ones so it produces maximum morphing.
    if distance_normalization == "absolute":
        d_linear = np.clip(d_linear, 0.0, 1.0).astype(np.float32)
    else:
        d_linear = normalize_distance_scores(
            d_linear,
            mode="minmax",
            constant_policy=constant_policy,
        )

    d_shaped = d_linear ** max(0.05, float(gamma))
    d_perceptual = perceptual_map_01(d_shaped, mode=perceptual_mode, curve=perceptual_curve)

    # Critical for gradual audible change:
    # give all nonzero areas a low-level floor, and let user amplify.
    d_out = np.clip(level_floor + level_gain * d_perceptual, 0.0, 1.0)
    return d_out.astype(np.float32)


# -----------------------------
# Continuous MFCC morph
# -----------------------------

def apply_continuous_pronunciation_morph(
    mfcc: np.ndarray,
    pronunciation_level: np.ndarray,
    word: str,
    vowel_shift_strength: float = 1.25,
    centralization_strength: float = 0.45,
    glide_strength: float = 0.35,
    rhotic_strength: float = 0.40,
):
    C, T = mfcc.shape
    path = candidate_path_for_word(word)
    out = mfcc.copy()

    low = slice(1, min(7, C))
    high = slice(max(8, C // 3), C)

    # Frame-wise gradual vowel target movement.
    for t in range(T):
        level = float(np.clip(pronunciation_level[t], 0.0, 1.0))

        # Vowel shift interpolation.
        shift = continuous_vowel_shift(path, level, C)
        if shift.size:
            out[low, t] += vowel_shift_strength * level * shift

        # Centralization grows continuously, not candidate-wise.
        local_mean = float(np.mean(out[low, t]))
        out[low, t] = (1.0 - centralization_strength * level) * out[low, t] + (centralization_strength * level) * local_mean

        # Gradual glide/rhotic coloration.
        weights = continuous_insert_weights(word, level)

        y = weights.get("Y", 0.0)
        w = weights.get("W", 0.0)
        er = weights.get("ER", 0.0)
        ah = weights.get("AH", 0.0)

        if y > 0:
            y_vec = np.array([0.18, 0.10, -0.05, 0.04], dtype=np.float32)[:max(0, min(5, C)-1)]
            out[1:min(5, C), t] += glide_strength * y * y_vec
            out[high, t] *= 1.0 - 0.18 * y

        if w > 0:
            w_vec = np.array([-0.18, -0.08, 0.08, -0.04], dtype=np.float32)[:max(0, min(5, C)-1)]
            out[1:min(5, C), t] += glide_strength * w * w_vec
            out[high, t] *= 1.0 - 0.22 * w

        if ah > 0:
            ah_shift = vector_for_vowel("AH", C)
            out[low, t] += 0.45 * glide_strength * ah * ah_shift

        if er > 0:
            er_shift = vector_for_vowel("ER", C)
            out[low, t] += rhotic_strength * er * er_shift
            out[high, t] *= 1.0 - 0.20 * er

    # Smooth transitions gently so the morph is continuous.
    for c in range(C):
        sm = smooth_1d(out[c], 3)
        local = np.clip(0.25 * pronunciation_level, 0.0, 0.25)
        out[c] = (1.0 - local) * out[c] + local * sm

    return out


def apply_gradual_timing(mfcc, phase, pronunciation_level, word, morph_strength, timing_expansion=0.18):
    """
    Small continuous duration expansion rather than thresholded phone insertion.
    This makes high-level morphs breathe without sudden jumps.
    """
    strength = float(np.clip(morph_strength, 0.0, 1.0))
    avg_level = float(np.mean(pronunciation_level))
    max_level = float(np.max(pronunciation_level))

    if strength <= 1e-5 or max_level < 0.15:
        return mfcc, phase, []

    expansion = 1.0 + timing_expansion * strength * avg_level
    if word.lower() in CANDIDATE_PATHS:
        expansion += 0.10 * strength * float(np.mean(np.clip((pronunciation_level - 0.65) / 0.35, 0.0, 1.0)))

    new_len = max(1, int(round(mfcc.shape[1] * expansion)))
    out_mfcc = resample(mfcc, new_len, axis=1).astype(np.float32)
    out_phase = resample(phase, new_len, axis=1).astype(np.float32)
    return out_mfcc, out_phase, [{"effect": "continuous_duration_expansion", "scale": float(expansion)}]


def summarize_candidate_mix(path, pronunciation_level):
    states = []
    for q in [0.0, 0.25, 0.5, 0.75, 1.0]:
        idx = int(round(q * (len(pronunciation_level) - 1)))
        level = float(pronunciation_level[idx])
        state = interpolate_candidate(path, level)
        states.append({
            "position": q,
            "level": level,
            "left": state["left"]["grapheme"],
            "right": state["right"]["grapheme"],
            "alpha": state["alpha"],
        })
    return states


def process(
    input_wav: str | Path,
    output_wav: str | Path,
    word: str = "no",
    distance_mode: str = "test",
    distance_csv: str | None = None,
    distance_time_units: str = "normalized",
    morph_strength: float = 1.0,
    distance_gamma: float = 1.0,
    perceptual_mode: str = "log",
    perceptual_curve: float = 6.0,
    level_floor: float = 0.0,
    level_gain: float = 1.0,
    distance_normalization: str = "absolute",
    constant_policy: str = "preserve",
    vowel_shift_strength: float = 1.25,
    centralization_strength: float = 0.45,
    glide_strength: float = 0.35,
    rhotic_strength: float = 0.40,
    timing_expansion: float = 0.18,
    min_original_mix: float = 0.10,
    seed: int = 7,
    trace_json: str | Path | None = None,
):
    sr, x = load_wav(input_wav)
    n_fft, hop, n_mels, n_mfcc = 1024, 256, 80, 32
    mfcc, phase = extract_mfcc(x, sr, n_fft, hop, n_mels, n_mfcc)
    C, T = mfcc.shape

    if distance_csv:
        dt, ds = load_distance_csv(distance_csv)
    elif distance_mode == "test":
        dt, ds = test_distance_curve()
    else:
        dt, ds = None, None

    frame_distance = distance_to_frames(
        n_frames=T,
        audio_len=len(x),
        sr=sr,
        hop=hop,
        times=dt,
        scores=ds,
        time_units=distance_time_units,
        gamma=distance_gamma,
        perceptual_mode=perceptual_mode,
        perceptual_curve=perceptual_curve,
        level_floor=level_floor,
        level_gain=level_gain,
        distance_normalization=distance_normalization,
        constant_policy=constant_policy,
    )

    morph_strength = float(np.clip(morph_strength, 0.0, 1.0))
    pronunciation_level = np.clip(frame_distance * morph_strength, 0.0, 1.0)

    mfcc_morphed = apply_continuous_pronunciation_morph(
        mfcc,
        pronunciation_level,
        word=word,
        vowel_shift_strength=vowel_shift_strength,
        centralization_strength=centralization_strength,
        glide_strength=glide_strength,
        rhotic_strength=rhotic_strength,
    )

    mfcc_timed, phase_timed, timing_edits = apply_gradual_timing(
        mfcc_morphed,
        phase,
        pronunciation_level,
        word=word,
        morph_strength=morph_strength,
        timing_expansion=timing_expansion,
    )

    y = mfcc_to_audio(mfcc_timed, phase_timed, sr, n_fft, hop, n_mels)

    if len(y) != len(x):
        y = resample(y, len(x)).astype(np.float32)

    # Continuous mix: previous version had a fixed 0.28 floor at full strength.
    # Lower min_original_mix gives stronger audible morph, but keeps word core.
    original_mix = min_original_mix + (0.80 - min_original_mix) * (1.0 - morph_strength)
    out = original_mix * x + (1.0 - original_mix) * y

    rng = np.random.default_rng(seed)
    sample_curve = np.interp(np.linspace(0, 1, len(out)), np.linspace(0, 1, len(pronunciation_level)), pronunciation_level)
    out += rng.normal(0, 1, len(out)).astype(np.float32) * (0.004 * sample_curve)
    out = np.tanh(out * (1.0 + 0.20 * morph_strength))
    save_wav(output_wav, sr, out)

    path = candidate_path_for_word(word)
    trace = {
        "input_wav": str(input_wav),
        "output_wav": str(output_wav),
        "word": word,
        "mode": "continuous_interpolated_candidate_path",
        "morph_strength": morph_strength,
        "candidate_path": path,
        "distance_mode": distance_mode,
        "distance_csv": distance_csv,
        "distance_time_units": distance_time_units,
        "distance_gamma": distance_gamma,
        "perceptual_mode": perceptual_mode,
        "perceptual_curve": perceptual_curve,
        "level_floor": level_floor,
        "level_gain": level_gain,
        "distance_normalization": distance_normalization,
        "constant_policy": constant_policy,
        "mfcc_frames": int(T),
        "frame_distance_min": float(np.min(frame_distance)),
        "frame_distance_max": float(np.max(frame_distance)),
        "pronunciation_level_mean": float(np.mean(pronunciation_level)),
        "pronunciation_level_max": float(np.max(pronunciation_level)),
        "candidate_interpolation_samples": summarize_candidate_mix(path, pronunciation_level),
        "continuous_parameters": {
            "vowel_shift_strength": vowel_shift_strength,
            "centralization_strength": centralization_strength,
            "glide_strength": glide_strength,
            "rhotic_strength": rhotic_strength,
            "timing_expansion": timing_expansion,
            "min_original_mix": min_original_mix,
        },
        "edits": {
            "timing": timing_edits,
        },
        "endpoint": {
            "level": 1.0,
            "grapheme": path[-1]["grapheme"],
            "approximate_phones": path[-1]["approximate_phones"],
        },
    }

    if trace_json:
        Path(trace_json).write_text(json.dumps(trace, indent=2))

    return trace


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_wav")
    p.add_argument("output_wav")
    p.add_argument("--word", default="no")
    p.add_argument("--distance-mode", choices=["none", "test"], default="test")
    p.add_argument("--distance-csv", default=None)
    p.add_argument("--distance-time-units", choices=["normalized", "seconds"], default="normalized")
    p.add_argument("--distance-gamma", type=float, default=1.0)
    p.add_argument("--perceptual-mode", choices=["linear", "log", "inverse_log", "sigmoid"], default="log")
    p.add_argument("--perceptual-curve", type=float, default=6.0)

    # Continuous graduality controls.
    p.add_argument("--level-floor", type=float, default=0.08, help="Adds a low-level audible morph floor after distance mapping.")
    p.add_argument("--level-gain", type=float, default=1.15, help="Amplifies mapped distance before clipping.")
    p.add_argument(
        "--distance-normalization",
        choices=["absolute", "minmax"],
        default="absolute",
        help="absolute preserves 0..1 scores as-is; minmax treats scores as relative.",
    )
    p.add_argument(
        "--constant-policy",
        choices=["preserve", "high", "low"],
        default="preserve",
        help="Behavior for constant curves under minmax. preserve keeps ones as ones.",
    )
    p.add_argument("--vowel-shift-strength", type=float, default=1.25)
    p.add_argument("--centralization-strength", type=float, default=0.45)
    p.add_argument("--glide-strength", type=float, default=0.35)
    p.add_argument("--rhotic-strength", type=float, default=0.40)
    p.add_argument("--timing-expansion", type=float, default=0.18)
    p.add_argument("--min-original-mix", type=float, default=0.10)

    # Sweep behavior.
    p.add_argument("--sweep", action="store_true", default=True, help="Generate a 0..1 morph-strength sweep. Enabled by default.")
    p.add_argument("--no-sweep", action="store_false", dest="sweep", help="Generate only one output using --morph-strength.")
    p.add_argument("--morph-strength", type=float, default=1.0, help="Used only when --no-sweep is set.")
    p.add_argument("--sweep-steps", type=int, default=10)
    p.add_argument("--sweep-start", type=float, default=0.0)
    p.add_argument("--sweep-end", type=float, default=1.0)

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--trace-json", default=None)

    args = p.parse_args()

    def output_name_for_strength(base_output: str, strength: float) -> str:
        tmp = splitext(base_output)
        # Keep your original naming pattern, but format the float so filenames are cleaner.
        strength_token = f"{strength:.3f}".rstrip("0").rstrip(".")
        return f"{args.perceptual_mode}_" + tmp[0] + f"{strength_token}" + tmp[1]

    def trace_name_for_strength(base_trace: str | None, out_fname: str, strength: float) -> str | None:
        if base_trace is None:
            tmp = splitext(out_fname)
            return tmp[0] + "_trace.json"

        tmp = splitext(base_trace)
        strength_token = f"{strength:.3f}".rstrip("0").rstrip(".")
        return tmp[0] + f"_{strength_token}" + tmp[1]

    if args.sweep:
        steps = max(2, int(args.sweep_steps))
        for morph_strength in np.linspace(args.sweep_start, args.sweep_end, steps):
            out_fname = output_name_for_strength(args.output_wav, float(morph_strength))
            trace_fname = trace_name_for_strength(args.trace_json, out_fname,
                                                  float(morph_strength)) if \
                    args.trace_json is not None else None

            process(
                input_wav=args.input_wav,
                output_wav=out_fname,
                word=args.word,
                distance_mode=args.distance_mode,
                distance_csv=args.distance_csv,
                distance_time_units=args.distance_time_units,
                morph_strength=float(morph_strength),
                distance_gamma=args.distance_gamma,
                perceptual_mode=args.perceptual_mode,
                perceptual_curve=args.perceptual_curve,
                level_floor=args.level_floor,
                level_gain=args.level_gain,
                distance_normalization=args.distance_normalization,
                constant_policy=args.constant_policy,
                vowel_shift_strength=args.vowel_shift_strength,
                centralization_strength=args.centralization_strength,
                glide_strength=args.glide_strength,
                rhotic_strength=args.rhotic_strength,
                timing_expansion=args.timing_expansion,
                min_original_mix=args.min_original_mix,
                seed=args.seed,
                trace_json=trace_fname,
            )
    else:
        process(
            input_wav=args.input_wav,
            output_wav=args.output_wav,
            word=args.word,
            distance_mode=args.distance_mode,
            distance_csv=args.distance_csv,
            distance_time_units=args.distance_time_units,
            morph_strength=args.morph_strength,
            distance_gamma=args.distance_gamma,
            perceptual_mode=args.perceptual_mode,
            perceptual_curve=args.perceptual_curve,
            level_floor=args.level_floor,
            level_gain=args.level_gain,
            distance_normalization=args.distance_normalization,
            constant_policy=args.constant_policy,
            vowel_shift_strength=args.vowel_shift_strength,
            centralization_strength=args.centralization_strength,
            glide_strength=args.glide_strength,
            rhotic_strength=args.rhotic_strength,
            timing_expansion=args.timing_expansion,
            min_original_mix=args.min_original_mix,
            seed=args.seed,
            trace_json=args.trace_json,
        )


if __name__ == "__main__":
    main()
