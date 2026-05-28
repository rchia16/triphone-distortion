"""
Guide:
    * selection first filters candidates by voice mode (female vs male), word/label,
    and candidate (global vs per-phoneme).
    * For triphone feedback control, collapse the EEG temporal mismatch wave to one
    whole-word augmentation-strength target, then rank library assets by closeness
    to that strength instead of rewarding canonical phones first.

How to read `candidate_temporal_profile` in logs:
    * It is a per-time-step distortion signature inferred from candidate phones
    relative to canonical phones.
    * Each value is approximately in `[0, 1]`:
        0.0 -> near-canonical at that local segment
        1.0 -> strong local phonetic distortion (or insertion/deletion gap penalty)
    * It is not an EEG signal. It is a model-side distortion template.
    * It is compared to `temporal_target_profile` (derived from per_time_l2) using
    MSE; lower MSE means a better temporal fit for the current trial.

Environment variables (selection behavior):
    * `TRIPHONE_ASSET_ROOT`: root containing triphone libraries grouped by
    voice/word.
    * `REQUESTED_VOICE`: requested bank (`female`/`male` aliases are supported).
    * `TRIPHONE_LEVEL_WEIGHT`: base selector weight for level proximity.
    * `L2_LEVEL_MIN`, `L2_LEVEL_MAX`: normalize L2 to `[0,1]` control space.
    * `L2_BLEND_MEAN`, `L2_BLEND_P90`: blend global mean and p90 mismatch when
    computing target strength.
    * `LOW_DISTANCE_DEADBAND`: values below this normalized distance map to zero
    strength (prevents jitter near solved state).
    * `LEVEL_POLICY`:
        `distance_targeted` (default) -> temporal/control-aware augmentation.
        `highest_strict`, `highest`, `high_bias` -> alternate high-level policies.
    * `HIGH_LEVEL_MIN`: lower bound for `highest`/`high_only` policy.
    * `TEMPORAL_TOP_K`: number of top selector candidates to temporally rerank.
    * `TEMPORAL_MATCH_WEIGHT`: how strongly temporal fit influences final choice.
    * `TOP_K_DEBUG`: how many final ranked candidates to print.
    * `PHONE_GAP_PENALTY`: penalty for phone insert/delete during alignment;
    larger values make length mismatch look more distorted.
"""
from pathlib import Path
import os
import argparse
import json
import pprint
import sys

import numpy as np
from scipy.fftpack import dct
from scipy.io import wavfile
from scipy.signal import stft, resample

from Template_l2_compare_v2 import (
    compare_signal_to_prebuilt_template,
    build_template_from_dataset,
    get_one_trial_from_dataset,
)
from deep_phone_candidate_stack_blabber_triphone import articulatory_distance
from select_blabber_asset import (
    adapt_comparison_result,
    normalize_voice_mode,
)

from select_blabber_asset_triphone import load_assets, select_assets, asset_level


REQUESTED_VOICE = "female"
TRIPHONE_ASSET_ROOT = Path(
    os.environ.get(
        "TRIPHONE_ASSET_ROOT",
        r"/data/raqchia/audio-assets/speech-assets-triphone",
    )
)
GENERATION_MODE = "triphone"

# Selection-time level influence in `select_assets`:
#   level_score = max(0, 1 - abs(asset_level - target_level))
#   total += TRIPHONE_LEVEL_WEIGHT * level_score
#
# Intuition:
#   0.0 -> ignore level (purely phone/similarity driven)
#   ~0.5-1.0 -> weak level preference
#   ~2.0-4.0 -> moderate level influence
#   ~6.0+ -> level is dominant (default for global search mode)
#
# There is no hard clamp in the selector; keep this non-negative and usually in
# [0, 10] for predictable behavior.
TRIPHONE_LEVEL_WEIGHT = 6.0

# normalisation levers
L2_LEVEL_MIN = float(os.environ.get("L2_LEVEL_MIN", "0.06"))
L2_LEVEL_MAX = float(os.environ.get("L2_LEVEL_MAX", "0.30"))

# regulariser for distance mapping (global mean vs 90th percentile distance)
L2_BLEND_MEAN = float(os.environ.get("L2_BLEND_MEAN", "0.6"))
# regulariser for distance mapping (global mean vs 90th percentile distance)
L2_BLEND_P90 = float(os.environ.get("L2_BLEND_P90", "0.4"))

# increase for softer
LOW_DISTANCE_DEADBAND = float(os.environ.get("LOW_DISTANCE_DEADBAND", "0.2"))
LEVEL_POLICY = str(os.environ.get("LEVEL_POLICY", "distance_targeted")).strip().lower()
# decrease for softer
HIGH_LEVEL_MIN = float(os.environ.get("HIGH_LEVEL_MIN", "0.85"))

TOP_K_DEBUG = int(os.environ.get("TOP_K_DEBUG", "5"))
# decrease for softer
TEMPORAL_TOP_K = int(os.environ.get("TEMPORAL_TOP_K", "5"))
# decrease for softer
TEMPORAL_MATCH_WEIGHT = float(os.environ.get("TEMPORAL_MATCH_WEIGHT", "3.0"))
PHONE_GAP_PENALTY = 0.65
RANKING_MODE = str(os.environ.get("RANKING_MODE", "temporal")).strip().lower()
CMVN_N_FFT = int(os.environ.get("CMVN_N_FFT", "1024"))
CMVN_HOP = int(os.environ.get("CMVN_HOP", "256"))
CMVN_N_MELS = int(os.environ.get("CMVN_N_MELS", "80"))
CMVN_N_MFCC = int(os.environ.get("CMVN_N_MFCC", "32"))
CMVN_DTW_MAX_FRAMES = int(os.environ.get("CMVN_DTW_MAX_FRAMES", "0"))

LEXICON = {
    "go": ["G", "OW"],
    "stop": ["S", "T", "OW", "P"],
    "bath": ["B", "AE", "TH"],
    "food": ["F", "UW", "D"],
    "yes": ["Y", "EH", "S"],
    "no": ["N", "OW"],
    "pain": ["P", "EY", "N"],
    "help": ["HH", "EH", "L", "P"],
}


def clamp_unit(value: float) -> float:
    """Clamp a numeric value to the closed interval [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _percentile(values: list[float], q: float) -> float:
    """Return a linearly interpolated percentile for a non-empty numeric list."""
    if not values:
        return 0.0
    if q <= 0:
        return float(min(values))
    if q >= 100:
        return float(max(values))
    ordered = sorted(float(v) for v in values)
    rank = (len(ordered) - 1) * (q / 100.0)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    w = rank - lo
    return ordered[lo] * (1.0 - w) + ordered[hi] * w


def _normalize_l2_to_level(value: float) -> float:
    """Normalize raw L2 magnitude into selector control space [0, 1]."""
    denom = max(1e-8, float(L2_LEVEL_MAX) - float(L2_LEVEL_MIN))
    return clamp_unit((float(value) - float(L2_LEVEL_MIN)) / denom)


def global_target_level(comparison: dict) -> tuple[float, dict]:
    """
    Map per-time L2 mismatch values to a single [0,1] target level.

    This target level is passed into selector scoring (not generation). During
    generation, candidate suitability already includes its own internal level
    match term based on phone-distance proximity to that candidate's level.
    """
    values = [float(value) for value in comparison["per_time_l2"]]
    if not values:
        return 0.0, {"mean_l2": 0.0, "p90_l2": 0.0, "blended_l2": 0.0}

    mean_l2 = sum(values) / float(len(values))
    p90_l2 = _percentile(values, 90.0)
    blended_l2 = float(L2_BLEND_MEAN) * mean_l2 + float(L2_BLEND_P90) * p90_l2
    normalized_distance = _normalize_l2_to_level(blended_l2)
    if normalized_distance <= LOW_DISTANCE_DEADBAND:
        target_strength = 0.0
    else:
        target_strength = clamp_unit(
            (normalized_distance - LOW_DISTANCE_DEADBAND) / max(1e-8, 1.0 - LOW_DISTANCE_DEADBAND)
        )
    return target_strength, {
        "mean_l2": mean_l2,
        "p90_l2": p90_l2,
        "blended_l2": blended_l2,
        "normalized_distance": normalized_distance,
        "target_strength": target_strength,
        "l2_level_min": float(L2_LEVEL_MIN),
        "l2_level_max": float(L2_LEVEL_MAX),
        "blend_mean_weight": float(L2_BLEND_MEAN),
        "blend_p90_weight": float(L2_BLEND_P90),
        "low_distance_deadband": float(LOW_DISTANCE_DEADBAND),
    }


def library_json_path(word: str, voice_mode: str) -> Path:
    """Resolve the triphone library JSON path for a word and voice mode."""
    return TRIPHONE_ASSET_ROOT / voice_mode / GENERATION_MODE / word / f"{word}_blabber_scored.json"


def _asset_level(asset: dict) -> float | None:
    """Read the effective distortion level from an asset payload."""
    return asset_level(asset)


def _load_audio_mono(path: str | Path) -> tuple[int, np.ndarray]:
    """Load a WAV file as a mono float32 waveform in [-1, 1]."""
    sr, audio = wavfile.read(str(path))
    audio = np.asarray(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if np.issubdtype(audio.dtype, np.integer):
        denom = float(max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max))
        audio = audio.astype(np.float32) / max(1.0, denom)
    else:
        audio = audio.astype(np.float32)
    audio = np.nan_to_num(audio)
    peak = float(np.max(np.abs(audio)))
    if peak > 0:
        audio = audio / peak
    return int(sr), audio.astype(np.float32)


def _hz_to_mel(hz: np.ndarray | float) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _mel_to_hz(mel: np.ndarray | float) -> np.ndarray:
    return 700.0 * (10.0 ** (np.asarray(mel) / 2595.0) - 1.0)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, fmin: float = 40.0, fmax: float | None = None) -> np.ndarray:
    if fmax is None:
        fmax = sr / 2.0
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        if center > left:
            fb[i - 1, left:center] = (np.arange(left, center) - left) / max(1, center - left)
        if right > center:
            fb[i - 1, center:right] = (right - np.arange(center, right)) / max(1, right - center)
    return fb


def _extract_mfcc(audio: np.ndarray, sr: int) -> np.ndarray:
    """Extract MFCC features for CMVN-based ranking."""
    _, _, z = stft(audio, fs=sr, nperseg=CMVN_N_FFT, noverlap=CMVN_N_FFT - CMVN_HOP, nfft=CMVN_N_FFT, boundary="zeros")
    mag = np.abs(z) + 1e-8
    mel = _mel_filterbank(sr, CMVN_N_FFT, CMVN_N_MELS) @ mag
    logmel = np.log(np.maximum(mel, 1e-8))
    mfcc = dct(logmel, axis=0, norm="ortho")[:CMVN_N_MFCC]
    return mfcc.astype(np.float32)


def _cmvn(features: np.ndarray) -> np.ndarray:
    """Apply per-utterance CMVN."""
    if features.size == 0:
        return features.astype(np.float32)
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    return ((features - mean) / np.maximum(std, 1e-8)).astype(np.float32)


def _resample_time_axis(features: np.ndarray, frames: int) -> np.ndarray:
    """Resample features along the time axis to a fixed frame count."""
    if features.shape[1] == frames:
        return features
    if frames <= 0:
        return features[:, :0]
    return resample(features, frames, axis=1).astype(np.float32)


def _dtw_distance(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute DTW distance between two feature matrices using squared Euclidean frame cost."""
    n, m = X.shape[1], Y.shape[1]
    if n == 0 or m == 0:
        return float("inf")
    prev = np.full(m + 1, np.inf, dtype=np.float64)
    curr = np.full(m + 1, np.inf, dtype=np.float64)
    prev[0] = 0.0
    for i in range(1, n + 1):
        curr[0] = np.inf
        xi = X[:, i - 1]
        for j in range(1, m + 1):
            cost = float(np.sum((xi - Y[:, j - 1]) ** 2))
            curr[j] = cost + min(curr[j - 1], prev[j], prev[j - 1])
        prev, curr = curr, prev
    return float(prev[m] / max(1, n + m))


def _cmvn_distance(reference_path: str | Path, candidate_path: str | Path) -> float:
    """Score candidate audio against the reference audio using CMVN-normalized MFCC DTW."""
    ref_sr, ref_audio = _load_audio_mono(reference_path)
    cand_sr, cand_audio = _load_audio_mono(candidate_path)
    ref_mfcc = _cmvn(_extract_mfcc(ref_audio, ref_sr))
    cand_mfcc = _cmvn(_extract_mfcc(cand_audio, cand_sr))
    if ref_mfcc.size == 0 or cand_mfcc.size == 0:
        return float("inf")
    if CMVN_DTW_MAX_FRAMES > 0:
        ref_mfcc = _resample_time_axis(ref_mfcc, CMVN_DTW_MAX_FRAMES)
        cand_mfcc = _resample_time_axis(cand_mfcc, CMVN_DTW_MAX_FRAMES)
    return _dtw_distance(ref_mfcc, cand_mfcc)


def _find_cmvn_reference_asset(rows: list[dict], canonical: list[str]) -> dict | None:
    """Prefer the ground-truth audio as the CMVN reference; fall back to the closest canonical match."""
    for row in rows:
        asset = row["asset"]
        phones = list(asset.get("phones") or [])
        canonical_phones = list(asset.get("canonical_phones") or canonical)
        if bool(asset.get("is_ground_truth")) or phones == canonical_phones:
            return asset
    return rows[0]["asset"] if rows else None


def _normalized_time_distance_profile(comparison: dict) -> list[float]:
    """Convert per-time L2 values into a deadbanded target distortion curve."""
    values = [float(v) for v in comparison["per_time_l2"]]
    if not values:
        return []
    profile: list[float] = []
    for value in values:
        normalized = _normalize_l2_to_level(value)
        if normalized <= LOW_DISTANCE_DEADBAND:
            profile.append(0.0)
        else:
            profile.append(
                clamp_unit((normalized - LOW_DISTANCE_DEADBAND) / max(1e-8, 1.0 - LOW_DISTANCE_DEADBAND))
            )
    return profile


def _align_phone_distances(source: list[str], candidate: list[str]) -> list[float]:
    """Align source/candidate phone sequences and emit local distortion costs."""
    n = len(source)
    m = len(candidate)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + PHONE_GAP_PENALTY
        back[i][0] = (i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + PHONE_GAP_PENALTY
        back[0][j] = (0, j - 1)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sub_cost = dp[i - 1][j - 1] + articulatory_distance(source[i - 1], candidate[j - 1])
            del_cost = dp[i - 1][j] + PHONE_GAP_PENALTY
            ins_cost = dp[i][j - 1] + PHONE_GAP_PENALTY
            best_cost = min(sub_cost, del_cost, ins_cost)
            dp[i][j] = best_cost
            if best_cost == sub_cost:
                back[i][j] = (i - 1, j - 1)
            elif best_cost == del_cost:
                back[i][j] = (i - 1, j)
            else:
                back[i][j] = (i, j - 1)

    distances: list[float] = []
    i, j = n, m
    while i > 0 or j > 0:
        prev = back[i][j]
        if prev is None:
            break
        pi, pj = prev
        if pi == i - 1 and pj == j - 1:
            distances.append(articulatory_distance(source[i - 1], candidate[j - 1]))
        else:
            distances.append(PHONE_GAP_PENALTY)
        i, j = pi, pj
    distances.reverse()
    return distances or [0.0]


def _resample_profile(values: list[float], size: int) -> list[float]:
    """Resample a 1-D profile to a fixed length using linear interpolation."""
    if size <= 0:
        return []
    if not values:
        return [0.0] * size
    if len(values) == 1:
        return [float(values[0])] * size

    result: list[float] = []
    for idx in range(size):
        pos = ((len(values) - 1) * idx) / max(1, size - 1)
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        frac = pos - lo
        result.append(values[lo] * (1.0 - frac) + values[hi] * frac)
    return result


def candidate_temporal_signature(canonical: list[str], candidate_phones: list[str], steps: int) -> list[float]:
    """
    Build a time-shaped candidate distortion profile.

    Steps:
      1. Align canonical phones to candidate phones.
      2. Convert each aligned phone pair into local articulatory distance.
      3. Resample to match EEG time resolution (`steps`).
    """
    local_distances = _align_phone_distances(canonical, candidate_phones)
    return _resample_profile(local_distances, steps)


def temporal_profile_match(target_profile: list[float], candidate_profile: list[float]) -> tuple[float, dict]:
    """
    Compare EEG-derived temporal target profile vs candidate profile.

    `temporal_match` is `1 - clipped_mse`, so higher is better.
    """
    if not target_profile or not candidate_profile or len(target_profile) != len(candidate_profile):
        return 0.0, {"temporal_match": 0.0, "temporal_mse": 1.0}

    mse = sum((a - b) ** 2 for a, b in zip(target_profile, candidate_profile)) / float(len(target_profile))
    match = max(0.0, 1.0 - min(1.0, mse))
    return match, {
        "temporal_match": match,
        "temporal_mse": mse,
    }


def rerank_top_candidates_by_temporal_profile(
    *,
    rows: list[dict],
    comparison: dict,
    canonical: list[str],
) -> list[dict]:
    """Rerank preselected candidates by temporal profile fit to EEG mismatch."""
    target_profile = _normalized_time_distance_profile(comparison)
    reranked: list[dict] = []
    for row in rows:
        asset = row["asset"]
        candidate_phones = list(asset.get("phones") or [])
        # 0 more canonical, 1 more distorted according to match
        candidate_profile = candidate_temporal_signature(canonical, candidate_phones, len(target_profile))
        temporal_match, temporal_details = temporal_profile_match(target_profile, candidate_profile)
        combined_score = float(row["score"]) + TEMPORAL_MATCH_WEIGHT * temporal_match
        details = dict(row["score_details"])
        details.update(temporal_details)
        reranked.append({
            "score": combined_score,
            "base_score": row["score"],
            "temporal_target_profile": target_profile,
            "candidate_temporal_profile": candidate_profile,
            "score_details": details,
            "asset": asset,
        })

    reranked.sort(
        key=lambda r: (
            -r["score"],
            -r["score_details"].get("temporal_match", 0.0),
            -(r["asset"].get("suitability_score") or 0.0),
            r["asset"].get("candidate_text") or "",
            r["asset"].get("asset_id") or "",
        )
    )
    return reranked


def rerank_candidates_by_cmvn_audio(
    *,
    rows: list[dict],
    canonical: list[str],
) -> list[dict]:
    """Rerank candidates by audio-space CMVN distance to a reference asset."""
    reference_asset = _find_cmvn_reference_asset(rows, canonical)
    if reference_asset is None:
        return []
    reference_audio = reference_asset.get("audio_path") or reference_asset.get("wav_path") or reference_asset.get("output_audio_path")
    if not reference_audio:
        return []

    reranked: list[dict] = []
    for row in rows:
        asset = row["asset"]
        candidate_audio = asset.get("audio_path") or asset.get("wav_path") or asset.get("output_audio_path")
        if not candidate_audio:
            continue
        distance = _cmvn_distance(reference_audio, candidate_audio)
        details = dict(row["score_details"])
        details.update(
            {
                "cmvn_reference_asset_id": reference_asset.get("asset_id"),
                "cmvn_distance": distance,
            }
        )
        reranked.append(
            {
                "score": -distance,
                "base_score": row["score"],
                "score_details": details,
                "asset": asset,
                "cmvn_reference_asset": reference_asset,
                "cmvn_distance": distance,
            }
        )

    reranked.sort(
        key=lambda r: (
            r["cmvn_distance"],
            -(r["asset"].get("suitability_score") or 0.0),
            r["asset"].get("candidate_text") or "",
            r["asset"].get("asset_id") or "",
        )
    )
    return reranked


def select_triphone_global_asset(comparison: dict, requested_voice: str) -> dict:
    """Select one triphone asset using global + temporal feedback criteria."""
    word = str(comparison.get("label_name") or "").strip().lower()
    if word not in LEXICON:
        raise ValueError(f"No canonical phone sequence configured for word '{word}'.")

    voice_mode = normalize_voice_mode(requested_voice)
    library_path = library_json_path(word, voice_mode)
    if not library_path.is_file():
        raise FileNotFoundError(f"Missing triphone library JSON: {library_path}")

    target_level, target_level_info = global_target_level(comparison)
    assets, _payload = load_assets(library_path)
    min_level = None
    max_level = None
    effective_target_level = target_level
    feedback_mode = "distance_targeted"
    if LEVEL_POLICY in {"highest_strict", "strict_highest"}:
        max_available = max((_asset_level(a) for a in assets if _asset_level(a) is not None), default=None)
        if max_available is not None:
            min_level = max_available - 1e-9
            max_level = max_available + 1e-9
        effective_target_level = 1.0
        feedback_mode = "canonical"
    elif LEVEL_POLICY in {"highest", "high_only"}:
        # Hard-filter to the highest bucket region.
        min_level = clamp_unit(HIGH_LEVEL_MIN)
        effective_target_level = 1.0
        feedback_mode = "canonical"
    elif LEVEL_POLICY in {"high_bias", "biased_high"}:
        # Keep all levels searchable but bias rank toward high distortion.
        effective_target_level = max(target_level, 0.85)
        feedback_mode = "canonical"
    elif LEVEL_POLICY in {"distance_targeted", "targeted"}:
        effective_target_level = target_level

    selected = select_assets(
        assets,
        word=word,
        phones=LEXICON[word],
        triphones=[],
        voice=voice_mode,
        target_level=effective_target_level,
        min_level=min_level,
        max_level=max_level,
        top_k=max(1, TOP_K_DEBUG),
        level_weight=TRIPHONE_LEVEL_WEIGHT,
        global_search=True,
        feedback_mode=feedback_mode,
    )
    if not selected:
        raise ValueError(f"No triphone assets matched word='{word}' voice='{voice_mode}'.")

    if RANKING_MODE in {"cmvn", "audio_cmvn"}:
        temporal_pool = selected
        reranked = rerank_candidates_by_cmvn_audio(
            rows=temporal_pool,
            canonical=LEXICON[word],
        )
        if not reranked:
            temporal_pool = selected[:max(1, TEMPORAL_TOP_K)]
            reranked = rerank_top_candidates_by_temporal_profile(
                rows=temporal_pool,
                comparison=comparison,
                canonical=LEXICON[word],
            )
    else:
        temporal_pool = selected[:max(1, TEMPORAL_TOP_K)]
        reranked = rerank_top_candidates_by_temporal_profile(
            rows=temporal_pool,
            comparison=comparison,
            canonical=LEXICON[word],
        )
    best = reranked[0]
    asset = best["asset"]
    return {
        "word": word,
        "voice_mode": voice_mode,
        "generation_mode": GENERATION_MODE,
        "library_json": str(library_path),
        "audio_path": asset.get("audio_path") or asset.get("wav_path") or asset.get("output_audio_path"),
        "asset_id": asset.get("asset_id"),
        "target_level": target_level,
        "effective_target_level": effective_target_level,
        "target_level_info": target_level_info,
        "feedback_mode": feedback_mode,
        "level_policy": LEVEL_POLICY,
        "ranking_mode": RANKING_MODE,
        "high_level_min": clamp_unit(HIGH_LEVEL_MIN),
        "requested_generation_signature": [target_level],
        "selected_phoneme_grades": asset.get("phoneme_values", [asset.get("level")]),
        "score": best["score"],
        "base_score": best.get("base_score"),
        "score_details": best["score_details"],
        "asset_metadata": asset,
        "top_candidates": [
            {
                "asset_id": row["asset"].get("asset_id"),
                "candidate_text": row["asset"].get("candidate_text"),
                "level": _asset_level(row["asset"]),
                "phone_distance": row["asset"].get("phone_distance"),
                "is_ground_truth": bool(row["asset"].get("is_ground_truth")) or (
                    list(row["asset"].get("phones") or []) == list(row["asset"].get("canonical_phones") or [])
                ),
                "score": row["score"],
                "base_score": row.get("base_score"),
                "score_details": row["score_details"],
                "candidate_temporal_profile": row.get("candidate_temporal_profile"),
                "cmvn_distance": row.get("cmvn_distance"),
                "cmvn_reference_asset_id": row.get("score_details", {}).get("cmvn_reference_asset_id"),
            }
            for row in reranked[:TOP_K_DEBUG]
        ],
        "times": list(comparison["times"]),
        "per_time_l2": list(comparison["per_time_l2"]),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--json-out", default=None, help="Write the selection result to this JSON file.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite --json-out if it already exists.")
    p.add_argument("--ranking-mode", default=RANKING_MODE, choices=["temporal", "cmvn", "audio_cmvn"], help="Ranking mode to use for final candidate selection.")
    p.add_argument("--requested-voice", default=REQUESTED_VOICE, choices=["female", "male", "woman", "man"], help="Requested voice bank.")
    args = p.parse_args()

    RANKING_MODE = str(args.ranking_mode).strip().lower()
    REQUESTED_VOICE = str(args.requested_voice).strip().lower()

    """
    Take from `Function call copy_v2.ipynb`
    """
    # -----------------------------------------------------------------
    # 1) Configure trial/template slice
    # -----------------------------------------------------------------
    template_days = [1]
    template_sessions = range(1, 3)
    label = 4

    # -----------------------------------------------------------------
    # 2) Build template bank
    # -----------------------------------------------------------------
    template_object = build_template_from_dataset(
        template_days=template_days,
        template_sessions=template_sessions,
        nperseg=128,
        noverlap=96,
        eps=1e-8,
        fmax=50,
    )

    # -----------------------------------------------------------------
    # 3) Load one reference trial
    # -----------------------------------------------------------------
    trial, label_name, dataset = get_one_trial_from_dataset(
        day=1,
        sess=8,
        label=label,
        trial_index_within_label=0,
    )
    print("Loaded trial label:", label_name)

    # -----------------------------------------------------------------
    # 4) Compare trial against template (distance signal extraction)
    # -----------------------------------------------------------------
    result = compare_signal_to_prebuilt_template(
        template_object=template_object,
        input_signal=trial,
        input_label=label,
        output_mode="both",
        sr=250,
    )
    pprint.pprint(result)

    # -----------------------------------------------------------------
    # 5) Adapt comparison output and select asset
    # -----------------------------------------------------------------
    comparison = adapt_comparison_result(result)
    pprint.pprint(comparison)
    selection = select_triphone_global_asset(
        comparison=comparison,
        requested_voice=REQUESTED_VOICE,
    )

    # -----------------------------------------------------------------
    # 6) Report chosen asset and ranking diagnostics
    # -----------------------------------------------------------------
    print("Selected asset:", selection["audio_path"])
    print("Selected asset id:", selection["asset_id"])
    print("Generation mode:", selection["generation_mode"])
    print("Target level:", selection["target_level"])
    print("Effective target level:", selection["effective_target_level"])
    print("Target level info:", selection["target_level_info"])
    print("Feedback mode:", selection["feedback_mode"])
    print("Level policy:", selection["level_policy"], "high_level_min:", selection["high_level_min"])
    print("Score details:", selection["score_details"])
    print("Base score:", selection["base_score"])
    print("Top candidates:")
    pprint.pprint(selection["top_candidates"])

    if args.json_out:
        out_path = Path(args.json_out)
        if out_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing file: "\
                f"{out_path}. Pass --overwrite to replace it."
            )
        out_path.write_text(
            json.dumps(selection, indent=2, default=str),
            encoding="utf-8"
        )
