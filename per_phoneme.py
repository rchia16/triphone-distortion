#!/usr/bin/env python3
"""
Per-phoneme candidate selection and phone-asset stitching.

Purpose:
  This script maps an EEG/template mismatch signal to a new ARPABET phone
  sequence, then optionally pregenerates stitched word WAVs from isolated
  Australian English phone assets.

Inputs:
  - Target word, e.g. `yes`.
  - Comparison JSON containing `per_time_l2`.
  - Isolated phone inventory, default:
      /data/raqchia/audio-assets/aus-phones/cleaned

Important setup:
  Run `clean_aus_phone_inventory.py --overwrite` first if the cleaned inventory
  does not exist. The cleaned inventory should contain one isolated first phone
  segment per file, e.g. `j.wav`, `ɛ.wav`, `s.wav`.

How selection works:
  1. Resolve the word to canonical ARPABET phones with cmudict when available.
     Built-in fallback words are: yes, no, bath, pain, help, food, go, stop.
  2. Convert `comparison["per_time_l2"]` into one normalized target per phone.
  3. Generate same-length phone substitutions.
  4. Load or create phone-level pairwise CMVN distances from the inventory.
  5. Score each candidate phone-by-phone against the canonical phones.
  6. Reject candidates above `--cmvn-phone-max-threshold`.
  7. Return the best valid generated phone sequence and per-phone diagnostics.

CMVN pairwise cache:
  The script does not require `--library-json` or rendered word assets.
  It compares isolated phone audio directly. If the cache does not exist, it is
  created automatically at:
      <phone-inventory-root>/cmvn_pairwise_distances.json

  The cache stores raw CMVN-MFCC DTW distances plus normalized [0,1] distances.
  Selection uses the normalized distances so the default threshold scale remains
  interpretable. Use `--rebuild-cmvn-pairwise` to force recomputation.

Required comparison JSON shape:
  {
    "per_time_l2": [0.08, 0.12, 0.30],
    "times": [0.0, 0.1, 0.2]
  }

How to preview without audio:
  python per_phoneme.py --preview

  Preview uses `yes,no,bath,pain,help,food,go,stop` and the default signal
  `0.1,0.3,0.6,0.9`. For 2-phone words it averages the first and second half.
  For 3-phone words it smooths with first two, middle two, and last two values.

How to select a new phone word from EEG/template mismatch:
  python per_phoneme.py \\
    --word yes \\
    --comparison-json comparison.json \\
    --phone-inventory-root /data/raqchia/audio-assets/aus-phones/cleaned \\
    --cmvn-phone-max-threshold 0.75 \\
    --json-out yes_per_phoneme_selection.json

How to rebuild the CMVN pairwise cache:
  python per_phoneme.py \\
    --word yes \\
    --comparison-json comparison.json \\
    --phone-inventory-root /data/raqchia/audio-assets/aus-phones/cleaned \\
    --rebuild-cmvn-pairwise

How to pregenerate stitched per-phoneme word assets:
  python per_phoneme.py \\
    --pregenerate-assets \\
    --render-audio \\
    --render-audio-source phone-stitch \\
    --phone-inventory-root /data/raqchia/audio-assets/aus-phones/cleaned \\
    --words yes,no,bath,pain,help,food,go,stop \\
    --voices woman,man \\
    --granularity 2

Pregeneration output:
  - WAVs and JSON are written under `per-phoneme/<voice>/<word>/`.
  - Distance filenames look like `0.0_1.0_0.0.wav`.
  - Candidates with missing phone inventory assets are skipped.

Runtime controls:
  CMVN_PHONE_MAX_THRESHOLD: default max normalized per-phone CMVN distance.
  PHONEME_EEG_WEIGHT: score weight for filling the EEG-derived target.
  PHONEME_CMVN_WEIGHT: score weight for staying below the CMVN threshold.
  PHONEME_OPTIONS_PER_SLOT: max substitutions considered per canonical phone.
  PHONEME_DYNAMIC_MAX_CANDIDATES: max generated phone sequences to evaluate.
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
from types import SimpleNamespace
from typing import Any

import numpy as np

from Template_l2_compare_v2 import (
    build_template_from_dataset,
    compare_signal_to_prebuilt_template,
    get_one_trial_from_dataset,
)
from deep_phone_candidate_stack_blabber_triphone import (
    PHONE_FEATURES,
    articulatory_distance,
    make_renderer,
    render_candidate,
    validate_rendered_duration,
    variants_for_phone,
)
from my_test import (
    _cmvn,
    _extract_mfcc,
    _load_audio_mono,
    _normalized_time_distance_profile,
    _resample_profile,
    _resample_time_axis,
    _dtw_distance,
    clamp_unit,
    CMVN_DTW_MAX_FRAMES,
    TOP_K_DEBUG,
)
from select_blabber_asset import adapt_comparison_result, normalize_voice_mode


CMVN_PHONE_MAX_THRESHOLD = float(os.environ.get("CMVN_PHONE_MAX_THRESHOLD", "1.0"))
PHONEME_EEG_WEIGHT = float(os.environ.get("PHONEME_EEG_WEIGHT", "1.0"))
PHONEME_CMVN_WEIGHT = float(os.environ.get("PHONEME_CMVN_WEIGHT", "1.0"))
PHONEME_OPTIONS_PER_SLOT = int(os.environ.get("PHONEME_OPTIONS_PER_SLOT", "32"))
PHONEME_DYNAMIC_MAX_CANDIDATES = int(os.environ.get("PHONEME_DYNAMIC_MAX_CANDIDATES", "256"))
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

DEFAULT_PHONE_INVENTORY_ROOT = "/data/raqchia/audio-assets/aus-phones/cleaned"
DEFAULT_OUTPUT_ROOT = "per-phoneme"
DEFAULT_REQUESTED_VOICE = "man"

# Local mapping for the Australian isolated-phone inventory. Keep this local to
# avoid importing generator scripts with CLI/runtime side effects.
ARPABET_TO_AUS_IPA_CANDIDATES: dict[str, list[str]] = {
    "P": ["p"],
    "B": ["b"],
    "T": ["t"],
    "D": ["d"],
    "K": ["k"],
    "G": ["g", "ɡ"],
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
    "AE": ["æ"],
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
    """Convert CMUdict stress-marked ARPABET phones like AH0 to AH."""
    return re.sub(r"\d+$", "", str(phone).strip().upper())


def resolve_cmudict_phones(word: str, pronunciation_index: int = 0) -> list[str]:
    """Resolve canonical phones from cmudict at runtime."""
    normalized_word = str(word or "").strip().lower()
    if not normalized_word:
        raise ValueError("Cannot resolve phones for an empty word.")

    try:
        import cmudict
    except ImportError as exc:
        raise ImportError("cmudict is required. Install with: pip install cmudict") from exc

    pronunciations = cmudict.dict().get(normalized_word)
    if not pronunciations:
        raise ValueError(f"Word '{normalized_word}' was not found in cmudict.")

    index = max(0, min(int(pronunciation_index), len(pronunciations) - 1))
    phones = [strip_cmudict_stress(phone) for phone in pronunciations[index]]
    phones = [phone for phone in phones if phone]
    if not phones:
        raise ValueError(f"cmudict returned an empty pronunciation for word '{normalized_word}'.")
    return phones


def resolve_preview_phones(word: str) -> list[str]:
    """Use cmudict when installed; fall back to the fixed preview lexicon."""
    normalized = str(word).strip().lower()
    try:
        return resolve_cmudict_phones(normalized)
    except (ImportError, ValueError):
        if normalized in PREVIEW_LEXICON:
            return list(PREVIEW_LEXICON[normalized])
        raise


def _cmvn_distance_from_audio(reference_audio: np.ndarray, reference_sr: int, candidate_audio: np.ndarray, candidate_sr: int) -> float:
    """Score two in-memory audio snippets using CMVN-normalized MFCC DTW."""
    ref_mfcc = _cmvn(_extract_mfcc(reference_audio, reference_sr))
    cand_mfcc = _cmvn(_extract_mfcc(candidate_audio, candidate_sr))
    if ref_mfcc.size == 0 or cand_mfcc.size == 0:
        return float("inf")
    if CMVN_DTW_MAX_FRAMES > 0:
        ref_mfcc = _resample_time_axis(ref_mfcc, CMVN_DTW_MAX_FRAMES)
        cand_mfcc = _resample_time_axis(cand_mfcc, CMVN_DTW_MAX_FRAMES)
    return _dtw_distance(ref_mfcc, cand_mfcc)


def _slice_phone_audio(audio: np.ndarray, phone_index: int, phone_count: int) -> np.ndarray:
    """Approximate a phone segment by equally partitioning the word audio."""
    phone_count = max(1, int(phone_count))
    start = int(round((len(audio) * phone_index) / phone_count))
    end = int(round((len(audio) * (phone_index + 1)) / phone_count))
    if end <= start:
        end = min(len(audio), start + 1)
    return audio[start:end].astype(np.float32)


def asset_audio_path(asset: dict[str, Any]) -> str | None:
    return asset.get("audio_path") or asset.get("wav_path") or asset.get("output_audio_path")


def asset_phone_key(asset: dict[str, Any]) -> tuple[str, ...]:
    return tuple(str(phone) for phone in (asset.get("phones") or []))


def phoneme_segment_cmvn_distances(reference_audio_path: str | Path, candidate_audio_path: str | Path, phone_count: int) -> list[float]:
    """
    Estimate per-phone CMVN distances by comparing equal-duration word segments.

    Existing assets do not store phone-level alignments, so this intentionally
    uses deterministic equal segmentation.
    """
    ref_sr, ref_audio = _load_audio_mono(reference_audio_path)
    cand_sr, cand_audio = _load_audio_mono(candidate_audio_path)
    distances: list[float] = []
    for idx in range(max(1, int(phone_count))):
        ref_segment = _slice_phone_audio(ref_audio, idx, phone_count)
        cand_segment = _slice_phone_audio(cand_audio, idx, phone_count)
        distances.append(_cmvn_distance_from_audio(ref_segment, ref_sr, cand_segment, cand_sr))
    return distances


def per_phoneme_eeg_targets(comparison: dict[str, Any], phone_count: int) -> list[float]:
    """Map EEG mismatch over time to one normalized target per canonical phone."""
    target_profile = _normalized_time_distance_profile(comparison)
    return _resample_profile(target_profile, max(1, int(phone_count)))


def preview_targets_for_phone_count(signal: list[float], phone_count: int) -> list[float]:
    """Map a preview distance signal directly to phone slots."""
    values = [clamp_unit(value) for value in signal]
    if not values:
        return [0.0] * max(1, int(phone_count))

    phone_count = max(1, int(phone_count))
    if phone_count == 1:
        return [sum(values) / len(values)]
    if phone_count == 2:
        mid = max(1, len(values) // 2)
        first = values[:mid]
        second = values[mid:] or values[-1:]
        return [sum(first) / len(first), sum(second) / len(second)]
    if phone_count == 3:
        if len(values) == 1:
            return [values[0], values[0], values[0]]
        first = values[:2]
        middle_start = max(0, (len(values) // 2) - 1)
        middle = values[middle_start:middle_start + 2]
        last = values[-2:]
        return [
            sum(first) / len(first),
            sum(middle) / len(middle),
            sum(last) / len(last),
        ]
    if phone_count == len(values):
        return values
    return _resample_profile(values, phone_count)


def phone_options_for_target(phone: str, eeg_target: float, options_per_slot: int = PHONEME_OPTIONS_PER_SLOT) -> list[tuple[str, float, str]]:
    """Return same-slot phone substitutions whose articulatory distance is within target."""
    target = clamp_unit(eeg_target)
    options: list[tuple[str, float, str]] = [(phone, 0.0, f"canonical:{phone}")]
    for candidate, _declared_distance, label in variants_for_phone(phone, target):
        distance = float(articulatory_distance(phone, candidate))
        if distance <= target + 1e-9:
            options.append((candidate, distance, label))
    for candidate in PHONE_FEATURES:
        distance = float(articulatory_distance(phone, candidate))
        if distance <= target + 1e-9:
            options.append((candidate, distance, f"distance:{phone}->{candidate}"))

    dedup: dict[str, tuple[str, float, str]] = {}
    for candidate, distance, label in options:
        current = dedup.get(candidate)
        if current is None or distance > current[1]:
            dedup[candidate] = (candidate, distance, label)
    return sorted(dedup.values(), key=lambda item: (-item[1], item[0]))[:max(1, int(options_per_slot))]


def generate_dynamic_phone_sequences(
    canonical: list[str],
    eeg_targets: list[float],
    max_candidates: int = PHONEME_DYNAMIC_MAX_CANDIDATES,
) -> list[dict[str, Any]]:
    """Generate same-length phone candidates from per-phone EEG thresholds."""
    per_slot_options = [
        phone_options_for_target(phone, eeg_targets[idx] if idx < len(eeg_targets) else 0.0)
        for idx, phone in enumerate(canonical)
    ]

    generated: list[dict[str, Any]] = []
    for combo in itertools.product(*per_slot_options):
        phones = [item[0] for item in combo]
        phone_distances = [float(item[1]) for item in combo]
        operations = [item[2] for item in combo if item[1] > 0.0]
        generated.append(
            {
                "phones": phones,
                "phone_distances": phone_distances,
                "operations": operations,
                "mean_phone_distance": sum(phone_distances) / max(1, len(phone_distances)),
            }
        )

    generated.sort(key=lambda row: (-row["mean_phone_distance"], " ".join(row["phones"])))
    return generated[:max(1, int(max_candidates))]


def index_assets_by_phones(assets: list[dict[str, Any]]) -> dict[tuple[str, ...], list[dict[str, Any]]]:
    indexed: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for asset in assets:
        key = asset_phone_key(asset)
        if key:
            indexed.setdefault(key, []).append(asset)
    return indexed


def best_asset_for_generated_phones(
    assets_by_phones: dict[tuple[str, ...], list[dict[str, Any]]],
    phones: list[str],
    voice_mode: str,
) -> dict[str, Any] | None:
    matches = list(assets_by_phones.get(tuple(phones), []))
    if not matches:
        return None
    voice_matches = []
    for asset in matches:
        try:
            asset_voice = normalize_voice_mode(asset.get("voice_mode") or asset.get("gender") or voice_mode)
        except ValueError:
            continue
        if asset_voice == voice_mode:
            voice_matches.append(asset)
    pool = voice_matches or matches
    pool.sort(key=lambda asset: (-float(asset.get("suitability_score") or 0.0), str(asset.get("asset_id") or "")))
    return pool[0]


def synthetic_asset(word: str, phones: list[str], voice_mode: str, role: str) -> dict[str, Any]:
    """Create metadata for a phone sequence when no rendered asset exists."""
    phone_key = " ".join(phones)
    return {
        "asset_id": f"{word}__synthetic_{role}__{phone_key.replace(' ', '_')}",
        "asset_role": role,
        "is_ground_truth": role == "ground_truth",
        "word": word,
        "target_word": word,
        "candidate_text": word if role == "ground_truth" else phone_key,
        "phones": list(phones),
        "canonical_phones": list(phones) if role == "ground_truth" else None,
        "phone_key": phone_key,
        "voice_mode": voice_mode,
        "audio_path": None,
        "synthetic_asset": True,
    }


def find_cmvn_reference_asset(assets: list[dict[str, Any]], canonical: list[str], voice_mode: str) -> dict[str, Any] | None:
    assets_by_phones = index_assets_by_phones(assets)
    exact = best_asset_for_generated_phones(assets_by_phones, canonical, voice_mode)
    if exact is not None:
        return exact
    for asset in assets:
        phones = list(asset.get("phones") or [])
        canonical_phones = list(asset.get("canonical_phones") or canonical)
        if bool(asset.get("is_ground_truth")) or phones == canonical_phones:
            return asset
    return None


def default_phone_cmvn_distance(
    *,
    canonical_phone: str,
    generated_phone: str,
    phone_index: int,
    phone_count: int,
    reference_asset: dict[str, Any],
    candidate_asset: dict[str, Any],
) -> float:
    reference_audio = asset_audio_path(reference_asset)
    candidate_audio = asset_audio_path(candidate_asset)
    if not reference_audio or not candidate_audio:
        return float(articulatory_distance(canonical_phone, generated_phone))
    return float(phoneme_segment_cmvn_distances(reference_audio, candidate_audio, phone_count)[phone_index])


def select_dynamic_phoneme_candidate(
    *,
    word: str,
    canonical: list[str],
    comparison: dict[str, Any],
    pairwise_distances: dict[str, Any],
    cmvn_phone_max_threshold: float = CMVN_PHONE_MAX_THRESHOLD,
) -> dict[str, Any] | None:
    """Generate and rank phoneme-by-phoneme candidates using EEG and CMVN limits."""
    eeg_targets = per_phoneme_eeg_targets(comparison, len(canonical))
    generated = generate_dynamic_phone_sequences(canonical, eeg_targets)
    invalid: list[dict[str, Any]] = []
    ranked: list[dict[str, Any]] = []
    max_threshold = float(cmvn_phone_max_threshold)

    for candidate in generated:
        phones = list(candidate["phones"])
        cmvn_distances: list[float] = []
        missing_pairs: list[dict[str, str]] = []
        for idx, (canonical_phone, generated_phone) in enumerate(zip(canonical, phones)):
            distance = pairwise_cmvn_distance(pairwise_distances, canonical_phone, generated_phone)
            if distance is None:
                missing_pairs.append(
                    {
                        "source": strip_cmudict_stress(canonical_phone),
                        "generated": strip_cmudict_stress(generated_phone),
                    }
                )
            else:
                cmvn_distances.append(distance)

        if missing_pairs:
            invalid.append(
                {
                    "phones": phones,
                    "reason": "missing_cmvn_pairwise_distance",
                    "missing_pairs": missing_pairs,
                }
            )
            continue

        if any(distance > max_threshold for distance in cmvn_distances):
            invalid.append(
                {
                    "phones": phones,
                    "reason": "cmvn_phone_threshold_exceeded",
                    "per_phoneme_cmvn_distances": cmvn_distances,
                    "cmvn_phone_max_threshold": max_threshold,
                }
            )
            continue

        phone_distances = [float(value) for value in candidate["phone_distances"]]
        eeg_fill = [
            1.0 if target <= 1e-9 and distance <= 1e-9 else distance / max(target, 1e-9)
            for distance, target in zip(phone_distances, eeg_targets)
        ]
        cmvn_margin = [1.0 - min(1.0, distance / max(max_threshold, 1e-9)) for distance in cmvn_distances]
        mean_phone_distance = sum(phone_distances) / max(1, len(phone_distances))
        mean_eeg_fill = sum(eeg_fill) / max(1, len(eeg_fill))
        mean_cmvn_margin = sum(cmvn_margin) / max(1, len(cmvn_margin))
        score = mean_phone_distance + PHONEME_EEG_WEIGHT * mean_eeg_fill + PHONEME_CMVN_WEIGHT * mean_cmvn_margin
        ranked.append(
            {
                "score": score,
                "generated_phones": phones,
                "canonical_phones": list(canonical),
                "per_phoneme_eeg_targets": eeg_targets,
                "per_phoneme_distances": phone_distances,
                "per_phoneme_cmvn_distances": cmvn_distances,
                "cmvn_phone_max_threshold": max_threshold,
                "operations": candidate["operations"],
                "score_details": {
                    "dynamic_phoneme_score": score,
                    "mean_phone_distance": mean_phone_distance,
                    "mean_eeg_fill": mean_eeg_fill,
                    "mean_cmvn_margin": mean_cmvn_margin,
                    "distance_metric": pairwise_distances.get("distance_metric", "cmvn_mfcc_dtw"),
                    "cmvn_pairwise_inventory_root": pairwise_distances.get("inventory_root"),
                },
            }
        )

    if not ranked:
        return {
            "valid": False,
            "word": word,
            "canonical_phones": list(canonical),
            "generated_phones": list(canonical),
            "per_phoneme_eeg_targets": eeg_targets,
            "invalid_phoneme_candidates": invalid[:TOP_K_DEBUG],
        }

    ranked.sort(key=lambda row: (-row["score_details"]["mean_phone_distance"], -row["score"], " ".join(row["generated_phones"])))
    best = ranked[0]
    return {
        "valid": True,
        "word": word,
        "canonical_phones": list(canonical),
        "generated_phones": best["generated_phones"],
        "phoneme_map": [
            {
                "source": source,
                "generated": generated_phone,
                "eeg_target": eeg_targets[idx],
                "phone_distance": best["per_phoneme_distances"][idx],
                "cmvn_distance": best["per_phoneme_cmvn_distances"][idx],
            }
            for idx, (source, generated_phone) in enumerate(zip(canonical, best["generated_phones"]))
        ],
        "per_phoneme_eeg_targets": eeg_targets,
        "per_phoneme_distances": best["per_phoneme_distances"],
        "per_phoneme_cmvn_distances": best["per_phoneme_cmvn_distances"],
        "cmvn_phone_max_threshold": max_threshold,
        "score": best["score"],
        "score_details": best["score_details"],
        "asset": synthetic_asset(word, best["generated_phones"], "phone_inventory", "candidate"),
        "reference_asset": synthetic_asset(word, canonical, "phone_inventory", "ground_truth"),
        "cmvn_pairwise_distances_path": cmvn_pairwise_cache_path(Path(str(pairwise_distances.get("inventory_root") or DEFAULT_PHONE_INVENTORY_ROOT))),
        "invalid_phoneme_candidates": invalid[:TOP_K_DEBUG],
        "top_candidates": [
            {
                "generated_phones": row["generated_phones"],
                "score": row["score"],
                "score_details": row["score_details"],
                "per_phoneme_cmvn_distances": row["per_phoneme_cmvn_distances"],
            }
            for row in ranked[:TOP_K_DEBUG]
        ],
    }


def parse_distance_signal(value: str) -> list[float]:
    """Parse a comma-separated preview distance signal."""
    parts = [part.strip() for part in str(value).split(",") if part.strip()]
    if not parts:
        raise ValueError("Preview distance signal cannot be empty.")
    return [float(part) for part in parts]


def preview_rows(
    *,
    words: tuple[str, ...] = PREVIEW_WORDS,
    distance_signal: list[float] | None = None,
    cmvn_phone_max_threshold: float = CMVN_PHONE_MAX_THRESHOLD,
) -> list[dict[str, Any]]:
    """Generate preview rows for the built-in vocab without requiring assets."""
    signal = distance_signal if distance_signal is not None else [0.1, 0.3, 0.6, 0.9]
    rows: list[dict[str, Any]] = []
    for word in words:
        canonical = resolve_preview_phones(word)
        targets = preview_targets_for_phone_count(signal, len(canonical))
        generated_candidates = generate_dynamic_phone_sequences(canonical, targets)
        ranked = []
        for candidate in generated_candidates:
            distances = [articulatory_distance(src, dst) for src, dst in zip(canonical, candidate["phones"])]
            if any(distance > cmvn_phone_max_threshold for distance in distances):
                continue
            fill = [
                1.0 if target <= 1e-9 and distance <= 1e-9 else distance / max(target, 1e-9)
                for distance, target in zip(distances, targets)
            ]
            ranked.append((sum(distances) / len(distances), sum(fill) / len(fill), candidate["phones"], distances))
        ranked.sort(key=lambda item: (-item[0], -item[1], " ".join(item[2])))
        if ranked:
            _mean_distance, _mean_fill, generated, distances = ranked[0]
        else:
            generated = list(canonical)
            distances = [0.0] * len(canonical)
        phoneme_map = [
            {
                "source": source,
                "generated": generated_phone,
                "target": targets[idx],
                "distance": distances[idx],
            }
            for idx, (source, generated_phone) in enumerate(zip(canonical, generated))
        ]
        rows.append(
            {
                "word": word,
                "canonical_phones": canonical,
                "preview_targets": targets,
                "generated_phones": generated,
                "phoneme_map": phoneme_map,
            }
        )
    return rows


def print_preview(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        canonical = " ".join(row["canonical_phones"])
        generated = " ".join(row["generated_phones"])
        targets = ",".join(f"{value:.3f}" for value in row["preview_targets"])
        print(f"{row['word']}: {canonical} -> {generated}  targets={targets}")


def granularity_values(granularity: int) -> list[float]:
    """Return exactly `granularity` distance values from 0.0 to 1.0 inclusive."""
    if int(granularity) < 2:
        raise ValueError("--granularity must be at least 2.")
    count = int(granularity)
    return [i / float(count - 1) for i in range(count)]


def distance_label(value: float) -> str:
    """Format a distance value for stable filename keys."""
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def distance_key(values: tuple[float, ...] | list[float]) -> str:
    return "_".join(distance_label(value) for value in values)


def phone_inventory_names(phone: str) -> list[str]:
    """Return possible inventory basenames for an ARPABET phone."""
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
    """Find the first matching isolated-phone MP3/WAV asset for a phone."""
    for basename in phone_inventory_names(phone):
        for suffix in (".mp3", ".wav"):
            path = inventory_root / f"{basename}{suffix}"
            if path.is_file():
                return path
    return None


def resolve_phone_inventory_assets(phones: list[str], inventory_root: Path) -> tuple[list[Path], list[str]]:
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
        raise ValueError(f"Expected 16-bit PCM WAV after ffmpeg decode, got sample width {sample_width}: {path}")
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
    """Small dependency-free mono resampler for phone inventory clips."""
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
    """Decode one phone asset to mono float audio."""
    if path.suffix.lower() == ".wav":
        source_rate, audio = _load_audio_mono(path)
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


def cmvn_pairwise_cache_path(inventory_root: Path) -> Path:
    return inventory_root / "cmvn_pairwise_distances.json"


def available_inventory_phone_assets(inventory_root: Path) -> dict[str, Path]:
    """Return ARPABET phones that can be resolved to inventory audio assets."""
    phones = sorted(set(ARPABET_TO_AUS_IPA_CANDIDATES) | set(PHONE_FEATURES))
    assets: dict[str, Path] = {}
    for phone in phones:
        path = resolve_phone_inventory_asset(phone, inventory_root)
        if path is not None:
            assets[phone] = path
    return assets


def _cmvn_features_from_phone_asset(path: Path) -> np.ndarray:
    sample_rate, audio = decode_phone_asset_to_audio(path, sample_rate=24000)
    features = _cmvn(_extract_mfcc(audio, sample_rate))
    if CMVN_DTW_MAX_FRAMES > 0 and features.size:
        features = _resample_time_axis(features, CMVN_DTW_MAX_FRAMES)
    return features


def build_cmvn_pairwise_distances(inventory_root: Path) -> dict[str, Any]:
    """Build a pairwise CMVN-DTW matrix for all resolvable inventory phones."""
    phone_assets = available_inventory_phone_assets(inventory_root)
    if not phone_assets:
        raise FileNotFoundError(f"No phone assets found in inventory root: {inventory_root}")

    phone_features: dict[str, np.ndarray] = {}
    skipped: dict[str, str] = {}
    for phone, path in phone_assets.items():
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


def normalize_pairwise_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Add normalized [0,1] distances while preserving raw CMVN distances."""
    raw_distances = payload.get("raw_distances") or payload.get("distances") or {}
    max_distance = 0.0
    for row in raw_distances.values():
        for value in row.values():
            max_distance = max(max_distance, float(value))

    normalized: dict[str, dict[str, float]] = {}
    for source, row in raw_distances.items():
        normalized[source] = {}
        for target, value in row.items():
            normalized[source][target] = 0.0 if max_distance <= 0.0 else float(value) / max_distance

    payload["raw_distances"] = raw_distances
    payload["distances"] = normalized
    payload["raw_distance_max"] = max_distance
    payload["distance_scale"] = "normalized_0_1"
    return payload


def load_or_create_cmvn_pairwise_distances(inventory_root: Path, *, rebuild: bool = False) -> dict[str, Any]:
    """Load the CMVN phone-distance cache, creating it when absent."""
    cache_path = cmvn_pairwise_cache_path(inventory_root)
    if cache_path.is_file() and not rebuild:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        if payload.get("distance_scale") != "normalized_0_1" or "raw_distances" not in payload:
            payload = normalize_pairwise_payload(payload)
            cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    payload = build_cmvn_pairwise_distances(inventory_root)
    inventory_root.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def pairwise_cmvn_distance(pairwise: dict[str, Any], source_phone: str, generated_phone: str) -> float | None:
    """Read a pairwise CMVN distance from the cache using normalized ARPABET keys."""
    source = strip_cmudict_stress(source_phone)
    generated = strip_cmudict_stress(generated_phone)
    distances = pairwise.get("distances") or {}
    value = (distances.get(source) or {}).get(generated)
    if value is None:
        return None
    return float(value)


def crossfade_concat(segments: list[np.ndarray], *, sample_rate: int, gap_ms: float, crossfade_ms: float) -> np.ndarray:
    """Concatenate phone clips with a small gap/crossfade to reduce clicks."""
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
            if gap_samples:
                output = np.concatenate([output, np.zeros(gap_samples, dtype=np.float32), segment])
            else:
                output = np.concatenate([output, segment])
    return output.astype(np.float32)


def stitch_phone_assets(
    *,
    phones: list[str],
    source_paths: list[Path],
    output_wav: Path,
    overwrite: bool,
    gap_ms: float,
    crossfade_ms: float,
    sample_rate: int = 24000,
) -> dict[str, Any]:
    """Create one word WAV by stitching isolated phone assets."""
    if output_wav.exists() and not overwrite:
        return {
            "status": "exists",
            "source_phone_assets": [str(path) for path in source_paths],
        }

    decoded_segments = [decode_phone_asset_to_audio(path, sample_rate=sample_rate)[1] for path in source_paths]
    audio = crossfade_concat(decoded_segments, sample_rate=sample_rate, gap_ms=gap_ms, crossfade_ms=crossfade_ms)
    _write_wav_int16(output_wav, sample_rate, audio)
    return {
        "status": "ok",
        "source_phone_assets": [str(path) for path in source_paths],
        "duration_sec": float(len(audio) / sample_rate) if sample_rate else 0.0,
    }


def select_phones_for_distance_tuple(canonical: list[str], distances: tuple[float, ...] | list[float]) -> tuple[list[str], list[float], list[str]]:
    """Select the largest valid phone option for each requested distance."""
    phones: list[str] = []
    actual_distances: list[float] = []
    operations: list[str] = []
    for canonical_phone, target in zip(canonical, distances):
        options = phone_options_for_target(canonical_phone, float(target))
        selected_phone, selected_distance, operation = options[0]
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
    """Parse comma-separated integers and inclusive ranges such as 1,3-5."""
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


def render_args_for_voice(args: argparse.Namespace, voice_mode: str) -> SimpleNamespace:
    """Build the renderer arg shape expected by deep_phone_candidate_stack_blabber_triphone."""
    edge_voice_mode = "female" if voice_mode == "woman" else "male"
    return SimpleNamespace(
        tts_engine=args.tts_engine,
        kokoro_lang_code=args.kokoro_lang_code,
        kokoro_speed=args.kokoro_speed,
        edge_voice=args.edge_voice,
        edge_voice_mode=edge_voice_mode,
        edge_rate=args.edge_rate,
        edge_pitch=args.edge_pitch,
        piper_model=args.piper_model,
        piper_config=args.piper_config,
        piper_exe=args.piper_exe,
        piper_speaker=args.piper_speaker,
        piper_use_python_module=args.piper_use_python_module,
        piper_length_scale=args.piper_length_scale,
        piper_noise_scale=args.piper_noise_scale,
        piper_noise_w=args.piper_noise_w,
        piper_sentence_silence=args.piper_sentence_silence,
        coqui_model=args.coqui_model,
        coqui_speaker=args.coqui_speaker,
        coqui_language=args.coqui_language,
        speaker_wav=args.speaker_wav,
    )


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
) -> dict[str, Any]:
    key = distance_key(requested_distances)
    render_text = " ".join(generated)
    return {
        "schema_version": "1.0-per-phoneme-grid",
        "asset_id": f"{word}__{voice_mode}__{key}",
        "word": word,
        "target_word": word,
        "voice_mode": voice_mode,
        "generation_mode": "per_phoneme_grid",
        "candidate_text": render_text,
        "render_text": render_text,
        "phones": list(generated),
        "canonical_phones": list(canonical),
        "distance_tuple": [float(value) for value in requested_distances],
        "actual_phone_distances": list(actual_distances),
        "distance_key": key,
        "operations": list(operations),
        "audio_path": str(wav_path),
        "wav_path": str(wav_path),
        "output_audio_path": str(wav_path),
        "synthetic_asset": False,
    }


def pregenerate_word_assets(
    *,
    word: str,
    voice_mode: str,
    output_root: Path,
    distance_values: list[float],
    render_audio: bool,
    overwrite: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """Generate one word/voice per-phoneme asset grid."""
    canonical = resolve_preview_phones(word)
    word_dir = output_root / voice_mode / word
    word_dir.mkdir(parents=True, exist_ok=True)
    render_source = str(getattr(args, "render_audio_source", "tts") or "tts")
    phone_inventory_root = Path(getattr(args, "phone_inventory_root", DEFAULT_PHONE_INVENTORY_ROOT))

    renderer = None
    renderer_args = None
    if render_audio and render_source == "tts":
        renderer_args = render_args_for_voice(args, voice_mode)
        renderer = make_renderer(renderer_args)

    assets: list[dict[str, Any]] = []
    skipped_missing_phone = 0
    skipped_missing_phone_details: list[dict[str, Any]] = []
    stitched_count = 0
    existing_count = 0
    for requested_distances in itertools.product(distance_values, repeat=len(canonical)):
        generated, actual_distances, operations = select_phones_for_distance_tuple(canonical, requested_distances)
        key = distance_key(requested_distances)
        wav_path = word_dir / f"{key}.wav"
        source_phone_assets: list[Path] = []
        if render_audio and render_source == "phone-stitch":
            source_phone_assets, missing_phones = resolve_phone_inventory_assets(generated, phone_inventory_root)
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

        asset = make_per_phoneme_asset(
            word=word,
            voice_mode=voice_mode,
            canonical=canonical,
            generated=generated,
            requested_distances=requested_distances,
            actual_distances=actual_distances,
            operations=operations,
            wav_path=wav_path,
        )
        if render_audio and render_source == "phone-stitch":
            stitch_info = stitch_phone_assets(
                phones=generated,
                source_paths=source_phone_assets,
                output_wav=wav_path,
                overwrite=overwrite,
                gap_ms=args.phone_gap_ms,
                crossfade_ms=args.phone_crossfade_ms,
            )
            asset["audio_source"] = "phone_stitch"
            asset["phone_inventory_root"] = str(phone_inventory_root)
            asset["source_phone_assets"] = stitch_info["source_phone_assets"]
            asset["stitch_gap_ms"] = float(args.phone_gap_ms)
            asset["stitch_crossfade_ms"] = float(args.phone_crossfade_ms)
            if stitch_info["status"] == "exists":
                existing_count += 1
            else:
                stitched_count += 1
            if "duration_sec" in stitch_info:
                asset["stitched_duration_sec"] = stitch_info["duration_sec"]
            if wav_path.exists():
                ok_duration, duration_info = validate_rendered_duration(
                    wav_path,
                    generated,
                    max_duration_sec=args.max_duration_sec,
                    max_duration_ratio=args.max_duration_ratio,
                )
                asset["duration_info"] = duration_info
                if not ok_duration and not args.allow_overlong:
                    asset["render_status"] = str(duration_info["status"])
                else:
                    asset["render_status"] = f"stitched_{stitch_info['status']}"
            else:
                asset["render_status"] = "stitch_missing_output"
        elif render_audio and renderer is not None and renderer_args is not None:
            if overwrite or not wav_path.exists():
                render_candidate(renderer, renderer_args.tts_engine, asset["render_text"], wav_path, voice=None)
            if wav_path.exists():
                ok_duration, duration_info = validate_rendered_duration(
                    wav_path,
                    generated,
                    max_duration_sec=args.max_duration_sec,
                    max_duration_ratio=args.max_duration_ratio,
                )
                asset["duration_info"] = duration_info
                if not ok_duration and not args.allow_overlong:
                    asset["render_status"] = str(duration_info["status"])
                else:
                    asset["render_status"] = "ok"
        else:
            asset["render_status"] = "metadata_only"
        assets.append(asset)

    skipped_requested = skipped_missing_phone

    payload = {
        "schema_version": "1.0-per-phoneme-grid",
        "generation_mode": "per_phoneme_grid",
        "word": word,
        "voice_mode": voice_mode,
        "canonical_phones": canonical,
        "granularity": len(distance_values),
        "distance_values": list(distance_values),
        "asset_count": len(assets),
        "render_audio_source": render_source,
        "skipped_missing_phone_count": skipped_requested,
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
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload


def pregenerate_assets(args: argparse.Namespace) -> list[dict[str, Any]]:
    values = granularity_values(args.granularity)
    output_root = Path(args.output_root)
    payloads: list[dict[str, Any]] = []
    for voice_mode in parse_voices(args.voices):
        for word in parse_words(args.words):
            payloads.append(
                pregenerate_word_assets(
                    word=word,
                    voice_mode=voice_mode,
                    output_root=output_root,
                    distance_values=values,
                    render_audio=bool(args.render_audio and not args.metadata_only),
                    overwrite=bool(args.overwrite),
                    args=args,
                )
            )
    return payloads


def per_phoneme_library_json_path(output_root: str | Path, voice_mode: str, word: str) -> Path:
    """Resolve the pregenerated per-phoneme library JSON for one voice/word."""
    return Path(output_root) / normalize_voice_mode(voice_mode) / word.strip().lower() / f"{word.strip().lower()}_per_phoneme_assets.json"


def load_per_phoneme_assets(path: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    assets = payload.get("assets")
    if not isinstance(assets, list) and isinstance(payload.get("library"), dict):
        assets = payload["library"].get("assets")
    if not isinstance(assets, list):
        raise ValueError(f"Per-phoneme library does not contain an assets list: {path}")
    return assets, payload


def asset_audio_exists(asset: dict[str, Any]) -> bool:
    audio = asset_audio_path(asset)
    return bool(audio) and Path(str(audio)).is_file()


def per_phoneme_library_needs_generation(path: Path, *, require_audio: bool) -> bool:
    if not path.is_file():
        return True
    if not require_audio:
        return False
    try:
        assets, _payload = load_per_phoneme_assets(path)
    except Exception:
        return True
    return not assets or any(not asset_audio_exists(asset) for asset in assets)


def ensure_per_phoneme_libraries(args: argparse.Namespace) -> list[dict[str, Any]]:
    """Generate the full configured vocab/voice asset grid when any library is missing."""
    output_root = Path(args.output_root)
    require_audio = bool(args.render_audio and not args.metadata_only)
    missing = [
        per_phoneme_library_json_path(output_root, voice, word)
        for voice in parse_voices(args.voices)
        for word in parse_words(args.words)
        if per_phoneme_library_needs_generation(
            per_phoneme_library_json_path(output_root, voice, word),
            require_audio=require_audio,
        )
    ]
    if not missing:
        return []
    print(
        f"Generating per-phoneme libraries for {len(parse_voices(args.voices))} voices "
        f"and {len(parse_words(args.words))} words under {output_root}"
    )
    return pregenerate_assets(args)


def ensure_cmvn_pairwise_cache(args: argparse.Namespace) -> dict[str, Any]:
    return load_or_create_cmvn_pairwise_distances(
        Path(args.phone_inventory_root),
        rebuild=bool(args.rebuild_cmvn_pairwise),
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
        return preview_targets_for_phone_count(parse_distance_signal(preview_signal), phone_count), "preview_signal"
    if comparison is None:
        raise ValueError("comparison is required when --preview-signal is not provided")
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
        distance_tuple = [float(v) for v in asset.get("distance_tuple") or []]
        actual_distances = [float(v) for v in asset.get("actual_phone_distances") or []]
        if len(distance_tuple) != len(target_signal):
            continue
        tuple_mse = vector_mse(distance_tuple, target_signal)
        actual_mse = vector_mse(actual_distances, target_signal)
        ranked.append(
            {
                "asset": asset,
                "score": -tuple_mse,
                "score_details": {
                    "selection_metric": "distance_tuple_mse",
                    "distance_tuple_mse": tuple_mse,
                    "actual_phone_distance_mse": actual_mse,
                    "target_signal_source": target_source,
                },
            }
        )

    if not ranked:
        raise ValueError(f"No per-phoneme assets matched word='{word}' voice='{voice_mode}' and target length={len(target_signal)}.")

    ranked.sort(
        key=lambda row: (
            row["score_details"]["distance_tuple_mse"],
            row["score_details"]["actual_phone_distance_mse"],
            str(row["asset"].get("asset_id") or ""),
        )
    )
    best = ranked[0]
    asset = best["asset"]
    return {
        "word": word,
        "voice_mode": voice_mode,
        "generation_mode": "per_phoneme_grid",
        "library_json": str(library_path),
        "audio_path": asset_audio_path(asset),
        "asset_id": asset.get("asset_id"),
        "target_signal": target_signal,
        "target_signal_source": target_source,
        "requested_generation_signature": target_signal,
        "selected_phoneme_grades": asset.get("distance_tuple"),
        "actual_phone_distances": asset.get("actual_phone_distances"),
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
            for row in ranked[:max(1, int(top_k))]
        ],
        "times": list(comparison.get("times", [])) if comparison else [],
        "per_time_l2": list(comparison.get("per_time_l2", [])) if comparison else [],
    }


def build_default_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Mirror my_test.py: build a template, load one probe trial, then compare."""
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
    parser = argparse.ArgumentParser(description="Run cmudict per-phoneme EEG/CMVN candidate selection.")
    parser.add_argument("--preview", action="store_true", help="Preview all built-in vocab words with a synthetic distance signal.")
    parser.add_argument("--pregenerate-assets", action="store_true", help="Generate a full per-phoneme asset grid for the vocab.")
    parser.add_argument(
        "--preview-signal",
        default=None,
        help="Comma-separated direct distance signal. In normal selection mode, --word is required when this is provided.",
    )
    parser.add_argument("--granularity", type=int, default=2, help="Number of distance options per phoneme for --pregenerate-assets.")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT, help="Parent directory for per-phoneme asset libraries.")
    parser.add_argument("--voices", default="man,woman", help="Comma-separated voice modes for --pregenerate-assets.")
    parser.add_argument("--words", default=",".join(PREVIEW_WORDS), help="Comma-separated words for --pregenerate-assets.")
    parser.add_argument("--render-audio", action="store_true", help="Render WAVs while pre-generating assets.")
    parser.add_argument("--render-audio-source",
                        choices=["tts", "phone-stitch"],
                        default="phone-stitch",
                        help="Audio source for --render-audio during "\
                        "--pregenerate-assets.")
    parser.add_argument("--phone-inventory-root", 
                        default=DEFAULT_PHONE_INVENTORY_ROOT,
                        help="Flat MP3/WAV isolated-phone inventory for "\
                        "--render-audio-source phone-stitch.")
    parser.add_argument("--phone-gap-ms", type=float, default=20.0, help="Silence gap between stitched phone assets.")
    parser.add_argument("--phone-crossfade-ms", type=float, default=5.0, help="Crossfade between stitched phone assets.")
    parser.add_argument("--metadata-only", action="store_true", help="Write JSON metadata and WAV paths without rendering audio.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing WAVs during --render-audio.")
    parser.add_argument("--word", default=None)
    parser.add_argument("--comparison-json", default=None, help="JSON file containing at least per_time_l2 and optionally times.")
    parser.add_argument("--requested-voice", default=DEFAULT_REQUESTED_VOICE, choices=["female", "male", "woman", "man"], help="Requested per-phoneme voice bank.")
    parser.add_argument("--rebuild-cmvn-pairwise", action="store_true", help="Recompute the phone-inventory CMVN pairwise cache.")
    parser.add_argument("--json-out", default=None)
    parser.add_argument("--cmvn-phone-max-threshold", type=float, default=CMVN_PHONE_MAX_THRESHOLD)
    parser.add_argument("--template-days", default="1", help="Comma-separated days/ranges for template building, e.g. 1 or 1-3.")
    parser.add_argument("--template-sessions", default="1-2", help="Comma-separated sessions/ranges for template building.")
    parser.add_argument("--probe-day", type=int, default=1)
    parser.add_argument("--probe-session", type=int, default=8)
    parser.add_argument("--label", type=int, default=4)
    parser.add_argument("--trial-index", type=int, default=0)
    parser.add_argument("--sr", type=int, default=250)
    parser.add_argument("--nperseg", type=int, default=128)
    parser.add_argument("--noverlap", type=int, default=96)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--fmax", type=float, default=50)
    parser.add_argument("--tts-engine", choices=["kokoro", "edge", "piper", "coqui"], default="coqui")
    parser.add_argument("--kokoro-lang-code", type=str, default="b")
    parser.add_argument("--kokoro-speed", type=float, default=1.0)
    parser.add_argument("--edge-voice", type=str, default=None)
    parser.add_argument("--edge-rate", type=str, default="+0%")
    parser.add_argument("--edge-pitch", type=str, default="+0Hz")
    parser.add_argument("--piper-exe", type=str, default="piper")
    parser.add_argument("--piper-use-python-module", action="store_true")
    parser.add_argument("--piper-model", type=str, default="piper-models/en_GB-jenny_dioco-medium.onnx")
    parser.add_argument("--piper-config", type=str, default="piper-models/en_GB-jenny_dioco-medium.onnx.json")
    parser.add_argument("--piper-speaker", type=int, default=None)
    parser.add_argument("--piper-length-scale", type=float, default=1.2)
    parser.add_argument("--piper-noise-scale", type=float, default=None)
    parser.add_argument("--piper-noise-w", type=float, default=None)
    parser.add_argument("--piper-sentence-silence", type=float, default=None)
    parser.add_argument("--coqui-model", type=str, default="tts_models/en/ljspeech/tacotron2-DDC")
    parser.add_argument("--coqui-speaker", type=str, default=None)
    parser.add_argument("--coqui-language", type=str, default=None)
    parser.add_argument("--speaker-wav", type=str, default=None)
    parser.add_argument("--max-duration-sec", type=float, default=3.0)
    parser.add_argument("--max-duration-ratio", type=float, default=3.0)
    parser.add_argument("--allow-overlong", action="store_true")
    args = parser.parse_args()

    output_root = str(
        Path(args.output_root)/Path(f"granularity_{args.granularity}")
    )

    if args.preview:
        rows = preview_rows(
            distance_signal=parse_distance_signal(args.preview_signal or "0.1,0.3,0.6,0.9"),
            cmvn_phone_max_threshold=args.cmvn_phone_max_threshold,
        )
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        else:
            print_preview(rows)
        return

    if args.pregenerate_assets:
        payloads = pregenerate_assets(args)
        summary = {
            "output_root": output_root,
            "voice_count": len(parse_voices(args.voices)),
            "word_count": len(parse_words(args.words)),
            "payload_count": len(payloads),
            "asset_count": sum(int(payload["asset_count"]) for payload in payloads),
            "render_audio": bool(args.render_audio and not args.metadata_only),
            "render_audio_source": args.render_audio_source,
            "skipped_missing_phone_count": sum(int(payload.get("skipped_missing_phone_count") or 0) for payload in payloads),
            "stitched_count": sum(int(payload.get("stitched_count") or 0) for payload in payloads),
            "existing_audio_count": sum(int(payload.get("existing_audio_count") or 0) for payload in payloads),
        }
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
        else:
            print(json.dumps(summary, indent=2, default=str))
        return

    args.render_audio = bool(args.render_audio or not args.metadata_only)
    args.render_audio_source = "phone-stitch"

    pairwise_distances = ensure_cmvn_pairwise_cache(args)
    ensure_per_phoneme_libraries(args)

    comparison: dict[str, Any] | None = None
    if args.preview_signal is not None:
        if not args.word:
            parser.error("--word is required when --preview-signal is used for normal selection.")
        word = args.word.strip().lower()
    elif args.comparison_json:
        comparison = json.loads(Path(args.comparison_json).read_text(encoding="utf-8"))
        word = str(args.word or comparison.get("label_name") or "").strip().lower()
        if not word:
            parser.error("--comparison-json must include label_name, or pass --word.")
    else:
        comparison = build_default_comparison(args)
        word = str(args.word or comparison.get("label_name") or "").strip().lower()

    if word not in parse_words(args.words):
        raise ValueError(f"Selected word '{word}' is not included in --words={args.words!r}; cannot guarantee its library exists.")

    result = select_per_phoneme_asset_from_library(
        word=word,
        requested_voice=args.requested_voice,
        output_root=output_root,
        comparison=comparison,
        preview_signal=args.preview_signal,
        top_k=TOP_K_DEBUG,
    )
    result["cmvn_pairwise_distances_path"] = str(cmvn_pairwise_cache_path(Path(args.phone_inventory_root)))
    result["cmvn_pairwise_phone_count"] = len(pairwise_distances.get("phones") or [])

    payload = json.dumps(result, indent=2, default=str)
    if args.json_out:
        Path(args.json_out).write_text(payload, encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
