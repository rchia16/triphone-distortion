#!/usr/bin/env python3
"""
deep_phone_candidate_stack_blabber_triphone.py

Phone-level pseudoword candidate generator for blabber-library assets.

Level discrimination decides which transformations are eligible; min/max 
distortion decides whether the resulting candidate lands in the desired 
acoustic/phonological distance range.

Current defaults:
  TTS backend:
    piper

  Default Piper model/config:
    piper-models/en_GB-jenny_dioco-medium.onnx
    piper-models/en_GB-jenny_dioco-medium.onnx.json

  Alternate Piper model/config:
    piper-models/en_GB-northern_english_male-medium.onnx
    piper-models/en_GB-northern_english_male-medium.onnx.json

  Piper clarity default:
    --piper-length-scale 1.2

What this script generates:
  1. A backward-compatible raw `candidates` list.
  2. A blabber-library-compatible `assets` / `blabber_assets` list.
  3. A triphone search index usable by select_blabber_asset.py-style selectors.
  4. Optional rendered TTS audio for each candidate.
  5. Optional torchaudio embedding scores against a reference WAV.

Core pipeline:
  1. Start with a target word and canonical phone sequence.
     Supported words:
       no, yes, go, stop, pain, help, bath, food

  2. Generate candidate phone sequences using articulatory mutation rules:
       vowel shifts
       consonant substitutions
       optional glide / weak-vowel / rhotic insertions

  3. Convert each candidate phone sequence into compact pseudo-graphemes:
       N UH ER   -> nuher
       N AH W OW -> nuhwoh
       N Y ER OW -> nyeroh

  4. Preserve ground truth at every level by default:
       Y EH S -> yes
       N OW   -> no
       G OW   -> go

     This means each augmentation level contains the canonical word asset plus
     level-appropriate augmented assets. Disable with:
       --no-ground-truth-each-level

  5. Emit triphone-searchable blabber assets:
       phones:         ["N", "AH", "ER"]
       phone_key:      "N AH ER"
       triphone_keys:  ["<s>|N|AH", "N|AH|ER", "AH|ER|</s>"]
       biphone_keys:   ["<s>|N", "N|AH", "AH|ER", "ER|</s>"]

  6. Optionally render audio with Piper, Edge TTS, Kokoro, or Coqui.
     Piper is the default and is recommended for local batch pseudoword rendering.

  7. Optionally score rendered candidates with torchaudio SSL/ASR embeddings.
     Default bundle:
       HUBERT_LARGE

  8. Reject overlong TTS artifacts using duration guards:
       --max-duration-sec
       --max-duration-ratio

Output JSON shape:
  {
    "candidates": [...],
    "blabber_schema_version": "2.0-triphone",
    "blabber_assets": [...],
    "assets": [...],
    "search_index": {...},
    "library": {
      "schema_version": "2.0-triphone",
      "assets": [...],
      "search_index": {...}
    }
  }

Important asset fields:
  {
    "asset_id": "...",
    "word": "yes",
    "candidate_text": "yes",
    "phones": ["Y", "EH", "S"],
    "canonical_phones": ["Y", "EH", "S"],
    "phone_key": "Y EH S",
    "triphone_keys": ["<s>|Y|EH", "Y|EH|S", "EH|S|</s>"],
    "is_ground_truth": true,
    "asset_role": "ground_truth",
    "level": 1.0,
    "audio_path": "..."
  }

Examples:

Rule-only library generation:
  python deep_phone_candidate_stack_blabber_triphone.py \
    --word yes \
    --levels 8 \
    --skip-neural \
    --json-out yes_blabber_library.json

Render Piper audio without torchaudio scoring:
  python deep_phone_candidate_stack_blabber_triphone.py \
    --word yes \
    --levels 8 \
    --skip-neural \
    --render-audio \
    --render-dir candidates_yes_piper \
    --json-out yes_blabber_library.json

Render and score with torchaudio embeddings:
  python deep_phone_candidate_stack_blabber_triphone.py \
    --word yes \
    --levels 8 \
    --reference-wav aus-en_yes.wav \
    --render-dir candidates_yes_piper \
    --json-out yes_blabber_scored.json

Use the alternate Piper voice:
  python deep_phone_candidate_stack_blabber_triphone.py \
    --word yes \
    --levels 8 \
    --piper-model piper-models/en_GB-northern_english_male-medium.onnx \
    --piper-config piper-models/en_GB-northern_english_male-medium.onnx.json \
    --skip-neural \
    --render-audio

Render with Australian Edge TTS:
  python deep_phone_candidate_stack_blabber_triphone.py \
    --word yes \
    --levels 8 \
    --tts-engine edge \
    --edge-voice en-AU-NatashaNeural \
    --skip-neural \
    --render-audio

Adjust articulation length:
  --piper-length-scale 1.05   near-ground-truth / natural
  --piper-length-scale 1.15   clear default range
  --piper-length-scale 1.20   current default, clearer short words
  --piper-length-scale 1.30   stronger annunciation for max augmentation

Search generated assets with the triphone selector:
  python select_blabber_asset_triphone.py yes_blabber_library.json \
    --word yes \
    --phones Y EH S \
    --target-level 1.0 \
    --top-k 5
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import wave
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------
# Tiny built-in lexicon. Replace with CMUdict / g2p-en later.
# ---------------------------------------------------------------------

LEXICON: dict[str, list[str]] = {
    "no": ["N", "OW"],
    "yes": ["Y", "EH", "S"],
    "go": ["G", "OW"],
    "stop": ["S", "T", "AA", "P"],
    "pain": ["P", "EY", "N"],
    "help": ["HH", "EH", "L", "P"],
    "bath": ["B", "AE", "TH"],
    "food": ["F", "UW", "D"],
}


PHONE_FEATURES: dict[str, dict[str, Any]] = {
    "P": {"type": "consonant", "manner": "stop", "place": "bilabial", "voicing": "voiceless"},
    "B": {"type": "consonant", "manner": "stop", "place": "bilabial", "voicing": "voiced"},
    "T": {"type": "consonant", "manner": "stop", "place": "alveolar", "voicing": "voiceless"},
    "D": {"type": "consonant", "manner": "stop", "place": "alveolar", "voicing": "voiced"},
    "K": {"type": "consonant", "manner": "stop", "place": "velar", "voicing": "voiceless"},
    "G": {"type": "consonant", "manner": "stop", "place": "velar", "voicing": "voiced"},
    "S": {"type": "consonant", "manner": "fricative", "place": "alveolar", "voicing": "voiceless"},
    "Z": {"type": "consonant", "manner": "fricative", "place": "alveolar", "voicing": "voiced"},
    "SH": {"type": "consonant", "manner": "fricative", "place": "postalveolar", "voicing": "voiceless"},
    "TH": {"type": "consonant", "manner": "fricative", "place": "dental", "voicing": "voiceless"},
    "F": {"type": "consonant", "manner": "fricative", "place": "labiodental", "voicing": "voiceless"},
    "V": {"type": "consonant", "manner": "fricative", "place": "labiodental", "voicing": "voiced"},
    "HH": {"type": "consonant", "manner": "fricative", "place": "glottal", "voicing": "voiceless"},
    "N": {"type": "consonant", "manner": "nasal", "place": "alveolar", "voicing": "voiced"},
    "M": {"type": "consonant", "manner": "nasal", "place": "bilabial", "voicing": "voiced"},
    "NG": {"type": "consonant", "manner": "nasal", "place": "velar", "voicing": "voiced"},
    "L": {"type": "consonant", "manner": "liquid", "place": "alveolar", "voicing": "voiced"},
    "R": {"type": "consonant", "manner": "liquid", "place": "postalveolar", "voicing": "voiced"},
    "Y": {"type": "consonant", "manner": "glide", "place": "palatal", "voicing": "voiced"},
    "W": {"type": "consonant", "manner": "glide", "place": "labiovelar", "voicing": "voiced"},
    "OW": {"type": "vowel", "height": "mid", "backness": "back", "rounding": "rounded", "diphthong": True},
    "UW": {"type": "vowel", "height": "high", "backness": "back", "rounding": "rounded", "diphthong": False},
    "UH": {"type": "vowel", "height": "high_mid", "backness": "back", "rounding": "rounded", "diphthong": False},
    "AO": {"type": "vowel", "height": "low_mid", "backness": "back", "rounding": "rounded", "diphthong": False},
    "AA": {"type": "vowel", "height": "low", "backness": "back", "rounding": "unrounded", "diphthong": False},
    "AH": {"type": "vowel", "height": "mid", "backness": "central", "rounding": "unrounded", "diphthong": False},
    "ER": {"type": "vowel", "height": "mid", "backness": "central", "rounding": "unrounded", "rhotic": True},
    "AE": {"type": "vowel", "height": "low", "backness": "front", "rounding": "unrounded", "diphthong": False},
    "EH": {"type": "vowel", "height": "mid", "backness": "front", "rounding": "unrounded", "diphthong": False},
    "IH": {"type": "vowel", "height": "high_mid", "backness": "front", "rounding": "unrounded", "diphthong": False},
    "IY": {"type": "vowel", "height": "high", "backness": "front", "rounding": "unrounded", "diphthong": False},
    "EY": {"type": "vowel", "height": "mid", "backness": "front", "rounding": "unrounded", "diphthong": True},
    "AY": {"type": "vowel", "height": "low_to_high", "backness": "front", "rounding": "unrounded", "diphthong": True},
}

VOWELS = {p for p, f in PHONE_FEATURES.items() if f["type"] == "vowel"}

EDGE_TTS_VOICES = {
    "female": "en-AU-NatashaNeural",
    "woman": "en-AU-NatashaNeural",
    "male": "en-AU-WilliamNeural",
    "man": "en-AU-WilliamNeural",
}

VOWEL_VARIANTS: dict[str, list[tuple[str, float]]] = {
    "OW": [("OW", 0.00), ("UW", 0.25), ("AH", 0.35), ("AO", 0.40), ("ER", 0.55)],
    "UW": [("UW", 0.00), ("UH", 0.20), ("AH", 0.35), ("AO", 0.45), ("ER", 0.55)],
    "EY": [("EY", 0.00), ("EH", 0.25), ("IH", 0.30), ("AH", 0.45), ("ER", 0.55)],
    "EH": [("EH", 0.00), ("IH", 0.25), ("AH", 0.35), ("ER", 0.50)],
    "AE": [("AE", 0.00), ("EH", 0.25), ("AH", 0.35), ("AO", 0.55), ("ER", 0.60)],
    "AA": [("AA", 0.00), ("AO", 0.25), ("AH", 0.35), ("ER", 0.55)],
}

CONSONANT_VARIANTS: dict[str, list[tuple[str, float]]] = {
    "G": [("G", 0.00), ("K", 0.25), ("D", 0.45)],
    "P": [("P", 0.00), ("B", 0.25)],
    "B": [("B", 0.00), ("P", 0.25)],
    "T": [("T", 0.00), ("D", 0.25)],
    "S": [("S", 0.00), ("Z", 0.25), ("SH", 0.40)],
    "N": [("N", 0.00), ("M", 0.35), ("NG", 0.40)],
    "F": [("F", 0.00), ("V", 0.25)],
    "TH": [("TH", 0.00), ("F", 0.40)],
}

INSERTIONS: list[tuple[list[str], float, str]] = [
    (["Y"], 0.55, "palatal_glide"),
    (["W"], 0.65, "rounded_glide"),
    (["AH"], 0.70, "weak_schwa"),
    (["ER"], 0.80, "rhotic_vowel"),
]

PHONE_TO_TEXT = {
    "N": "n", "M": "m", "NG": "ng", "G": "g", "K": "k", "D": "d", "T": "t",
    "P": "p", "B": "b", "S": "s", "Z": "z", "SH": "sh", "TH": "th", "F": "f",
    "V": "v", "HH": "h", "L": "l", "R": "r", "Y": "y", "W": "w",
    "OW": "oh", "UW": "oo", "UH": "uh", "AO": "aw", "AA": "ah", "AH": "uh",
    "ER": "er", "AE": "a", "EH": "eh", "IH": "ih", "IY": "ee", "EY": "ay", "AY": "eye",
}


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def is_vowel_phone(phone: str) -> bool:
    return phone in VOWELS


def safe_filename(text: str) -> str:
    keep = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_"}:
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep).strip("_") or "candidate"


def wav_duration_seconds(path: str | Path) -> float:
    with contextlib.closing(wave.open(str(path), "rb")) as wf:
        return wf.getnframes() / float(wf.getframerate())


def estimate_expected_duration_seconds(phones: list[str]) -> float:
    duration = 0.12
    for p in phones:
        if p in VOWELS:
            duration += 0.24
        elif p in {"Y", "W", "L", "R"}:
            duration += 0.12
        else:
            duration += 0.09
    return max(0.35, duration)


def validate_rendered_duration(
    wav_path: str | Path,
    phones: list[str],
    max_duration_sec: float = 3.0,
    max_duration_ratio: float = 3.0,
) -> tuple[bool, dict[str, float | str]]:
    actual = wav_duration_seconds(wav_path)
    expected = estimate_expected_duration_seconds(phones)
    allowed = min(float(max_duration_sec), float(max_duration_ratio) * expected)
    ok = actual <= allowed
    return ok, {
        "actual_duration_sec": actual,
        "expected_duration_sec": expected,
        "allowed_duration_sec": allowed,
        "status": "ok" if ok else "rejected_overlong_tts_artifact",
    }


def phones_to_pseudoword(phones: list[str]) -> str:
    """
    Convert phones to a compact pseudo-word.

    Syllable simulation has been removed by default.

    Examples:
        N UH ER      -> nuher
        N AH W OW    -> nuhwoh
        N Y ER OW    -> nyeroh

    The duration guard remains active downstream, so if a compact pseudo-word
    causes a TTS runaway artifact, that candidate is rejected/scored as unusable.
    """
    return "".join(PHONE_TO_TEXT.get(p, p.lower()) for p in phones)


def candidate_grapheme_for_phones(
    word: str,
    canonical: list[str],
    phones: list[str],
    level: float,
    ground_truth_level: float = 0.05,
) -> str:
    """
    Use the literal target word for near-ground-truth candidates.

    Example:
        word="yes", phones=["Y", "EH", "S"], level=0.0 -> "yes"

    This avoids rendering canonical phone spellings like "yehs" when the desired
    ground truth is simply the real word.
    """
    if phones == canonical:
        return word
    return phones_to_pseudoword(phones)


def articulatory_distance(a: str, b: str) -> float:
    if a == b:
        return 0.0

    fa, fb = PHONE_FEATURES.get(a), PHONE_FEATURES.get(b)
    if fa is None or fb is None:
        return 0.75
    if fa.get("type") != fb.get("type"):
        return 0.90

    if fa["type"] == "vowel":
        keys = ["height", "backness", "rounding", "diphthong", "rhotic"]
        weights = [0.30, 0.30, 0.20, 0.15, 0.05]
    else:
        keys = ["manner", "place", "voicing"]
        weights = [0.45, 0.35, 0.20]

    return min(1.0, sum(w for k, w in zip(keys, weights) if fa.get(k) != fb.get(k)))


def phone_sequence_distance(source: list[str], candidate: list[str]) -> float:
    max_len = max(len(source), len(candidate), 1)
    total = 0.0
    for i in range(max_len):
        if i >= len(source) or i >= len(candidate):
            total += 0.65
        else:
            total += articulatory_distance(source[i], candidate[i])
    return total / max_len


def pronunciation_similarity(source: list[str], candidate: list[str]) -> float:
    dist = phone_sequence_distance(source, candidate)

    source_vowels = [p for p in source if p in VOWELS]
    cand_vowels = [p for p in candidate if p in VOWELS]
    if source_vowels and cand_vowels:
        vowel_sim = 1.0 - articulatory_distance(source_vowels[0], cand_vowels[0])
    else:
        vowel_sim = 1.0 if not source_vowels and not cand_vowels else 0.0

    first_sim = 1.0 - articulatory_distance(source[0], candidate[0]) if source and candidate else 0.0
    syllable_sim = 1.0 - min(1.0, 0.25 * abs(len(source_vowels) - len(cand_vowels)))

    return max(0.0, min(1.0, 0.40 * (1.0 - dist) + 0.25 * first_sim + 0.25 * vowel_sim + 0.10 * syllable_sim))


# ---------------------------------------------------------------------
# Candidate data and generation
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------
# Blabber library / triphone search output
# ---------------------------------------------------------------------

BLABBER_SCHEMA_VERSION = "2.0-triphone"


def phone_sequence_key(phones: list[str]) -> str:
    """Stable key for exact phone-sequence search."""
    return " ".join(phones)


def triphone_keys(
    phones: list[str],
    left_boundary: str = "<s>",
    right_boundary: str = "</s>",
) -> list[str]:
    """
    Build searchable triphone keys with explicit word boundaries.

    Example:
        ["N", "OW"] ->
            ["<s>|N|OW", "N|OW|</s>"]

        ["N", "Y", "ER", "OW"] ->
            ["<s>|N|Y", "N|Y|ER", "Y|ER|OW", "ER|OW|</s>"]
    """
    padded = [left_boundary] + list(phones) + [right_boundary]
    return [
        f"{padded[i]}|{padded[i + 1]}|{padded[i + 2]}"
        for i in range(len(padded) - 2)
    ]


def biphone_keys(
    phones: list[str],
    left_boundary: str = "<s>",
    right_boundary: str = "</s>",
) -> list[str]:
    padded = [left_boundary] + list(phones) + [right_boundary]
    return [
        f"{padded[i]}|{padded[i + 1]}"
        for i in range(len(padded) - 1)
    ]


def phone_inventory_keys(phones: list[str]) -> list[str]:
    return sorted(set(phones))


def prepare_piper_text(text: str) -> str:
    """
    Piper behaves more reliably on short words and pseudowords when the input
    looks like a complete utterance. A terminal stop and trailing newline help
    prevent clipped endings at the synthesis boundary.
    """
    normalized = " ".join(str(text).strip().split())
    if not normalized:
        raise ValueError("Cannot render empty text with Piper.")
    if normalized[-1] not in ".!?":
        normalized += "."
    return normalized + "\n"


def gender_for_voice_mode(voice_mode: str | None) -> str | None:
    if voice_mode == "woman":
        return "female"
    if voice_mode == "man":
        return "male"
    return None


def infer_piper_voice_mode(model_path: str | None) -> str | None:
    model = str(model_path or "").lower()
    if "northern_english_male" in model:
        return "man"
    if "jenny_dioco" in model:
        return "woman"
    return None


def infer_edge_voice_mode(voice_name: str | None) -> str | None:
    voice = str(voice_name or "").strip().lower()
    if not voice:
        return None
    if "natasha" in voice:
        return "woman"
    if "william" in voice:
        return "man"
    return None


def infer_kokoro_voice_mode(voice_name: str | None) -> str | None:
    voice = str(voice_name or "").strip().lower()
    if voice.startswith("bf_"):
        return "woman"
    if voice.startswith("bm_"):
        return "man"
    return None


def infer_voice_mode_from_voice_name(voice_name: str | None) -> str | None:
    return (
        infer_edge_voice_mode(voice_name)
        or infer_kokoro_voice_mode(voice_name)
        or gender_for_voice_mode(str(voice_name or "").strip().lower())
    )


def resolve_edge_voice_mode(args) -> str | None:
    explicit = None if args.voice_mode == "auto" else args.voice_mode
    if explicit is not None:
        return explicit
    if args.edge_voice:
        return infer_edge_voice_mode(args.edge_voice)
    edge_voices = [v.strip() for v in str(args.edge_voices or "").split(",") if v.strip()]
    if len(edge_voices) > 1:
        return None
    if len(edge_voices) == 1:
        return infer_edge_voice_mode(edge_voices[0])
    return "woman" if args.edge_voice_mode in {"female", "woman"} else "man"


def resolve_tts_voices(args) -> list[str]:
    if args.tts_engine == "kokoro":
        return [v.strip() for v in args.kokoro_voices.split(",") if v.strip()]
    if args.tts_engine == "piper":
        return [f"piper_speaker_{args.piper_speaker}" if args.piper_speaker is not None else "piper"]
    if args.tts_engine == "edge":
        edge_voices = [v.strip() for v in str(args.edge_voices or "").split(",") if v.strip()]
        if edge_voices:
            return edge_voices
        if args.edge_voice:
            return [args.edge_voice]
        return [EDGE_TTS_VOICES[args.edge_voice_mode]]
    return [args.coqui_speaker or "coqui"]


def make_blabber_asset(
    c: "Candidate",
    canonical_phones: list[str],
    asset_index: int,
    voice_mode: str | None = None,
) -> dict[str, Any]:
    """
    Convert an internal Candidate to a build_blabber_library.py /
    select_blabber_asset.py friendly asset.

    The raw candidate is preserved under `candidate`; top-level fields expose
    the triphone search method.
    """
    voice = c.voice or "none"
    asset_id = (
        f"{c.word}"
        f"__L{c.level:.3f}"
        f"__{safe_filename(c.grapheme)}"
        f"__{safe_filename(voice)}"
        f"__{asset_index:04d}"
    )

    phones = list(c.phones)
    canonical = list(canonical_phones)
    tri = triphone_keys(phones)
    bi = biphone_keys(phones)
    inventory = phone_inventory_keys(phones)

    is_ground_truth = phones == canonical

    effective_voice_mode = voice_mode or infer_voice_mode_from_voice_name(c.voice)
    gender = gender_for_voice_mode(effective_voice_mode)
    return {
        "schema_version": BLABBER_SCHEMA_VERSION,
        "asset_id": asset_id,
        "is_ground_truth": is_ground_truth,
        "asset_role": "ground_truth" if is_ground_truth else "augmentation",
        "generation_mode": "global",

        "word": c.word,
        "target_word": c.word,
        "candidate_text": c.grapheme,
        "grapheme": c.grapheme,
        "phones": phones,
        "canonical_phones": canonical,
        "phone_key": phone_sequence_key(phones),
        "canonical_phone_key": phone_sequence_key(canonical),

        "triphone_keys": tri,
        "triphones": tri,
        "biphone_keys": bi,
        "phone_inventory": inventory,
        "search_terms": sorted(set(
            [c.word, c.grapheme, phone_sequence_key(phones), phone_sequence_key(canonical)]
            + phones
            + canonical
            + tri
            + bi
            + inventory
        )),

        "level": c.level,
        "morph_level": c.level,
        "distortion_level": c.level,
        "phoneme_values": [c.level],
        "quality": c.level,
        "phone_distance": c.phone_distance,
        "phone_similarity": c.phone_similarity,
        "neural_distance": c.neural_distance,
        "neural_similarity": c.neural_similarity,
        "suitability_score": c.suitability_score,
        "operations": list(c.operations),

        "voice": c.voice,
        "voice_mode": effective_voice_mode,
        "gender": gender,
        "source_phonemes": canonical,
        "output_phonemes": phones,
        "wav_path": c.wav_path,
        "audio_path": c.wav_path,
        "output_audio_path": c.wav_path,
        "duration_info": c.duration_info,

        "candidate": asdict(c),
    }


def build_blabber_search_index(assets: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Precompute inverted indexes for select_blabber_asset.py.

    Searchable buckets:
      - by_word
      - by_voice
      - by_candidate_text
      - by_phone_key
      - by_triphone
      - by_biphone
      - by_phone
    """
    index: dict[str, dict[str, list[str]]] = {
        "by_asset_id": {},
        "by_word": {},
        "by_voice": {},
        "by_candidate_text": {},
        "by_phone_key": {},
        "by_triphone": {},
        "by_biphone": {},
        "by_phone": {},
    }

    def add(bucket: str, key: str | None, asset_id: str) -> None:
        if key is None:
            key = "none"
        index[bucket].setdefault(str(key), []).append(asset_id)

    for a in assets:
        asset_id = a["asset_id"]
        index["by_asset_id"][asset_id] = [asset_id]
        add("by_word", a.get("word"), asset_id)
        add("by_voice", a.get("voice"), asset_id)
        add("by_candidate_text", a.get("candidate_text"), asset_id)
        add("by_phone_key", a.get("phone_key"), asset_id)

        for key in a.get("triphone_keys", []):
            add("by_triphone", key, asset_id)
        for key in a.get("biphone_keys", []):
            add("by_biphone", key, asset_id)
        for key in a.get("phone_inventory", []):
            add("by_phone", key, asset_id)

    return index


@dataclass
class Candidate:
    word: str
    level: float
    grapheme: str
    phones: list[str]
    phone_distance: float
    phone_similarity: float
    neural_distance: float | None
    neural_similarity: float | None
    suitability_score: float | None
    operations: list[str]
    voice: str | None = None
    wav_path: str | None = None
    duration_info: dict[str, Any] | None = None


def variants_for_phone(phone: str, max_distance: float) -> list[tuple[str, float, str]]:
    if phone in VOWELS:
        return [(p, d, f"vowel:{phone}->{p}") for p, d in VOWEL_VARIANTS.get(phone, [(phone, 0.0)]) if d <= max_distance]
    return [(p, d, f"consonant:{phone}->{p}") for p, d in CONSONANT_VARIANTS.get(phone, [(phone, 0.0)]) if d <= max_distance]


def generate_phone_search_space(
    word: str,
    canonical: list[str],
    level: float,
    max_candidates: int = 128,
    preserve_first_phone: bool = True,
    ground_truth_level: float = 0.05,
    include_ground_truth_each_level: bool = True,
) -> list[Candidate]:
    """
    Generate phone candidates for a requested level.

    Important behavior:
      - `level` controls how far augmented candidates may drift.
      - the canonical / ground-truth candidate is kept visible at every level
        when include_ground_truth_each_level=True.

    This fixes the bug where ground truth only appeared at level 0. In a triphone
    library, selection often needs to compare the canonical asset against
    augmented assets at the same level bucket.
    """
    level = max(0.0, min(1.0, float(level)))
    max_phone_distance = 0.05 + 0.65 * level

    per_phone_options = []
    for i, p in enumerate(canonical):
        if i == 0 and preserve_first_phone:
            per_phone_options.append([(p, 0.0, "preserve:first_phone")])
        else:
            per_phone_options.append(variants_for_phone(p, max_phone_distance))

    raw = []
    for combo in itertools.product(*per_phone_options):
        phones = [x[0] for x in combo]
        ops = [x[2] for x in combo if "->" in x[2] and not x[2].endswith(f"->{x[0]}")]
        raw.append((phones, ops))

    expanded = list(raw)
    for phones, ops in raw:
        for insertion, threshold, label in INSERTIONS:
            if level >= threshold:
                insert_pos = 1
                for idx, p in enumerate(phones):
                    if p in VOWELS:
                        insert_pos = idx
                        break
                expanded.append((phones[:insert_pos] + insertion + phones[insert_pos:], ops + [f"insert:{'+'.join(insertion)}:{label}"]))

    # Always keep a canonical ground-truth candidate in every level bucket.
    # This asset uses the literal target word, but its level is the current
    # library level so select_blabber_asset.py can find the ground truth inside
    # the same augmentation level.
    if include_ground_truth_each_level:
        expanded.append((list(canonical), ["ground_truth:canonical_visible_at_level"]))

    candidates = []
    seen = set()
    for phones, ops in expanded:
        key = tuple(phones)

        # Do not deduplicate away the explicit ground-truth candidate just
        # because the canonical sequence was already generated naturally.
        is_explicit_gt = "ground_truth:canonical_visible_at_level" in ops
        seen_key = (key, "ground_truth") if is_explicit_gt else (key, "candidate")
        if seen_key in seen:
            continue
        seen.add(seen_key)

        phone_dist = phone_sequence_distance(canonical, phones)
        phone_sim = pronunciation_similarity(canonical, phones)

        # Looser at higher levels, conservative at lower levels.
        # Ground truth is always allowed.
        threshold = 0.82 - 0.30 * level
        if phone_sim < threshold and not is_explicit_gt:
            continue

        grapheme = word if phones == canonical else candidate_grapheme_for_phones(
            word,
            canonical,
            phones,
            level,
            ground_truth_level=ground_truth_level,
        )

        candidate = Candidate(
            word=word,
            level=level,
            grapheme=grapheme,
            phones=phones,
            phone_distance=phone_dist,
            phone_similarity=phone_sim,
            neural_distance=None,
            neural_similarity=None,
            suitability_score=None,
            operations=ops,
        )

        # Tag canonical candidates for selectors/builders.
        if phones == canonical and "ground_truth:canonical" not in candidate.operations:
            candidate.operations = candidate.operations + ["ground_truth:canonical"]

        candidates.append(candidate)

    def candidate_sort_key(c: Candidate):
        is_gt = tuple(c.phones) == tuple(canonical)
        desired_distance = level * 0.55

        # At level 0, ground truth should rank first.
        # At higher levels, augmented candidates should still rank by closeness
        # to the requested augmentation strength, but ground truth remains present
        # and searchable.
        if level <= ground_truth_level:
            gt_rank = 0 if is_gt else 1
        else:
            gt_rank = 1 if is_gt else 0

        return (
            gt_rank,
            abs(c.phone_distance - desired_distance),
            -c.phone_similarity,
            c.grapheme,
        )

    candidates.sort(key=candidate_sort_key)

    # Preserve ground truth after truncation.
    if include_ground_truth_each_level:
        gt_candidates = [c for c in candidates if tuple(c.phones) == tuple(canonical)]
        non_gt_candidates = [c for c in candidates if tuple(c.phones) != tuple(canonical)]
        # Keep a single canonical per level; prefer literal word and explicit tag.
        gt = gt_candidates[:1]
        candidates = (gt + non_gt_candidates)[:max_candidates] if level <= ground_truth_level else (non_gt_candidates[:max_candidates - len(gt)] + gt)
    else:
        candidates = candidates[:max_candidates]

    return candidates


def clone_candidate_for_voice(c: Candidate, voice: str) -> Candidate:
    data = asdict(c)
    data["voice"] = voice
    return Candidate(**data)


def summarize_level_candidates(
    candidates: list[Candidate],
    canonical: list[str],
) -> list[dict[str, Any]]:
    """
    Build an explicit per-level summary so callers can verify that the
    canonical/base word is present in each level bucket and counted.
    """
    by_level: dict[float, list[Candidate]] = {}
    for candidate in candidates:
        by_level.setdefault(float(candidate.level), []).append(candidate)

    summary: list[dict[str, Any]] = []
    canonical_tuple = tuple(canonical)
    for level in sorted(by_level):
        level_candidates = by_level[level]
        ground_truth = [c for c in level_candidates if tuple(c.phones) == canonical_tuple]
        summary.append({
            "level": level,
            "candidate_count": len(level_candidates),
            "ground_truth_count": len(ground_truth),
            "ground_truth_present": bool(ground_truth),
            "ground_truth_words": sorted({c.grapheme for c in ground_truth}),
        })
    return summary


# ---------------------------------------------------------------------
# Kokoro rendering
# ---------------------------------------------------------------------

class KokoroRenderer:
    """
    Kokoro renderer with default voices:
      - bf_emma
      - bm_lewis

    Typical kokoro usage:
      from kokoro import KPipeline
      pipeline = KPipeline(lang_code="b")
      generator = pipeline(text, voice="bf_emma")
      for gs, ps, audio in generator: ...
    """

    def __init__(self, lang_code: str = "b", speed: float = 1.0):
        try:
            from kokoro import KPipeline
        except Exception as exc:
            raise RuntimeError(
                "Kokoro is not installed or could not be imported. Install kokoro "
                "or run with --skip-neural for candidate generation only."
            ) from exc

        self.lang_code = lang_code
        self.speed = speed
        self.pipeline = KPipeline(lang_code=lang_code)

    def render(self, text: str, out_wav: str | Path, voice: str) -> None:
        try:
            import soundfile as sf
        except Exception as exc:
            raise RuntimeError("soundfile is required to write Kokoro audio. Install with: pip install soundfile") from exc

        chunks = []
        generator = self.pipeline(text, voice=voice, speed=self.speed)
        for _, _, audio in generator:
            audio_np = np.asarray(audio, dtype=np.float32)
            if audio_np.ndim > 1:
                audio_np = audio_np.reshape(-1)
            chunks.append(audio_np)

        if not chunks:
            raise RuntimeError(f"Kokoro generated no audio for text={text!r}, voice={voice!r}")

        wav = np.concatenate(chunks)
        sf.write(str(out_wav), wav, 24000)



# ---------------------------------------------------------------------
# Piper renderer
# ---------------------------------------------------------------------

class PiperRenderer:
    """
    Piper CLI renderer.

    Piper is a strong backend for pseudoword batch rendering because it is fast,
    local, and usually more predictable than autoregressive neural TTS.

    Requires a local Piper executable and ONNX voice model.

    Example:
      --tts-engine piper
      --piper-model en_GB-alan-medium.onnx
      --piper-config en_GB-alan-medium.onnx.json

    If --piper-config is omitted, Piper will try its default behavior.
    """

    def __init__(
        self,
        model_path: str,
        config_path: str | None = None,
        executable: str = "piper",
        speaker: int | None = None,
        length_scale: float | None = None,
        noise_scale: float | None = None,
        noise_w: float | None = None,
        sentence_silence: float | None = None,
        use_python_module: bool = False,
    ):
        self.use_python_module = bool(use_python_module)

        exe = shutil.which(executable) or shutil.which("piper") or shutil.which("piper-tts")

        # pip-installed Piper sometimes works as an executable, and in some
        # environments it is easier to invoke as `python -m piper`.
        if self.use_python_module:
            self.command_prefix = [sys.executable, "-m", "piper"]
        elif exe is not None:
            self.command_prefix = [exe]
        else:
            self.command_prefix = [sys.executable, "-m", "piper"]

        self.model_path = str(model_path)
        self.config_path = str(config_path) if config_path else None
        self.speaker = speaker
        self.length_scale = length_scale
        self.noise_scale = noise_scale
        self.noise_w = noise_w
        self.sentence_silence = sentence_silence

        if not Path(self.model_path).exists():
            raise RuntimeError(f"Piper model not found: {self.model_path}")
        if self.config_path and not Path(self.config_path).exists():
            raise RuntimeError(f"Piper config not found: {self.config_path}")

    def render(self, text: str, out_wav: str | Path, voice: str | None = None) -> None:
        render_text = prepare_piper_text(text)
        cmd = [
            *self.command_prefix,
            "--model", self.model_path,
            "--output_file", str(out_wav),
        ]

        if self.config_path:
            cmd.extend(["--config", self.config_path])

        if self.speaker is not None:
            cmd.extend(["--speaker", str(self.speaker)])

        if self.length_scale is not None:
            cmd.extend(["--length_scale", str(self.length_scale)])

        if self.noise_scale is not None:
            cmd.extend(["--noise_scale", str(self.noise_scale)])

        if self.noise_w is not None:
            cmd.extend(["--noise_w", str(self.noise_w)])

        if self.sentence_silence is not None:
            cmd.extend(["--sentence_silence", str(self.sentence_silence)])

        # Piper reads text from stdin.
        proc = subprocess.run(
            cmd,
            input=render_text,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Piper rendering failed for text="
                + repr(text)
                + " prepared_as="
                + repr(render_text)
                + "\\nSTDERR:\\n"
                + proc.stderr
            )


# ---------------------------------------------------------------------
# Edge TTS renderer
# ---------------------------------------------------------------------

class EdgeTTSRenderer:
    def __init__(self, voice: str, rate: str = "+0%", pitch: str = "+0Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            raise RuntimeError("ffmpeg is required for --tts-engine edge to convert MP3 output into WAV.")

    def render(self, text: str, out_wav: str | Path, voice: str | None = None) -> None:
        try:
            import edge_tts
        except Exception as exc:
            raise RuntimeError("edge-tts is not installed. Install with: pip install edge-tts") from exc

        selected_voice = voice or self.voice
        normalized = " ".join(str(text).strip().split())
        if not normalized:
            raise ValueError("Cannot render empty text with Edge TTS.")
        if normalized[-1] not in ".!?":
            normalized += "."

        async def _render_mp3(mp3_path: Path) -> None:
            communicate = edge_tts.Communicate(
                text=normalized,
                voice=selected_voice,
                rate=self.rate,
                pitch=self.pitch,
            )
            await communicate.save(str(mp3_path))

        out_wav = Path(out_wav)
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            mp3_path = Path(tmp.name)

        try:
            asyncio.run(_render_mp3(mp3_path))
            subprocess.run(
                [
                    self.ffmpeg,
                    "-y",
                    "-i",
                    str(mp3_path),
                    "-ar",
                    "24000",
                    "-ac",
                    "1",
                    str(out_wav),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            mp3_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------
# Optional Coqui renderer retained as fallback
# ---------------------------------------------------------------------

class CoquiRenderer:
    def __init__(
        self,
        model_name: str,
        speaker: str | None = None,
        language: str | None = None,
        speaker_wav: str | None = None,
    ):
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise RuntimeError("Coqui TTS is not installed. Install with: pip install TTS") from exc

        self.tts = TTS(model_name=model_name)
        self.speaker = speaker
        self.language = language
        self.speaker_wav = speaker_wav

    def render(self, text: str, out_wav: str | Path, voice: str | None = None) -> None:
        kwargs = {}
        if self.speaker is not None:
            kwargs["speaker"] = self.speaker
        if self.language is not None:
            kwargs["language"] = self.language
        if self.speaker_wav is not None:
            kwargs["speaker_wav"] = self.speaker_wav
        self.tts.tts_to_file(text=text, file_path=str(out_wav), **kwargs)


# ---------------------------------------------------------------------
# Torchaudio neural embedding scorer
# ---------------------------------------------------------------------

class TorchAudioEmbeddingScorer:
    def __init__(self, bundle_name: str = "WAV2VEC2_ASR_BASE_960H", device: str | None = None):
        try:
            import torch
            import torchaudio
        except Exception as exc:
            raise RuntimeError(
                "torch/torchaudio could not be imported. Use --skip-neural or "
                "install matching CPU/CUDA builds of torch and torchaudio."
            ) from exc

        self.torch = torch
        self.torchaudio = torchaudio
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        bundle = getattr(torchaudio.pipelines, bundle_name)
        self.sample_rate = bundle.sample_rate
        self.model = bundle.get_model().to(self.device).eval()

    def load_audio(self, wav_path: str | Path):
        torch = self.torch
        torchaudio = self.torchaudio
        with torch.no_grad():
            wav, sr = torchaudio.load(str(wav_path))
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            return wav.to(self.device)

    def embed(self, wav_path: str | Path):
        torch = self.torch
        with torch.no_grad():
            wav = self.load_audio(wav_path)
            if hasattr(self.model, "extract_features"):
                features, _ = self.model.extract_features(wav)
                feat = features[-1]
            else:
                emission, _ = self.model(wav)
                feat = emission

            emb = feat.mean(dim=1).squeeze(0)
            emb = torch.nn.functional.normalize(emb, dim=0)
            return emb.detach().cpu()

    def cosine_distance(self, a, b) -> float:
        torch = self.torch
        sim = torch.nn.functional.cosine_similarity(a, b, dim=0).item()
        return float(max(0.0, min(2.0, 1.0 - sim)))


# ---------------------------------------------------------------------
# Rendering / scoring loop
# ---------------------------------------------------------------------

def render_candidate(
    renderer,
    engine: str,
    text: str,
    out_wav: Path,
    voice: str | None,
) -> None:
    if engine == "kokoro":
        assert voice is not None
        renderer.render(text, out_wav, voice=voice)
    elif engine == "edge":
        renderer.render(text, out_wav, voice=voice)
    elif engine == "piper":
        renderer.render(text, out_wav, voice=voice)
    elif engine == "coqui":
        renderer.render(text, out_wav, voice=voice)
    else:
        raise ValueError(f"Unsupported renderer engine: {engine}")


def make_renderer(args):
    if args.tts_engine == "kokoro":
        return KokoroRenderer(lang_code=args.kokoro_lang_code, speed=args.kokoro_speed)
    if args.tts_engine == "edge":
        voice = args.edge_voice or EDGE_TTS_VOICES[args.edge_voice_mode]
        return EdgeTTSRenderer(
            voice=voice,
            rate=args.edge_rate,
            pitch=args.edge_pitch,
        )
    if args.tts_engine == "piper":
        if not args.piper_model:
            raise RuntimeError("--piper-model is required when --tts-engine piper")
        return PiperRenderer(
            model_path=args.piper_model,
            config_path=args.piper_config,
            executable=args.piper_exe,
            speaker=args.piper_speaker,
            use_python_module=args.piper_use_python_module,
            length_scale=args.piper_length_scale,
            noise_scale=args.piper_noise_scale,
            noise_w=args.piper_noise_w,
            sentence_silence=args.piper_sentence_silence,
        )
    if args.tts_engine == "coqui":
        return CoquiRenderer(
            model_name=args.coqui_model,
            speaker=args.coqui_speaker,
            language=args.coqui_language,
            speaker_wav=args.speaker_wav,
        )
    raise ValueError(f"Unsupported TTS engine: {args.tts_engine}")


def synthesize_reference_if_needed(
    word: str,
    reference_wav: str | None,
    renderer,
    args,
    work_dir: Path,
) -> Path:
    if reference_wav:
        return Path(reference_wav)

    work_dir.mkdir(parents=True, exist_ok=True)
    voices = resolve_tts_voices(args)
    voice = voices[0] if voices else None
    ref = work_dir / f"reference_{safe_filename(word)}_{safe_filename(voice or 'coqui')}.wav"
    if not ref.exists():
        render_candidate(renderer, args.tts_engine, word, ref, voice=voice)
    return ref


def score_candidates_with_audio(
    candidates: list[Candidate],
    reference_wav: Path | None,
    render_dir: Path,
    renderer,
    args,
    target_level: float,
    scorer: TorchAudioEmbeddingScorer | None,
    ground_truth_audio_paths: dict[tuple[str, str], Path] | None = None,
) -> list[Candidate]:
    render_dir.mkdir(parents=True, exist_ok=True)

    ref_emb = scorer.embed(reference_wav) if scorer is not None and reference_wav is not None else None

    voices = resolve_tts_voices(args)

    scored: list[Candidate] = []

    for base_i, base_c in enumerate(candidates):
        for voice in voices:
            c = clone_candidate_for_voice(base_c, voice)
            is_ground_truth = "ground_truth:canonical" in c.operations
            ground_truth_key = (c.word, voice)
            cached_ground_truth = (
                ground_truth_audio_paths.get(ground_truth_key)
                if ground_truth_audio_paths is not None
                else None
            )
            if is_ground_truth and cached_ground_truth is not None:
                wav_path = cached_ground_truth
                if "ground_truth:reused_level_0_audio" not in c.operations:
                    c.operations = c.operations + ["ground_truth:reused_level_0_audio"]
            else:
                wav_path = render_dir / f"{base_i:04d}_L{c.level:.2f}_{safe_filename(c.grapheme)}_{safe_filename(voice)}.wav"

            if (not is_ground_truth or cached_ground_truth is None) and (args.render_existing or not wav_path.exists()):
                render_candidate(renderer, args.tts_engine, c.grapheme, wav_path, voice=voice)

            if (
                is_ground_truth
                and ground_truth_audio_paths is not None
                and cached_ground_truth is None
                and abs(float(c.level)) < 1e-8
            ):
                ground_truth_audio_paths[ground_truth_key] = wav_path

            ok_duration, duration_info = validate_rendered_duration(
                wav_path,
                c.phones,
                max_duration_sec=args.max_duration_sec,
                max_duration_ratio=args.max_duration_ratio,
            )
            c.wav_path = str(wav_path)
            c.duration_info = duration_info

            if not ok_duration and not args.allow_overlong:
                c.neural_distance = 999.0
                c.neural_similarity = 0.0
                c.suitability_score = 0.0
                c.operations = c.operations + [str(duration_info["status"])]
                scored.append(c)
                continue

            desired_phone_dist = target_level * 0.55
            level_match = 1.0 - min(1.0, abs(c.phone_distance - desired_phone_dist) / 0.55)

            if scorer is not None and ref_emb is not None:
                emb = scorer.embed(wav_path)
                neural_dist = scorer.cosine_distance(ref_emb, emb)
                neural_sim = 1.0 - min(1.0, neural_dist)
                c.neural_distance = neural_dist
                c.neural_similarity = neural_sim
                c.suitability_score = 0.45 * neural_sim + 0.30 * c.phone_similarity + 0.25 * level_match
            else:
                c.suitability_score = 0.60 * c.phone_similarity + 0.40 * level_match

            scored.append(c)

    scored.sort(key=lambda x: (-(x.suitability_score or 0.0), x.neural_distance or 999.0, x.grapheme, x.voice or ""))
    return scored


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()

    p.add_argument("--word", required=True, choices=sorted(LEXICON))
    p.add_argument(
        "--voice-mode",
        choices=["auto", "woman", "man"],
        default="auto",
        help="Voice bank metadata to write into generated triphone assets.",
    )
    p.add_argument("--level", type=float, default=0.75, help="Target candidate level 0..1.")
    p.add_argument("--levels", type=int, default=None, help="Generate a ladder of levels instead of one level.")
    p.add_argument("--max-candidates", type=int, default=64)
    p.add_argument(
        "--ground-truth-level",
        type=float,
        default=0.05,
        help="Levels <= this render the canonical phone sequence as the literal word, e.g. yes not yehs.",
    )
    p.add_argument(
        "--no-ground-truth-each-level",
        action="store_true",
        help="Disable the default behavior that keeps the canonical/base word in every level bucket.",
    )

    p.add_argument("--reference-wav", type=str, default=None)
    p.add_argument("--render-dir", type=str, default="kokoro_candidate_audio")
    p.add_argument("--json-out", type=str, default=None)

    p.add_argument("--tts-engine", choices=["kokoro", "edge", "piper", "coqui"],
                   default="piper")

    # Kokoro controls.
    p.add_argument("--kokoro-voices", type=str, default="bf_emma,bm_lewis")
    p.add_argument("--kokoro-lang-code", type=str, default="b")
    p.add_argument("--kokoro-speed", type=float, default=1.0)

    # Edge controls.
    p.add_argument("--edge-voice", type=str, default=None)
    p.add_argument("--edge-voices", type=str, default=None)
    p.add_argument("--edge-voice-mode", choices=["female", "male", "woman", "man"], default="female")
    p.add_argument("--edge-rate", type=str, default="+0%")
    p.add_argument("--edge-pitch", type=str, default="+0Hz")

    # Piper controls.
    p.add_argument("--piper-exe", type=str, default="piper")
    p.add_argument("--piper-use-python-module", action="store_true", help="Invoke Piper as `python -m piper`, useful for some pip installs.")
    p.add_argument("--piper-model", type=str, 
                   default="piper-models/en_GB-jenny_dioco-medium.onnx",
                   choices=["piper-models/en_GB-jenny_dioco-medium.onnx",
                            "piper-models/en_GB-northern_english_male-medium.onnx"],
                   help="Path to Piper ONNX voice model. Required for --tts-engine piper.")
    p.add_argument("--piper-config", type=str,
                   default="piper-models/en_GB-jenny_dioco-medium.onnx.json",
                   choices=["piper-models/en_GB-jenny_dioco-medium.onnx.json",
                            "piper-models/en_GB-northern_english_male-medium.onnx.json"],
                   help="Optional Piper JSON config path.")
    p.add_argument("--piper-speaker", type=int, default=None, help="Speaker id for multi-speaker Piper models.")
    p.add_argument("--piper-length-scale", type=float, default=1.2, 
                   help="Piper length_scale. Larger is slower/longer.")
    p.add_argument("--piper-noise-scale", type=float, default=None)
    p.add_argument("--piper-noise-w", type=float, default=None)
    p.add_argument("--piper-sentence-silence", type=float, default=None)

    # Coqui fallback controls.
    p.add_argument("--coqui-model", type=str, default="tts_models/en/ljspeech/tacotron2-DDC")
    p.add_argument("--coqui-speaker", type=str, default=None)
    p.add_argument("--coqui-language", type=str, default=None)
    p.add_argument("--speaker-wav", type=str, default=None)

    # Torchaudio scoring controls.
    p.add_argument("--bundle", type=str, default="HUBERT_LARGE")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--skip-neural", action="store_true", help="No torchaudio embedding scoring. Candidate generation still works.")
    p.add_argument("--render-audio", action="store_true", help="Render candidate audio even when --skip-neural is set.")

    # Artifact guards.
    p.add_argument("--max-duration-sec", type=float, default=3.0)
    p.add_argument("--max-duration-ratio", type=float, default=3.0)
    p.add_argument("--allow-overlong", action="store_true")
    p.add_argument("--render-existing", action="store_true")

    args = p.parse_args()

    word = args.word.lower().strip()
    canonical = LEXICON[word]
    resolved_voice_mode = None if args.voice_mode == "auto" else args.voice_mode
    if resolved_voice_mode is None and args.tts_engine == "piper":
        resolved_voice_mode = infer_piper_voice_mode(args.piper_model)
    if resolved_voice_mode is None and args.tts_engine == "edge":
        resolved_voice_mode = resolve_edge_voice_mode(args)
    resolved_gender = gender_for_voice_mode(resolved_voice_mode)

    if args.levels is not None:
        levels = [i / max(1, args.levels - 1) for i in range(args.levels)]
    else:
        levels = [args.level]

    all_candidates: list[Candidate] = []
    for lvl in levels:
        all_candidates.extend(generate_phone_search_space(
            word=word,
            canonical=canonical,
            level=lvl,
            max_candidates=args.max_candidates,
            preserve_first_phone=True,
            ground_truth_level=args.ground_truth_level,
            include_ground_truth_each_level=not args.no_ground_truth_each_level,
        ))

    # Deduplicate by phone sequence and level.
    dedup = {}
    for c in all_candidates:
        dedup[(round(c.level, 4), tuple(c.phones))] = c
    base_candidates = list(dedup.values())
    level_summary = summarize_level_candidates(base_candidates, canonical)
    if not args.no_ground_truth_each_level:
        missing_ground_truth_levels = [
            entry["level"] for entry in level_summary if not entry["ground_truth_present"]
        ]
        if missing_ground_truth_levels:
            raise RuntimeError(
                "Canonical/base word is missing from one or more level buckets "
                f"despite ground-truth-each-level being enabled: {missing_ground_truth_levels}"
            )

    metadata: dict[str, Any] = {
        "word": word,
        "canonical_phones": canonical,
        "voice_mode": resolved_voice_mode,
        "gender": resolved_gender,
        "generation_mode": "global",
        "levels": levels,
        "level_summary": level_summary,
        "ground_truth_level": args.ground_truth_level,
        "ground_truth_each_level": not args.no_ground_truth_each_level,
        "ground_truth_policy": "canonical literal word is visible in every level bucket unless --no-ground-truth-each-level is set; rendered L0 ground-truth audio is reused at higher levels when available",
        "tts_engine": args.tts_engine,
        "kokoro_voices": [v.strip() for v in args.kokoro_voices.split(",") if v.strip()] if args.tts_engine == "kokoro" else None,
        "kokoro_lang_code": args.kokoro_lang_code if args.tts_engine == "kokoro" else None,
        "kokoro_speed": args.kokoro_speed if args.tts_engine == "kokoro" else None,
        "edge_voices": resolve_tts_voices(args) if args.tts_engine == "edge" else None,
        "edge_rate": args.edge_rate if args.tts_engine == "edge" else None,
        "edge_pitch": args.edge_pitch if args.tts_engine == "edge" else None,
        "piper_model": args.piper_model if args.tts_engine == "piper" else None,
        "piper_config": args.piper_config if args.tts_engine == "piper" else None,
        "piper_speaker": args.piper_speaker if args.tts_engine == "piper" else None,
        "piper_length_scale": args.piper_length_scale if args.tts_engine == "piper" else None,
        "coqui_model": args.coqui_model if args.tts_engine == "coqui" else None,
        "torchaudio_bundle": None if args.skip_neural else args.bundle,
        "max_duration_sec": args.max_duration_sec,
        "max_duration_ratio": args.max_duration_ratio,
        "reject_overlong": not args.allow_overlong,
        "mode": "render_only_or_rule_based" if args.skip_neural else "tts_plus_torchaudio_embedding_scoring",
    }

    should_render = args.render_audio or not args.skip_neural

    if should_render:
        render_dir = Path(args.render_dir)
        renderer = make_renderer(args)
        scorer = None if args.skip_neural else TorchAudioEmbeddingScorer(bundle_name=args.bundle, device=args.device)

        reference_wav = None
        if not args.skip_neural:
            reference_wav = synthesize_reference_if_needed(word, args.reference_wav, renderer, args, render_dir)
            metadata["reference_wav"] = str(reference_wav)

        scored: list[Candidate] = []
        ground_truth_audio_paths: dict[tuple[str, str], Path] = {}
        for lvl in levels:
            subset = [c for c in base_candidates if abs(c.level - lvl) < 1e-8]
            scored.extend(score_candidates_with_audio(
                candidates=subset,
                reference_wav=reference_wav,
                render_dir=render_dir / f"level_{lvl:.2f}",
                renderer=renderer,
                args=args,
                target_level=lvl,
                scorer=scorer,
                ground_truth_audio_paths=ground_truth_audio_paths,
            ))
        candidates = scored
    else:
        candidates = []
        for c in base_candidates:
            desired = c.level * 0.55
            level_match = 1.0 - min(1.0, abs(c.phone_distance - desired) / 0.55)
            c.suitability_score = 0.60 * c.phone_similarity + 0.40 * level_match
            candidates.append(c)
        candidates.sort(key=lambda c: (-(c.suitability_score or 0), c.grapheme))

    blabber_assets = [
        make_blabber_asset(
            c,
            canonical_phones=canonical,
            asset_index=i,
            voice_mode=resolved_voice_mode,
        )
        for i, c in enumerate(candidates)
    ]
    blabber_index = build_blabber_search_index(blabber_assets)

    payload = {
        **metadata,

        # Backward-compatible raw candidate list.
        "candidates": [asdict(c) for c in candidates],

        # build_blabber_library.py / select_blabber_asset.py compatible output.
        "blabber_schema_version": BLABBER_SCHEMA_VERSION,
        "blabber_assets": blabber_assets,
        "assets": blabber_assets,
        "search_index": blabber_index,

        # Convenience object for tools expecting a single library object.
        "library": {
            "schema_version": BLABBER_SCHEMA_VERSION,
            "word": word,
            "canonical_phones": canonical,
            "canonical_phone_key": phone_sequence_key(canonical),
            "voice_mode": resolved_voice_mode,
            "gender": resolved_gender,
            "generation_mode": "global",
            "assets": blabber_assets,
            "search_index": blabber_index,
        },
    }

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(payload, indent=2))
        print(json.dumps({
            "json_out": args.json_out,
            "candidate_count": len(candidates),
            "asset_count": len(blabber_assets),
            "level_summary": level_summary,
            "schema_version": BLABBER_SCHEMA_VERSION,
        }, indent=2))
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
