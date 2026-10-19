#!/usr/bin/env python3
"""
Neighbour-conditioned PanPhon triphone candidate ranker.

This version replaces hard-coded level-gated insertions like:
    Y at level >= 0.55
    W at level >= 0.65
    AH at level >= 0.70
    ER at level >= 0.80

with local phonological insertion rules:

    Vowel + Vowel              -> Y or W glide insertion
    Difficult C + C cluster    -> AX or AH vowel epenthesis
    Nasal + Fricative          -> homorganic stop insertion P/T/K
    Liquid/Rhotic boundary     -> AX or ER smoothing insertion

It does not render audio and does not output level folders. It only writes:
    1. ranked transcript TXT
    2. JSON containing original word, distorted word, phones, IPA,
       PanPhon feature distance, insertion reason, and articulatory features.

Install:
    pip install panphon

Example:
    python panphon_contextual_insertion_ranker.py \
      --words bath stop go yes no pain help food \
      --out-dir triphone-panphon-contextual \
      --bins 8
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

import panphon
import panphon.distance


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


ARPABET_TO_IPA: dict[str, str] = {
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "ɡ",
    "F": "f", "V": "v", "TH": "θ", "DH": "ð",
    "S": "s", "Z": "z", "SH": "ʃ", "ZH": "ʒ", "HH": "h",
    "CH": "tʃ", "JH": "dʒ",
    "M": "m", "N": "n", "NG": "ŋ",
    "L": "l", "R": "ɹ", "Y": "j", "W": "w",
    "IY": "i", "IH": "ɪ", "EH": "ɛ", "AE": "æ",
    "AA": "ɑ", "AO": "ɔ", "UW": "u", "UH": "ʊ",
    "AH": "ʌ", "AX": "ə", "ER": "əɹ",
    "EY": "eɪ", "AY": "aɪ", "AW": "aʊ", "OW": "ɔʊ", "OY": "ɔɪ",
}


PHONE_TO_TEXT: dict[str, str] = {
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "g",
    "F": "f", "V": "v", "TH": "th", "DH": "dh",
    "S": "s", "Z": "z", "SH": "sh", "ZH": "zh", "HH": "h",
    "CH": "ch", "JH": "j",
    "M": "m", "N": "n", "NG": "ng",
    "L": "l", "R": "r", "Y": "y", "W": "w",
    "IY": "ee", "IH": "ih", "EH": "eh", "AE": "a",
    "AA": "ah", "AO": "aw", "UW": "oo", "UH": "uh",
    "AH": "uh", "AX": "uh", "ER": "er",
    "EY": "ay", "AY": "eye", "AW": "ow", "OW": "oh", "OY": "oy",
}


# These tables are only used to build a pool of substitution candidates.
# Final ranking is PanPhon-based.
VOWEL_VARIANTS: dict[str, list[tuple[str, float]]] = {
    "OW": [("OW", 0.00), ("UW", 0.25), ("AH", 0.35), ("AO", 0.40), ("ER", 0.55)],
    "UW": [("UW", 0.00), ("UH", 0.20), ("AH", 0.35), ("AO", 0.45), ("ER", 0.55)],
    "EY": [("EY", 0.00), ("EH", 0.25), ("AE", 0.35), ("AH", 0.45), ("ER", 0.55)],
    "EH": [("EH", 0.00), ("AE", 0.20), ("AH", 0.35), ("ER", 0.55)],
    "AE": [("AE", 0.00), ("EH", 0.20), ("AA", 0.30), ("AH", 0.35), ("ER", 0.55)],
    "AA": [("AA", 0.00), ("AO", 0.25), ("AH", 0.30), ("AE", 0.45), ("ER", 0.55)],
    "AO": [("AO", 0.00), ("AA", 0.25), ("OW", 0.35), ("AH", 0.40), ("ER", 0.55)],
    "AH": [("AH", 0.00), ("AX", 0.10), ("AA", 0.30), ("EH", 0.35), ("ER", 0.45)],
    "ER": [("ER", 0.00), ("AH", 0.45), ("AX", 0.45)],
    "IH": [("IH", 0.00), ("IY", 0.20), ("EH", 0.25), ("AH", 0.45)],
    "IY": [("IY", 0.00), ("IH", 0.20), ("EY", 0.35), ("EH", 0.45)],
    "UH": [("UH", 0.00), ("UW", 0.20), ("AH", 0.35)],
    "AX": [("AX", 0.00), ("AH", 0.10), ("ER", 0.45)],
}

CONSONANT_VARIANTS: dict[str, list[tuple[str, float]]] = {
    "P": [("P", 0.00), ("B", 0.20), ("F", 0.35), ("M", 0.45)],
    "B": [("B", 0.00), ("P", 0.20), ("V", 0.35), ("M", 0.45)],
    "T": [("T", 0.00), ("D", 0.20), ("S", 0.35), ("TH", 0.45)],
    "D": [("D", 0.00), ("T", 0.20), ("Z", 0.35), ("DH", 0.45)],
    "K": [("K", 0.00), ("G", 0.20), ("HH", 0.45)],
    "G": [("G", 0.00), ("K", 0.20), ("HH", 0.45)],
    "F": [("F", 0.00), ("V", 0.20), ("P", 0.35), ("TH", 0.40)],
    "V": [("V", 0.00), ("F", 0.20), ("B", 0.35), ("DH", 0.40)],
    "TH": [("TH", 0.00), ("DH", 0.20), ("F", 0.40), ("S", 0.40), ("T", 0.45)],
    "DH": [("DH", 0.00), ("TH", 0.20), ("V", 0.40), ("Z", 0.40), ("D", 0.45)],
    "S": [("S", 0.00), ("Z", 0.20), ("SH", 0.30), ("TH", 0.40), ("T", 0.45)],
    "Z": [("Z", 0.00), ("S", 0.20), ("ZH", 0.30), ("DH", 0.40), ("D", 0.45)],
    "SH": [("SH", 0.00), ("ZH", 0.20), ("S", 0.30), ("CH", 0.35)],
    "ZH": [("ZH", 0.00), ("SH", 0.20), ("Z", 0.30), ("JH", 0.35)],
    "HH": [("HH", 0.00), ("K", 0.45), ("G", 0.45)],
    "CH": [("CH", 0.00), ("JH", 0.20), ("SH", 0.35), ("T", 0.40)],
    "JH": [("JH", 0.00), ("CH", 0.20), ("ZH", 0.35), ("D", 0.40)],
    "M": [("M", 0.00), ("N", 0.35), ("B", 0.45), ("P", 0.45)],
    "N": [("N", 0.00), ("M", 0.35), ("NG", 0.35), ("D", 0.45), ("T", 0.45)],
    "NG": [("NG", 0.00), ("N", 0.35), ("G", 0.45), ("K", 0.45)],
    "L": [("L", 0.00), ("R", 0.30), ("W", 0.45), ("Y", 0.50)],
    "R": [("R", 0.00), ("L", 0.30), ("W", 0.40), ("ER", 0.45)],
    "Y": [("Y", 0.00), ("W", 0.40), ("L", 0.50)],
    "W": [("W", 0.00), ("Y", 0.40), ("R", 0.40), ("L", 0.45)],
}


VOWELS = set(VOWEL_VARIANTS)
NASALS = {"M", "N", "NG"}
FRICATIVES = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
STOPS = {"P", "B", "T", "D", "K", "G"}
AFFRICATES = {"CH", "JH"}
LIQUIDS = {"L", "R"}
GLIDES = {"Y", "W"}
OBSTRUENTS = STOPS | FRICATIVES | AFFRICATES

# Coarse sonority classes for cluster difficulty.
# Higher means more sonorous.
SONORITY = {
    **{p: 1 for p in STOPS},
    **{p: 2 for p in FRICATIVES | AFFRICATES},
    **{p: 3 for p in NASALS},
    **{p: 4 for p in LIQUIDS},
    **{p: 5 for p in GLIDES},
    **{p: 6 for p in VOWELS},
}

# Common English onset clusters that should not automatically be treated as difficult.
COMMON_ONSET_CLUSTERS = {
    ("S", "T"), ("S", "P"), ("S", "K"),
    ("P", "L"), ("P", "R"), ("B", "L"), ("B", "R"),
    ("T", "R"), ("D", "R"),
    ("K", "L"), ("K", "R"), ("G", "L"), ("G", "R"),
    ("F", "L"), ("F", "R"),
    ("TH", "R"),
    ("S", "L"), ("S", "M"), ("S", "N"), ("S", "W"),
    ("K", "W"), ("G", "W"),
}

# Common coda-like clusters in the command words should be mildly protected.
COMMON_CODA_CLUSTERS = {
    ("L", "P"),  # help
    ("N", "D"), ("N", "T"),
    ("M", "P"), ("M", "B"),
    ("S", "T"),
}


INSERTION_PENALTY = {
    "AX": 0.15,
    "AH": 0.20,
    "Y": 0.20,
    "W": 0.20,
    "ER": 0.30,
    "P": 0.25,
    "T": 0.25,
    "K": 0.25,
}


def all_phones_used_by_generator() -> set[str]:
    """Return every ARPABET phone that can appear in generated candidates."""
    phones: set[str] = set()

    for seq in LEXICON.values():
        phones.update(seq)

    for table in (VOWEL_VARIANTS, CONSONANT_VARIANTS):
        phones.update(table.keys())
        for variants in table.values():
            for phone, _distance in variants:
                phones.add(phone)

    phones.update(NASALS)
    phones.update(FRICATIVES)
    phones.update(STOPS)
    phones.update(AFFRICATES)
    phones.update(LIQUIDS)
    phones.update(GLIDES)

    for cluster in COMMON_ONSET_CLUSTERS | COMMON_CODA_CLUSTERS:
        phones.update(cluster)

    phones.update(INSERTION_PENALTY.keys())

    return phones


def validate_ipa_inventory() -> None:
    """
    Check that every ARPABET phone the generator can emit has:
      1. an ARPABET_TO_IPA mapping,
      2. a PHONE_TO_TEXT mapping,
      3. a PanPhon-readable IPA representation.

    This catches issues such as PanPhon not recognising plain IPA "o" in "oʊ".
    """
    phones = all_phones_used_by_generator()

    missing_ipa = sorted(phones - set(ARPABET_TO_IPA))
    missing_text = sorted(phones - set(PHONE_TO_TEXT))

    if missing_ipa:
        raise ValueError(
            "Missing ARPABET_TO_IPA mappings for: " + ", ".join(missing_ipa)
        )

    if missing_text:
        raise ValueError(
            "Missing PHONE_TO_TEXT mappings for: " + ", ".join(missing_text)
        )

    ft = panphon.FeatureTable()
    distance = panphon.distance.Distance()

    bad: list[str] = []
    for phone in sorted(phones):
        ipa = ARPABET_TO_IPA[phone]
        try:
            vectors = ft.word_to_vector_list(ipa, numeric=True)
            _ = distance.feature_edit_distance(ipa, ipa)
        except Exception as exc:
            bad.append(f"{phone}->{ipa} ({type(exc).__name__}: {exc})")
            continue

        if not vectors:
            bad.append(f"{phone}->{ipa} (no PanPhon feature vectors)")

    if bad:
        raise ValueError(
            "These IPA mappings are not recognised by PanPhon:\n  "
            + "\n  ".join(bad)
        )


def phone_to_ipa(phone: str) -> str:
    if phone not in ARPABET_TO_IPA:
        raise ValueError(f"No IPA mapping for ARPABET phone: {phone}")
    return ARPABET_TO_IPA[phone]


def phones_to_ipa_string(phones: list[str]) -> str:
    return "".join(phone_to_ipa(p) for p in phones)


def phone_class(phone: str) -> str:
    if phone in VOWELS:
        return "vowel"
    if phone in NASALS:
        return "nasal"
    if phone in LIQUIDS:
        return "liquid"
    if phone in GLIDES:
        return "glide"
    if phone in STOPS:
        return "stop"
    if phone in FRICATIVES:
        return "fricative"
    if phone in AFFRICATES:
        return "affricate"
    return "unknown"


def is_vowel(phone: str) -> bool:
    return phone in VOWELS


def is_consonant(phone: str) -> bool:
    return not is_vowel(phone)


def is_front_or_high_vowel(phone: str) -> bool:
    # Rough English vowel classes for glide choice.
    return phone in {"IY", "IH", "EH", "AE", "EY", "AY"}


def is_back_or_rounded_vowel(phone: str) -> bool:
    return phone in {"UW", "UH", "OW", "AO", "OY", "AW", "AA"}


def homorganic_stop_for_nasal(nasal: str) -> str | None:
    if nasal == "M":
        return "P"
    if nasal == "N":
        return "T"
    if nasal == "NG":
        return "K"
    return None


def panphon_phone_distance(dst: Any, a: str, b: str, normalizer_per_phone: float = 24.0) -> float:
    raw = float(dst.feature_edit_distance(phone_to_ipa(a), phone_to_ipa(b)))
    return min(1.0, raw / normalizer_per_phone)


def panphon_sequence_distance(
    dst: Any,
    source_phones: list[str],
    candidate_phones: list[str],
    normalizer_per_phone: float = 24.0,
) -> tuple[float, float]:
    raw = float(dst.feature_edit_distance(
        phones_to_ipa_string(source_phones),
        phones_to_ipa_string(candidate_phones),
    ))
    denom = max(len(source_phones), len(candidate_phones), 1) * normalizer_per_phone
    return min(1.0, raw / denom), raw


def difficult_cluster(left: str, right: str) -> bool:
    """Simple sonority/phonotactic heuristic for whether C+C invites epenthesis."""
    if not (is_consonant(left) and is_consonant(right)):
        return False

    if (left, right) in COMMON_ONSET_CLUSTERS or (left, right) in COMMON_CODA_CLUSTERS:
        return False

    # Obstruent-obstruent clusters are articulatorily demanding.
    if left in OBSTRUENTS and right in OBSTRUENTS:
        return True

    # Avoid falling sonority where an onset-like sequence would be expected.
    # This is not syllabified, so use as a weak heuristic.
    left_s = SONORITY.get(left, 0)
    right_s = SONORITY.get(right, 0)
    if left_s >= right_s and left not in {"S"}:
        return True

    return False


def contextual_insertions(
    left: str,
    right: str,
    dst: Any,
    strict_panphon_filter: bool,
    normalizer_per_phone: float,
    margin: float,
) -> list[dict[str, Any]]:
    """Return insertions licensed by the immediate left/right phone context."""
    proposed: list[dict[str, Any]] = []

    # 1. Vowel-vowel hiatus: insert a glide chosen by vowel quality.
    if is_vowel(left) and is_vowel(right):
        if is_front_or_high_vowel(left) or is_front_or_high_vowel(right):
            proposed.append({
                "phone": "Y",
                "reason": "vowel_vowel_hiatus_front_or_high_glide",
                "context": "V_V",
            })
        if is_back_or_rounded_vowel(left) or is_back_or_rounded_vowel(right):
            proposed.append({
                "phone": "W",
                "reason": "vowel_vowel_hiatus_back_or_rounded_glide",
                "context": "V_V",
            })

    # 2. Difficult consonant cluster: insert schwa-like vowel.
    if is_consonant(left) and is_consonant(right) and difficult_cluster(left, right):
        proposed.append({
            "phone": "AX",
            "reason": "difficult_consonant_cluster_schwa_epenthesis",
            "context": "C_C",
        })
        proposed.append({
            "phone": "AH",
            "reason": "difficult_consonant_cluster_reduced_vowel_epenthesis",
            "context": "C_C",
        })

    # 3. Nasal-fricative: insert a homorganic stop.
    if left in NASALS and right in FRICATIVES:
        stop = homorganic_stop_for_nasal(left)
        if stop is not None:
            proposed.append({
                "phone": stop,
                "reason": "nasal_fricative_homorganic_excrescent_stop",
                "context": "nasal_fricative",
            })

    # 4. Liquid/rhotic smoothing: use AX near liquids, ER only near R.
    if left in LIQUIDS or right in LIQUIDS:
        proposed.append({
            "phone": "AX",
            "reason": "liquid_boundary_schwa_smoothing",
            "context": "liquid_boundary",
        })
        if left == "R" or right == "R":
            proposed.append({
                "phone": "ER",
                "reason": "rhotic_boundary_er_colouring",
                "context": "rhotic_boundary",
            })

    # Dedupe by phone+reason.
    deduped = []
    seen = set()
    for item in proposed:
        key = (item["phone"], item["reason"])
        if key not in seen:
            seen.add(key)
            deduped.append(item)

    # Optional PanPhon plausibility filter:
    # accept if L-X-R is not much worse than L-R after adding a small insertion penalty.
    if not strict_panphon_filter:
        return deduped

    accepted = []
    direct = panphon_phone_distance(dst, left, right, normalizer_per_phone)
    for item in deduped:
        x = item["phone"]
        via = (
            panphon_phone_distance(dst, left, x, normalizer_per_phone)
            + panphon_phone_distance(dst, x, right, normalizer_per_phone)
            + INSERTION_PENALTY.get(x, 0.25)
        )
        item = dict(item)
        item["direct_transition_distance"] = direct
        item["inserted_transition_distance"] = via
        item["insertion_penalty"] = INSERTION_PENALTY.get(x, 0.25)

        if via <= direct + margin:
            accepted.append(item)

    return accepted


def level_values(bins: int) -> list[float]:
    if bins <= 1:
        return [1.0]
    return [round(i / (bins - 1), 4) for i in range(bins)]


def variant_options_for_phone(phone: str, max_phone_distance: float) -> list[str]:
    table = VOWEL_VARIANTS if phone in VOWEL_VARIANTS else CONSONANT_VARIANTS
    return [p for p, d in table.get(phone, [(phone, 0.0)]) if d <= max_phone_distance]


def generate_substitution_bases(
    canonical: list[str],
    bins: int,
    preserve_first_phone: bool,
) -> list[list[str]]:
    bases = set()

    for level in level_values(bins):
        max_phone_distance = 0.05 + 0.65 * level
        per_phone_options = []
        for idx, phone in enumerate(canonical):
            if preserve_first_phone and idx == 0:
                per_phone_options.append([phone])
            else:
                per_phone_options.append(variant_options_for_phone(phone, max_phone_distance))

        for combo in itertools.product(*per_phone_options):
            bases.add(tuple(combo))

    return [list(x) for x in sorted(bases)]


def generate_contextual_insertion_variants(
    base: list[str],
    dst: Any,
    max_insertions_per_candidate: int,
    strict_panphon_filter: bool,
    normalizer_per_phone: float,
    insertion_margin: float,
) -> list[tuple[list[str], list[dict[str, Any]]]]:
    """
    Generate candidates by inserting at most max_insertions_per_candidate phones
    across neighbour boundaries.
    """
    variants: list[tuple[list[str], list[dict[str, Any]]]] = [(base, [])]

    boundary_options: list[tuple[int, list[dict[str, Any]]]] = []
    for boundary_idx in range(len(base) - 1):
        left, right = base[boundary_idx], base[boundary_idx + 1]
        options = contextual_insertions(
            left=left,
            right=right,
            dst=dst,
            strict_panphon_filter=strict_panphon_filter,
            normalizer_per_phone=normalizer_per_phone,
            margin=insertion_margin,
        )
        if options:
            boundary_options.append((boundary_idx, options))

    # Single insertions.
    for boundary_idx, options in boundary_options:
        for option in options:
            cand = base[:boundary_idx + 1] + [option["phone"]] + base[boundary_idx + 1:]
            meta = dict(option)
            meta["boundary_index"] = boundary_idx
            meta["left_phone"] = base[boundary_idx]
            meta["right_phone"] = base[boundary_idx + 1]
            variants.append((cand, [meta]))

    # Multiple insertions at distinct boundaries.
    if max_insertions_per_candidate >= 2 and len(boundary_options) >= 2:
        for r in range(2, min(max_insertions_per_candidate, len(boundary_options)) + 1):
            for boundary_combo in itertools.combinations(boundary_options, r):
                # Use one insertion option per selected boundary.
                option_lists = [opts for _, opts in boundary_combo]
                for chosen_options in itertools.product(*option_lists):
                    # Insert from right to left so indices stay stable.
                    cand = list(base)
                    metas = []
                    for (boundary_idx, _), option in sorted(
                        zip(boundary_combo, chosen_options),
                        key=lambda pair: pair[0][0],
                        reverse=True,
                    ):
                        cand = cand[:boundary_idx + 1] + [option["phone"]] + cand[boundary_idx + 1:]
                        meta = dict(option)
                        meta["boundary_index"] = boundary_idx
                        meta["left_phone"] = base[boundary_idx]
                        meta["right_phone"] = base[boundary_idx + 1]
                        metas.append(meta)
                    metas.reverse()
                    variants.append((cand, metas))

    return variants


def phones_to_pseudoword(phones: list[str]) -> str:
    return "".join(PHONE_TO_TEXT.get(p, p.lower()) for p in phones)


def get_feature_vectors(phones: list[str]) -> list[dict[str, Any]]:
    ft = panphon.FeatureTable()
    names = list(ft.names)

    output = []
    for phone in phones:
        ipa = phone_to_ipa(phone)
        vectors = ft.word_to_vector_list(ipa, numeric=True)
        output.append({
            "phone": phone,
            "ipa": ipa,
            "panphon_feature_names": names,
            "segment_feature_vectors": [
                {name: int(value) for name, value in zip(names, vec)}
                for vec in vectors
            ],
        })

    return output


def make_candidate_record(
    word: str,
    canonical: list[str],
    candidate: list[str],
    dst: Any,
    insertion_events: list[dict[str, Any]],
    normalizer_per_phone: float,
) -> dict[str, Any]:
    norm_dist, raw_dist = panphon_sequence_distance(
        dst,
        canonical,
        candidate,
        normalizer_per_phone=normalizer_per_phone,
    )

    distorted_word = word if candidate == canonical else phones_to_pseudoword(candidate)

    return {
        "original_word": word,
        "distorted_word": distorted_word,
        "canonical_phones": canonical,
        "candidate_phones": candidate,
        "canonical_ipa": phones_to_ipa_string(canonical),
        "candidate_ipa": phones_to_ipa_string(candidate),
        "phonetic_distance": norm_dist,
        "panphon_raw_feature_edit_distance": raw_dist,
        "insertion_events": insertion_events,
        "articulatory_features": {
            "canonical": get_feature_vectors(canonical),
            "candidate": get_feature_vectors(candidate),
        },
    }


def generate_candidates_for_word(
    word: str,
    dst: Any,
    bins: int,
    preserve_first_phone: bool,
    max_insertions_per_candidate: int,
    max_candidates_per_word: int,
    strict_panphon_filter: bool,
    normalizer_per_phone: float,
    insertion_margin: float,
) -> list[dict[str, Any]]:
    if word not in LEXICON:
        raise ValueError(f"Unknown word {word!r}. Known words: {sorted(LEXICON)}")

    canonical = LEXICON[word]
    unique: dict[tuple[str, ...], dict[str, Any]] = {}

    bases = generate_substitution_bases(
        canonical=canonical,
        bins=bins,
        preserve_first_phone=preserve_first_phone,
    )

    for base in bases:
        for candidate, insertion_events in generate_contextual_insertion_variants(
            base=base,
            dst=dst,
            max_insertions_per_candidate=max_insertions_per_candidate,
            strict_panphon_filter=strict_panphon_filter,
            normalizer_per_phone=normalizer_per_phone,
            insertion_margin=insertion_margin,
        ):
            key = tuple(candidate)
            record = make_candidate_record(
                word=word,
                canonical=canonical,
                candidate=candidate,
                dst=dst,
                insertion_events=insertion_events,
                normalizer_per_phone=normalizer_per_phone,
            )

            prev = unique.get(key)
            if prev is None or record["phonetic_distance"] < prev["phonetic_distance"]:
                unique[key] = record

    ranked = sorted(
        unique.values(),
        key=lambda item: (
            item["phonetic_distance"],
            len(item["candidate_phones"]),
            item["distorted_word"],
            item["candidate_phones"],
        ),
    )
    return ranked[:max_candidates_per_word]


def write_transcript(path: Path, all_candidates: dict[str, list[dict[str, Any]]]) -> None:
    lines = []
    for word, candidates in all_candidates.items():
        lines.append(f"# {word}")
        lines.append("")
        for idx, c in enumerate(candidates):
            phones = " ".join(c["candidate_phones"])
            reasons = ",".join(e["reason"] for e in c["insertion_events"]) or "-"
            lines.append(
                f"{idx:04d}\t"
                f"{c['phonetic_distance']:.6f}\t"
                f"{c['original_word']} -> {c['distorted_word']}\t"
                f"/{c['candidate_ipa']}/\t"
                f"[{phones}]\t"
                f"insertions={reasons}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate PanPhon-ranked candidates with neighbour-conditioned insertions."
    )
    parser.add_argument(
        "--words",
        nargs="+",
        default=["bath", "stop", "go", "yes", "no", "pain", "help", "food"],
    )
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=Path("triphone-panphon-contextual"))
    parser.add_argument("--json-name", default="panphon_contextual_candidates.json")
    parser.add_argument("--transcript-name", default="panphon_contextual_transcript.txt")
    parser.add_argument("--normalizer-per-phone", type=float, default=24.0)
    parser.add_argument("--max-candidates-per-word", type=int, default=50)
    parser.add_argument("--max-insertions-per-candidate", type=int, default=2)
    parser.add_argument(
        "--do-not-preserve-first-phone",
        action="store_true",
        help="Allow first phone to mutate. Default preserves first phone.",
    )
    parser.add_argument(
        "--strict-panphon-filter",
        action="store_true",
        help=(
            "Only keep insertion events where L-X-R is not much worse than L-R "
            "after insertion penalty. Off by default because epenthesis can be "
            "phonotactically motivated even when it increases edit distance."
        ),
    )
    parser.add_argument(
        "--insertion-margin",
        type=float,
        default=0.10,
        help="Margin used only when --strict-panphon-filter is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    validate_ipa_inventory()
    dst = panphon.distance.Distance()

    all_candidates = {}
    for word in args.words:
        all_candidates[word] = generate_candidates_for_word(
            word=word,
            dst=dst,
            bins=args.bins,
            preserve_first_phone=not args.do_not_preserve_first_phone,
            max_insertions_per_candidate=args.max_insertions_per_candidate,
            max_candidates_per_word=args.max_candidates_per_word,
            strict_panphon_filter=args.strict_panphon_filter,
            normalizer_per_phone=args.normalizer_per_phone,
            insertion_margin=args.insertion_margin,
        )

    payload = {
        "metadata": {
            "description": (
                "PanPhon-ranked candidate transcript using neighbour-conditioned "
                "insertion rules rather than hard-coded level-gated insertions."
            ),
            "words": args.words,
            "bins_used_for_substitution_generation_only": args.bins,
            "preserve_first_phone": not args.do_not_preserve_first_phone,
            "max_insertions_per_candidate": args.max_insertions_per_candidate,
            "strict_panphon_filter": args.strict_panphon_filter,
            "insertion_margin": args.insertion_margin,
            "normalizer_per_phone": args.normalizer_per_phone,
            "insertion_rules": {
                "V_V": ["Y for front/high vowel context", "W for back/rounded vowel context"],
                "difficult_C_C": ["AX", "AH"],
                "nasal_fricative": ["homorganic P/T/K"],
                "liquid_boundary": ["AX"],
                "rhotic_boundary": ["ER"],
            },
            "arpabet_to_ipa": ARPABET_TO_IPA,
        },
        "candidates_by_word": all_candidates,
    }

    json_path = args.out_dir / args.json_name
    transcript_path = args.out_dir / args.transcript_name

    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_transcript(transcript_path, all_candidates)

    total = sum(len(v) for v in all_candidates.values())
    print(f"Wrote {total} candidates")
    print(f"JSON:       {json_path}")
    print(f"Transcript: {transcript_path}")


if __name__ == "__main__":
    main()

