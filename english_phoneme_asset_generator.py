import argparse
import asyncio
import csv
import html
import json
import re
import shutil
import subprocess
import sys
import tempfile
import wave
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

texttospeech = None

# -----------------------------------------------------------------------------
# Complete-ish English phoneme asset generator for accent-specific TTS assets.
#
# Outputs:
#   english_phoneme_assets/
#     manifest.csv
#     en_AU/female/consonants/*.wav
#     en_AU/female/vowels/*.wav
#     ...
#
# Notes:
# - This is an asset generator, not a phonetic ground-truth corpus.
# - TTS may not perfectly obey every IPA override. Spot-check the manifest.
# - For EEG/TTS experiments, prefer word/carrier assets over isolated consonants.
# -----------------------------------------------------------------------------

DEFAULT_OUTPUT_DIR = Path("english_phoneme_assets")
DEFAULT_AUDIO_FORMAT = "wav"  # wav is better than mp3 for analysis workflows
DEFAULT_SPEAKING_RATE = 0.9
DEFAULT_CROP_PADDING_SECONDS = 0.02
DEFAULT_AUTO_CLIP_PADDING_SECONDS = 0.015
ASSET_PROFILE_EXHAUSTIVE = "exhaustive"
ASSET_PROFILE_RICH_UNIQUE = "rich_unique"
ASSET_PROFILES = (ASSET_PROFILE_RICH_UNIQUE, ASSET_PROFILE_EXHAUSTIVE)
TTS_ENGINES = ("google", "edge", "piper", "kokoro", "coqui")

RICH_UNIQUE_CONTEXTS = {
    "consonants": {"word", "initial_syllable", "medial_syllable", "final_syllable", "minimal_pair"},
    "vowels": {"word", "hvd", "minimal_pair"},
}

# -----------------------------------------------------------------------------
# Voices
# -----------------------------------------------------------------------------

VOICES = {
    "en_AU": {
        "language_code": "en-AU",
        "female": "en-AU-Neural2-A",
        "male": "en-AU-Neural2-B",
    },
    "en_GB": {
        "language_code": "en-GB",
        "female": "en-GB-Neural2-A",
        "male": "en-GB-Neural2-B",
    },
    "en_US": {
        "language_code": "en-US",
        "female": "en-US-Neural2-F",
        "male": "en-US-Neural2-D",
    },
}

EDGE_VOICES = {
    "en_AU": {
        "female": "en-AU-NatashaNeural",
        "male": "en-AU-WilliamNeural",
    },
    "en_GB": {
        "female": "en-GB-SoniaNeural",
        "male": "en-GB-RyanNeural",
    },
    "en_US": {
        "female": "en-US-JennyNeural",
        "male": "en-US-GuyNeural",
    },
}

# -----------------------------------------------------------------------------
# Accent-specific IPA inventories
# -----------------------------------------------------------------------------

COMMON_CONSONANTS = [
    ("p", "p_stop", "pat"),
    ("b", "b_stop", "bat"),
    ("t", "t_stop", "tap"),
    ("d", "d_stop", "dad"),
    ("k", "k_stop", "cat"),
    ("g", "g_stop", "gap"),
    ("f", "f_fricative", "fat"),
    ("v", "v_fricative", "vat"),
    ("θ", "th_unvoiced", "thin"),
    ("ð", "th_voiced", "then"),
    ("s", "s_fricative", "sat"),
    ("z", "z_fricative", "zap"),
    ("ʃ", "sh_fricative", "shoe"),
    ("ʒ", "zh_fricative", "measure"),
    ("h", "h_fricative", "hat"),
    ("tʃ", "ch_affricate", "chin"),
    ("dʒ", "j_affricate", "jam"),
    ("m", "m_nasal", "mat"),
    ("n", "n_nasal", "net"),
    ("ŋ", "ng_nasal", "sing"),
    ("w", "w_approximant", "wet"),
    ("l", "l_approximant", "let"),
    ("r", "r_approximant", "red"),
    ("j", "y_approximant", "yet"),
]

INVENTORIES = {
    "en_AU": {
        "consonants": COMMON_CONSONANTS,
        "vowels": [
            ("ɪ", "kit", "kit"),
            ("e", "dress", "dress"),
            ("æ", "trap", "trap"),
            ("ɐ", "strut", "strut"),
            ("ɒ", "lot", "lot"),
            ("ʊ", "foot", "foot"),
            ("ə", "schwa", "about"),
            ("iː", "fleece", "fleece"),
            ("eː", "square", "square"),
            ("æː", "bath", "bath"),
            ("ɐː", "start", "start"),
            ("oː", "thought", "thought"),
            ("ʉː", "goose", "goose"),
            ("ɜː", "nurse", "nurse"),
            ("æɪ", "face", "face"),
            ("ɑɪ", "price", "price"),
            ("oɪ", "choice", "choice"),
            ("æɔ", "mouth", "mouth"),
            ("əʉ", "goat", "goat"),
            ("ɪə", "near", "near"),
            ("ʊə", "cure", "cure"),
        ],
    },
    "en_GB": {
        "consonants": COMMON_CONSONANTS,
        "vowels": [
            ("ɪ", "kit", "kit"),
            ("e", "dress", "dress"),
            ("æ", "trap", "trap"),
            ("ʌ", "strut", "strut"),
            ("ɒ", "lot", "lot"),
            ("ʊ", "foot", "foot"),
            ("ə", "schwa", "about"),
            ("iː", "fleece", "fleece"),
            ("ɑː", "start", "start"),
            ("ɔː", "thought", "thought"),
            ("uː", "goose", "goose"),
            ("ɜː", "nurse", "nurse"),
            ("eɪ", "face", "face"),
            ("aɪ", "price", "price"),
            ("ɔɪ", "choice", "choice"),
            ("aʊ", "mouth", "mouth"),
            ("əʊ", "goat", "goat"),
            ("ɪə", "near", "near"),
            ("eə", "square", "square"),
            ("ʊə", "cure", "cure"),
        ],
    },
    "en_US": {
        "consonants": COMMON_CONSONANTS,
        "vowels": [
            ("ɪ", "kit", "kit"),
            ("ɛ", "dress", "dress"),
            ("æ", "trap", "trap"),
            ("ʌ", "strut", "strut"),
            ("ɑ", "lot_father", "lot"),
            ("ɔ", "thought", "thought"),
            ("ʊ", "foot", "foot"),
            ("ə", "schwa", "about"),
            ("i", "fleece", "fleece"),
            ("u", "goose", "goose"),
            ("ɚ", "letter_r_colored", "letter"),
            ("ɝ", "nurse_r_colored", "nurse"),
            ("eɪ", "face", "face"),
            ("aɪ", "price", "price"),
            ("ɔɪ", "choice", "choice"),
            ("aʊ", "mouth", "mouth"),
            ("oʊ", "goat", "goat"),
        ],
    },
}

# -----------------------------------------------------------------------------
# ARPAbet reference labels for CMUdict-style compatibility.
# -----------------------------------------------------------------------------

IPA_TO_ARPABET = {
    "p": "P", "b": "B", "t": "T", "d": "D", "k": "K", "g": "G",
    "f": "F", "v": "V", "θ": "TH", "ð": "DH", "s": "S", "z": "Z",
    "ʃ": "SH", "ʒ": "ZH", "h": "HH", "tʃ": "CH", "dʒ": "JH",
    "m": "M", "n": "N", "ŋ": "NG", "w": "W", "l": "L", "r": "R", "j": "Y",
    "ɪ": "IH", "i": "IY", "iː": "IY", "e": "EH", "ɛ": "EH", "æ": "AE",
    "ɐ": "AH", "ʌ": "AH", "ɒ": "AA", "ɑ": "AA", "ɑː": "AA", "ɔ": "AO", "ɔː": "AO", "oː": "AO",
    "ʊ": "UH", "u": "UW", "uː": "UW", "ʉː": "UW", "ə": "AH", "ɚ": "ER", "ɝ": "ER", "ɜː": "ER",
    "eɪ": "EY", "æɪ": "EY", "aɪ": "AY", "ɑɪ": "AY", "ɔɪ": "OY", "oɪ": "OY",
    "aʊ": "AW", "æɔ": "AW", "oʊ": "OW", "əʊ": "OW", "əʉ": "OW",
    "ɪə": "IH_R", "eə": "EH_R", "ʊə": "UH_R", "eː": "EH_R", "æː": "AA", "ɐː": "AA",
}

ARPABET_TO_IPA = {
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "g",
    "F": "f", "V": "v", "TH": "θ", "DH": "ð", "S": "s", "Z": "z",
    "SH": "ʃ", "ZH": "ʒ", "HH": "h", "CH": "tʃ", "JH": "dʒ",
    "M": "m", "N": "n", "NG": "ŋ", "W": "w", "L": "l", "R": "r", "Y": "j",
    "IH": "ɪ", "IY": "i", "EH": "e", "AE": "æ", "AH": "ɐ", "AX": "ə",
    "AA": "ɐː", "AO": "oː", "UH": "ʊ", "UW": "ʉː", "ER": "ɜː",
    "EY": "æɪ", "AY": "ɑɪ", "OY": "oɪ", "AW": "æɔ", "OW": "əʉ",
    "LOT": "ɒ", "BATH": "æː", "SQUARE": "eː", "NEAR": "ɪə", "CURE": "ʊə",
}

DEFAULT_INVENTORY_VOICES = ("woman", "man")

# -----------------------------------------------------------------------------
# Filename-safe IPA mapping.
# Important: replace longer IPA sequences before shorter ones.
# -----------------------------------------------------------------------------

IPA_SAFE_REPLACEMENTS = {
    "tʃ": "ch", "dʒ": "j", "iː": "i_long", "eː": "e_long", "æː": "ae_long", "ɐː": "a_central_long",
    "ɑː": "aa_long", "ɔː": "aw_long", "uː": "u_long", "ʉː": "uw_central_long", "ɜː": "er_long",
    "eɪ": "ey", "æɪ": "ey_au", "aɪ": "ay", "ɑɪ": "ay_au", "ɔɪ": "oy", "oɪ": "oy_au",
    "aʊ": "aw_diph", "æɔ": "aw_diph_au", "oʊ": "ow", "əʊ": "ow_gb", "əʉ": "ow_au",
    "ɪə": "ih_r", "eə": "eh_r", "ʊə": "uh_r", "ɚ": "er_unstressed", "ɝ": "er_stressed",
    "ɪ": "ih", "ʊ": "uh", "ʉ": "uw_central", "ə": "schwa", "ɐ": "a_central", "æ": "ae",
    "ɑ": "aa", "ɒ": "o_rounded", "ɔ": "aw", "ɛ": "eh", "ɜ": "er", "ʌ": "uh_strut",
    "θ": "th_unvoiced", "ð": "th_voiced", "ʃ": "sh", "ʒ": "zh", "ŋ": "ng",
}


def safe_name(text: str) -> str:
    for symbol in sorted(IPA_SAFE_REPLACEMENTS, key=len, reverse=True):
        text = text.replace(symbol, "_" + IPA_SAFE_REPLACEMENTS[symbol] + "_")
    text = text.lower()
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_") or "blank"


def parse_csv_values(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def selected_arpabet_phones(value: Optional[str]) -> List[str]:
    phones = [phone.upper() for phone in parse_csv_values(value)] if value else sorted(ARPABET_TO_IPA)
    unknown = [phone for phone in phones if phone not in ARPABET_TO_IPA]
    if unknown:
        raise ValueError(f"Unsupported ARPABET phones: {unknown}. Supported: {sorted(ARPABET_TO_IPA)}")
    return phones


def copy_verified_inventory(args) -> None:
    """Copy an already-verified ARPABET inventory into a normalized layout."""
    if args.source_root is None:
        raise SystemExit("--copy-verified-inventory requires --source-root.")

    source_root = Path(args.source_root)
    output_root = Path(args.output_dir)
    phones = selected_arpabet_phones(args.phones)
    voices = parse_csv_values(args.inventory_voices) or list(DEFAULT_INVENTORY_VOICES)
    manifest = {
        "schema_version": "1.0-verified-arpabet-inventory",
        "mode": "copy_verified_inventory",
        "source_root": str(source_root),
        "output_root": str(output_root),
        "format": "wav",
        "phones": phones,
        "voices": voices,
        "assets": [],
    }

    copied = 0
    skipped = 0
    missing = 0
    for voice in voices:
        for phone in phones:
            source = source_root / voice / f"{phone}.wav"
            destination = output_root / voice / f"{phone}.wav"
            if not source.is_file():
                status = "missing_source"
                missing += 1
            elif destination.exists() and not args.overwrite:
                status = "exists"
                skipped += 1
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                status = "copied"
                copied += 1
            manifest["assets"].append(
                {
                    "phone": phone,
                    "ipa": ARPABET_TO_IPA[phone],
                    "voice": voice,
                    "source_path": str(source),
                    "audio_path": str(destination),
                    "status": status,
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "phoneme_inventory_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(
        {
            "manifest": str(manifest_path),
            "copied": copied,
            "existing": skipped,
            "missing": missing,
            "asset_count": len(manifest["assets"]),
        },
        indent=2,
        ensure_ascii=False,
    ))


def review_inventory_manifest(output_dir: Path) -> int:
    """Review a copied/generated inventory manifest for missing or unsafe assets."""
    manifest_path = output_dir / "phoneme_inventory_manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"Inventory manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    missing_assets = [
        asset
        for asset in manifest.get("assets", [])
        if not Path(str(asset.get("audio_path") or "")).is_file()
    ]
    unsafe = manifest.get("mode") in {"edge", "edge-ssml-unsafe"}
    summary = {
        "manifest": str(manifest_path),
        "mode": manifest.get("mode"),
        "unsafe": unsafe,
        "unsafe_reason": "edge_tts SSML can be spoken literally" if unsafe else None,
        "asset_count": len(manifest.get("assets", [])),
        "missing_count": len(missing_assets),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 2 if unsafe or missing_assets else 0

# -----------------------------------------------------------------------------
# Context tables
# Tuple format: rendered_ipa, prompt_text, natural_word
# - rendered_ipa: IPA passed to SSML ph=...
# - prompt_text: written prompt inside phoneme tag
# - natural_word: metadata-friendly human-readable target word
# -----------------------------------------------------------------------------

CONSONANT_CONTEXTS: Dict[str, Dict[str, Optional[Tuple[str, str, str]]]] = {
    "p": {"initial": ("pæ", "pa", "pat"), "medial": ("əpə", "upper", "upper"), "final": ("æp", "ap", "cap"), "minimal_pair": ("pæt", "pat", "pat")},
    "b": {"initial": ("bæ", "ba", "bat"), "medial": ("əbə", "about", "about"), "final": ("æb", "ab", "cab"), "minimal_pair": ("bæt", "bat", "bat")},
    "t": {"initial": ("tæ", "ta", "tap"), "medial": ("ətə", "utter", "utter"), "final": ("æt", "at", "cat"), "minimal_pair": ("tæp", "tap", "tap")},
    "d": {"initial": ("dæ", "da", "dad"), "medial": ("ədə", "udder", "udder"), "final": ("æd", "ad", "bad"), "minimal_pair": ("dæd", "dad", "dad")},
    "k": {"initial": ("kæ", "ka", "cat"), "medial": ("əkə", "anchor", "anchor"), "final": ("æk", "ack", "back"), "minimal_pair": ("kæt", "cat", "cat")},
    "g": {"initial": ("gæ", "ga", "gap"), "medial": ("əgə", "again", "again"), "final": ("æg", "ag", "bag"), "minimal_pair": ("gæp", "gap", "gap")},
    "f": {"initial": ("fæ", "fa", "fat"), "medial": ("əfə", "offer", "offer"), "final": ("æf", "af", "laugh"), "minimal_pair": ("fæt", "fat", "fat")},
    "v": {"initial": ("væ", "va", "vat"), "medial": ("əvə", "over", "over"), "final": ("æv", "av", "have"), "minimal_pair": ("væt", "vat", "vat")},
    "θ": {"initial": ("θɪ", "thi", "thin"), "medial": ("əθə", "ether", "ether"), "final": ("æθ", "ath", "bath"), "minimal_pair": ("θɪn", "thin", "thin")},
    "ð": {"initial": ("ðe", "the", "then"), "medial": ("əðə", "other", "other"), "final": ("eð", "eth", "breathe"), "minimal_pair": ("ðen", "then", "then")},
    "s": {"initial": ("sæ", "sa", "sat"), "medial": ("əsə", "essay", "essay"), "final": ("æs", "as", "bus"), "minimal_pair": ("sæt", "sat", "sat")},
    "z": {"initial": ("zæ", "za", "zap"), "medial": ("əzə", "busy", "busy"), "final": ("æz", "az", "buzz"), "minimal_pair": ("zæp", "zap", "zap")},
    "ʃ": {"initial": ("ʃuː", "shoo", "shoe"), "medial": ("əʃə", "washer", "washer"), "final": ("æʃ", "ash", "cash"), "minimal_pair": ("ʃuː", "shoe", "shoe")},
    "ʒ": {"initial": ("ʒɑ", "zha", "genre"), "medial": ("əʒə", "measure", "measure"), "final": ("ɑʒ", "azh", "garage"), "minimal_pair": ("ʒɑn", "genre", "genre")},
    "h": {"initial": ("hæ", "ha", "hat"), "medial": ("əhə", "ahead", "ahead"), "final": None, "minimal_pair": ("hæt", "hat", "hat")},
    "tʃ": {"initial": ("tʃɪ", "chi", "chin"), "medial": ("ətʃə", "teacher", "teacher"), "final": ("ætʃ", "ach", "match"), "minimal_pair": ("tʃɪn", "chin", "chin")},
    "dʒ": {"initial": ("dʒæ", "ja", "jam"), "medial": ("ədʒə", "major", "major"), "final": ("ædʒ", "adge", "judge"), "minimal_pair": ("dʒæm", "jam", "jam")},
    "m": {"initial": ("mæ", "ma", "mat"), "medial": ("əmə", "summer", "summer"), "final": ("æm", "am", "ham"), "minimal_pair": ("mæt", "mat", "mat")},
    "n": {"initial": ("næ", "na", "net"), "medial": ("ənə", "inner", "inner"), "final": ("æn", "an", "pan"), "minimal_pair": ("net", "net", "net")},
    "ŋ": {"initial": None, "medial": ("əŋə", "singer", "singer"), "final": ("ɪŋ", "ing", "sing"), "minimal_pair": ("sɪŋ", "sing", "sing")},
    "w": {"initial": ("wɛ", "we", "wet"), "medial": ("əwə", "away", "away"), "final": None, "minimal_pair": ("wet", "wet", "wet")},
    "l": {"initial": ("lɛ", "le", "let"), "medial": ("ələ", "allow", "allow"), "final": ("æl", "al", "feel"), "minimal_pair": ("let", "let", "let")},
    "r": {"initial": ("rɛ", "re", "red"), "medial": ("ərə", "around", "around"), "final": ("ɑr", "ar", "car"), "minimal_pair": ("red", "red", "red")},
    "j": {"initial": ("jɛ", "ye", "yet"), "medial": ("əjə", "beyond", "beyond"), "final": None, "minimal_pair": ("jet", "yet", "yet")},
}

VOWEL_CONTEXTS = {
    "en_AU": {
        "ɪ": {"word": ("ɪ", "kit"), "hvd": ("hɪd", "hid"), "minimal_pair": ("kɪt", "kit")},
        "e": {"word": ("e", "dress"), "hvd": ("hed", "head"), "minimal_pair": ("bed", "bed")},
        "æ": {"word": ("æ", "trap"), "hvd": ("hæd", "had"), "minimal_pair": ("kæt", "cat")},
        "ɐ": {"word": ("ɐ", "strut"), "hvd": ("hɐd", "hud"), "minimal_pair": ("kɐt", "cut")},
        "ɒ": {"word": ("ɒ", "lot"), "hvd": ("hɒd", "hod"), "minimal_pair": ("lɒt", "lot")},
        "ʊ": {"word": ("ʊ", "foot"), "hvd": ("hʊd", "hood"), "minimal_pair": ("fʊt", "foot")},
        "ə": {"word": ("ə", "about"), "hvd": ("həd", "heard"), "minimal_pair": ("əbæʉt", "about")},
        "iː": {"word": ("iː", "fleece"), "hvd": ("hiːd", "heed"), "minimal_pair": ("biːt", "beat")},
        "eː": {"word": ("eː", "square"), "hvd": ("heːd", "haired"), "minimal_pair": ("skeː", "square")},
        "æː": {"word": ("æː", "bath"), "hvd": ("hæːd", "hard"), "minimal_pair": ("bæːθ", "bath")},
        "ɐː": {"word": ("ɐː", "start"), "hvd": ("hɐːd", "hard"), "minimal_pair": ("stɐːt", "start")},
        "oː": {"word": ("oː", "thought"), "hvd": ("hoːd", "hoard"), "minimal_pair": ("toːt", "taught")},
        "ʉː": {"word": ("ʉː", "goose"), "hvd": ("hʉːd", "who'd"), "minimal_pair": ("gʉːs", "goose")},
        "ɜː": {"word": ("ɜː", "nurse"), "hvd": ("hɜːd", "heard"), "minimal_pair": ("nɜːs", "nurse")},
        "æɪ": {"word": ("æɪ", "face"), "hvd": ("hæɪd", "hayed"), "minimal_pair": ("fæɪs", "face")},
        "ɑɪ": {"word": ("ɑɪ", "price"), "hvd": ("hɑɪd", "hide"), "minimal_pair": ("prɑɪs", "price")},
        "oɪ": {"word": ("oɪ", "choice"), "hvd": ("hoɪd", "hoyed"), "minimal_pair": ("tʃoɪs", "choice")},
        "æɔ": {"word": ("æɔ", "mouth"), "hvd": ("hæɔd", "how'd"), "minimal_pair": ("mæɔθ", "mouth")},
        "əʉ": {"word": ("əʉ", "goat"), "hvd": ("həʉd", "hoed"), "minimal_pair": ("gəʉt", "goat")},
        "ɪə": {"word": ("ɪə", "near"), "hvd": ("hɪəd", "heared"), "minimal_pair": ("nɪə", "near")},
        "ʊə": {"word": ("ʊə", "cure"), "hvd": ("hʊəd", "hured"), "minimal_pair": ("kjʊə", "cure")},
    },
    "en_GB": {
        "ɪ": {"word": ("ɪ", "kit"), "hvd": ("hɪd", "hid"), "minimal_pair": ("kɪt", "kit")},
        "e": {"word": ("e", "dress"), "hvd": ("hed", "head"), "minimal_pair": ("bed", "bed")},
        "æ": {"word": ("æ", "trap"), "hvd": ("hæd", "had"), "minimal_pair": ("kæt", "cat")},
        "ʌ": {"word": ("ʌ", "strut"), "hvd": ("hʌd", "hud"), "minimal_pair": ("kʌt", "cut")},
        "ɒ": {"word": ("ɒ", "lot"), "hvd": ("hɒd", "hod"), "minimal_pair": ("lɒt", "lot")},
        "ʊ": {"word": ("ʊ", "foot"), "hvd": ("hʊd", "hood"), "minimal_pair": ("fʊt", "foot")},
        "ə": {"word": ("ə", "about"), "hvd": ("həd", "heard"), "minimal_pair": ("əbɑʊt", "about")},
        "iː": {"word": ("iː", "fleece"), "hvd": ("hiːd", "heed"), "minimal_pair": ("biːt", "beat")},
        "ɑː": {"word": ("ɑː", "start"), "hvd": ("hɑːd", "hard"), "minimal_pair": ("stɑːt", "start")},
        "ɔː": {"word": ("ɔː", "thought"), "hvd": ("hɔːd", "hoard"), "minimal_pair": ("tɔːt", "taught")},
        "uː": {"word": ("uː", "goose"), "hvd": ("huːd", "who'd"), "minimal_pair": ("guːs", "goose")},
        "ɜː": {"word": ("ɜː", "nurse"), "hvd": ("hɜːd", "heard"), "minimal_pair": ("nɜːs", "nurse")},
        "eɪ": {"word": ("eɪ", "face"), "hvd": ("heɪd", "hayed"), "minimal_pair": ("feɪs", "face")},
        "aɪ": {"word": ("aɪ", "price"), "hvd": ("haɪd", "hide"), "minimal_pair": ("praɪs", "price")},
        "ɔɪ": {"word": ("ɔɪ", "choice"), "hvd": ("hɔɪd", "hoyed"), "minimal_pair": ("tʃɔɪs", "choice")},
        "aʊ": {"word": ("aʊ", "mouth"), "hvd": ("haʊd", "how'd"), "minimal_pair": ("maʊθ", "mouth")},
        "əʊ": {"word": ("əʊ", "goat"), "hvd": ("həʊd", "hoed"), "minimal_pair": ("gəʊt", "goat")},
        "ɪə": {"word": ("ɪə", "near"), "hvd": ("hɪəd", "heared"), "minimal_pair": ("nɪə", "near")},
        "eə": {"word": ("eə", "square"), "hvd": ("heəd", "haired"), "minimal_pair": ("skeə", "square")},
        "ʊə": {"word": ("ʊə", "cure"), "hvd": ("hʊəd", "hured"), "minimal_pair": ("kjʊə", "cure")},
    },
    "en_US": {
        "ɪ": {"word": ("ɪ", "kit"), "hvd": ("hɪd", "hid"), "minimal_pair": ("kɪt", "kit")},
        "ɛ": {"word": ("ɛ", "dress"), "hvd": ("hɛd", "head"), "minimal_pair": ("bɛd", "bed")},
        "æ": {"word": ("æ", "trap"), "hvd": ("hæd", "had"), "minimal_pair": ("kæt", "cat")},
        "ʌ": {"word": ("ʌ", "strut"), "hvd": ("hʌd", "hud"), "minimal_pair": ("kʌt", "cut")},
        "ɑ": {"word": ("ɑ", "lot"), "hvd": ("hɑd", "hod"), "minimal_pair": ("lɑt", "lot")},
        "ɔ": {"word": ("ɔ", "thought"), "hvd": ("hɔd", "hawed"), "minimal_pair": ("θɔt", "thought")},
        "ʊ": {"word": ("ʊ", "foot"), "hvd": ("hʊd", "hood"), "minimal_pair": ("fʊt", "foot")},
        "ə": {"word": ("ə", "about"), "hvd": ("həd", "had"), "minimal_pair": ("əbaʊt", "about")},
        "i": {"word": ("i", "fleece"), "hvd": ("hid", "heed"), "minimal_pair": ("bit", "beat")},
        "u": {"word": ("u", "goose"), "hvd": ("hud", "who'd"), "minimal_pair": ("gus", "goose")},
        "ɚ": {"word": ("ɚ", "letter"), "hvd": ("hɚ", "her"), "minimal_pair": ("lɛtɚ", "letter")},
        "ɝ": {"word": ("ɝ", "nurse"), "hvd": ("hɝd", "heard"), "minimal_pair": ("nɝs", "nurse")},
        "eɪ": {"word": ("eɪ", "face"), "hvd": ("heɪd", "hayed"), "minimal_pair": ("feɪs", "face")},
        "aɪ": {"word": ("aɪ", "price"), "hvd": ("haɪd", "hide"), "minimal_pair": ("praɪs", "price")},
        "ɔɪ": {"word": ("ɔɪ", "choice"), "hvd": ("hɔɪd", "hoyed"), "minimal_pair": ("tʃɔɪs", "choice")},
        "aʊ": {"word": ("aʊ", "mouth"), "hvd": ("haʊd", "how'd"), "minimal_pair": ("maʊθ", "mouth")},
        "oʊ": {"word": ("oʊ", "goat"), "hvd": ("hoʊd", "hoed"), "minimal_pair": ("goʊt", "goat")},
    },
}

CONTEXT_POSITION = {
    "initial": "initial",
    "initial_syllable": "initial",
    "medial": "medial",
    "medial_syllable": "medial",
    "final": "final",
    "final_syllable": "final",
    "minimal_pair": "contrast",
    "minimal_pair_syllable": "contrast",
    "word": "lexical",
    "carrier_phrase": "lexical",
    "hvd": "vowel_nucleus",
}


def build_ssml(rendered_ipa: str, prompt_text: str, context: str) -> str:
    escaped_ipa = html.escape(rendered_ipa, quote=True)
    escaped_text = html.escape(prompt_text, quote=False)
    phoneme = f'<phoneme alphabet="ipa" ph="{escaped_ipa}">{escaped_text}</phoneme>'
    if context == "carrier_phrase":
        return f"<speak>Say {phoneme} again.</speak>"
    return f"<speak>{phoneme}</speak>"


def audio_config_for_format(audio_format: str, speaking_rate: float):
    if texttospeech is None:
        raise RuntimeError("google-cloud-texttospeech is required for asset generation, but not for --crop-phone.")
    if audio_format == "mp3":
        return texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate,
        )
    if audio_format == "wav":
        return texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=speaking_rate,
        )
    raise ValueError(f"Unsupported audio format: {audio_format}")


def synthesize(client, ssml: str, language_code: str, voice_name: str, filepath: Path, audio_format: str, speaking_rate: float, retries: int = 2):
    synthesis_input = texttospeech.SynthesisInput(ssml=ssml)
    voice = texttospeech.VoiceSelectionParams(language_code=language_code, name=voice_name)
    audio_config = audio_config_for_format(audio_format, speaking_rate)

    last_error = None
    for attempt in range(retries + 1):
        try:
            response = client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_bytes(response.audio_content)
            return
        except Exception as exc:
            last_error = exc
            time.sleep(0.75 * (attempt + 1))
    raise last_error


def normalize_render_text(text: str) -> str:
    normalized = " ".join(str(text).strip().split())
    if not normalized:
        raise ValueError("Cannot render empty text.")
    return normalized


def prepare_sentence_text(text: str) -> str:
    normalized = normalize_render_text(text)
    if normalized[-1] not in ".!?":
        normalized += "."
    return normalized


class GoogleRenderer:
    def __init__(self, audio_format: str, speaking_rate: float):
        global texttospeech
        from google.cloud import texttospeech as google_texttospeech
        texttospeech = google_texttospeech
        self.client = texttospeech.TextToSpeechClient()
        self.audio_format = audio_format
        self.speaking_rate = speaking_rate

    def render(self, *, ssml: str, text: str, language_code: str, voice_name: str, filepath: Path) -> None:
        synthesize(
            client=self.client,
            ssml=ssml,
            language_code=language_code,
            voice_name=voice_name,
            filepath=filepath,
            audio_format=self.audio_format,
            speaking_rate=self.speaking_rate,
        )


class EdgeRenderer:
    def __init__(self, rate: str = "+0%", pitch: str = "+0Hz"):
        self.rate = rate
        self.pitch = pitch
        self.ffmpeg = shutil.which("ffmpeg")
        if self.ffmpeg is None:
            raise RuntimeError("ffmpeg is required for --tts-engine edge.")

    def render(self, *, ssml: str, text: str, language_code: str, voice_name: str, filepath: Path) -> None:
        try:
            import edge_tts
        except Exception as exc:
            raise RuntimeError("edge-tts is not installed. Install with: pip install edge-tts") from exc

        render_text = prepare_sentence_text(text)

        async def _render_mp3(mp3_path: Path) -> None:
            communicate = edge_tts.Communicate(
                text=render_text,
                voice=voice_name,
                rate=self.rate,
                pitch=self.pitch,
            )
            await communicate.save(str(mp3_path))

        filepath.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            mp3_path = Path(tmp.name)

        try:
            asyncio.run(_render_mp3(mp3_path))
            subprocess.run(
                [self.ffmpeg, "-y", "-i", str(mp3_path), "-ar", "24000", "-ac", "1", str(filepath)],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        finally:
            mp3_path.unlink(missing_ok=True)


class PiperRenderer:
    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        executable: str = "piper",
        speaker: Optional[int] = None,
        length_scale: Optional[float] = None,
        noise_scale: Optional[float] = None,
        noise_w: Optional[float] = None,
        sentence_silence: Optional[float] = None,
        use_python_module: bool = False,
    ):
        exe = shutil.which(executable) or shutil.which("piper") or shutil.which("piper-tts")
        if use_python_module:
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

    def render(self, *, ssml: str, text: str, language_code: str, voice_name: str, filepath: Path) -> None:
        render_text = prepare_sentence_text(text)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        cmd = [*self.command_prefix, "--model", self.model_path, "--output_file", str(filepath)]
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

        proc = subprocess.run(cmd, input=render_text, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            raise RuntimeError(f"Piper rendering failed for text={render_text!r}\nSTDERR:\n{proc.stderr}")


class KokoroRenderer:
    def __init__(self, lang_code: str = "b", speed: float = 1.0):
        try:
            from kokoro import KPipeline
        except Exception as exc:
            raise RuntimeError("Kokoro is not installed. Install kokoro or choose another --tts-engine.") from exc
        self.pipeline = KPipeline(lang_code=lang_code)
        self.speed = speed

    def render(self, *, ssml: str, text: str, language_code: str, voice_name: str, filepath: Path) -> None:
        try:
            import numpy as np
            import soundfile as sf
        except Exception as exc:
            raise RuntimeError("numpy and soundfile are required for --tts-engine kokoro.") from exc

        chunks = []
        for _, _, audio in self.pipeline(normalize_render_text(text), voice=voice_name, speed=self.speed):
            audio_np = np.asarray(audio, dtype=np.float32)
            chunks.append(audio_np.reshape(-1))
        if not chunks:
            raise RuntimeError(f"Kokoro generated no audio for text={text!r}, voice={voice_name!r}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(filepath), np.concatenate(chunks), 24000)


class CoquiRenderer:
    def __init__(self, model_name: str, speaker: Optional[str] = None, language: Optional[str] = None, speaker_wav: Optional[str] = None):
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise RuntimeError("Coqui TTS is not installed. Install with: pip install TTS") from exc
        self.tts = TTS(model_name=model_name)
        self.speaker = speaker
        self.language = language
        self.speaker_wav = speaker_wav

    def render(self, *, ssml: str, text: str, language_code: str, voice_name: str, filepath: Path) -> None:
        kwargs = {}
        if self.speaker is not None:
            kwargs["speaker"] = self.speaker
        if self.language is not None:
            kwargs["language"] = self.language
        if self.speaker_wav is not None:
            kwargs["speaker_wav"] = self.speaker_wav
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.tts.tts_to_file(text=normalize_render_text(text), file_path=str(filepath), **kwargs)


def make_renderer(args):
    if args.tts_engine == "google":
        return GoogleRenderer(audio_format=args.format, speaking_rate=args.speaking_rate)
    if args.tts_engine == "edge":
        return EdgeRenderer(rate=args.edge_rate, pitch=args.edge_pitch)
    if args.tts_engine == "piper":
        if not args.piper_model:
            raise RuntimeError("--piper-model is required when --tts-engine piper")
        return PiperRenderer(
            model_path=args.piper_model,
            config_path=args.piper_config,
            executable=args.piper_exe,
            speaker=args.piper_speaker,
            length_scale=args.piper_length_scale,
            noise_scale=args.piper_noise_scale,
            noise_w=args.piper_noise_w,
            sentence_silence=args.piper_sentence_silence,
            use_python_module=args.piper_use_python_module,
        )
    if args.tts_engine == "kokoro":
        return KokoroRenderer(lang_code=args.kokoro_lang_code, speed=args.kokoro_speed)
    if args.tts_engine == "coqui":
        return CoquiRenderer(
            model_name=args.coqui_model,
            speaker=args.coqui_speaker,
            language=args.coqui_language,
            speaker_wav=args.speaker_wav,
        )
    raise ValueError(f"Unsupported TTS engine: {args.tts_engine}")


def resolve_asset_voice(args, asset: dict) -> str:
    if args.tts_engine == "google":
        return asset["voice"]
    if args.tts_engine == "edge":
        if asset["gender"] == "female" and args.edge_female_voice:
            return args.edge_female_voice
        if asset["gender"] == "male" and args.edge_male_voice:
            return args.edge_male_voice
        return EDGE_VOICES[asset["accent"]][asset["gender"]]
    if args.tts_engine == "kokoro":
        return args.kokoro_female_voice if asset["gender"] == "female" else args.kokoro_male_voice
    if args.tts_engine == "piper":
        return f"piper_speaker_{args.piper_speaker}" if args.piper_speaker is not None else "piper"
    if args.tts_engine == "coqui":
        return args.coqui_speaker or "coqui"
    raise ValueError(f"Unsupported TTS engine: {args.tts_engine}")


def read_alignment_csv(path: Path) -> List[dict]:
    """Read phone intervals from CSV with phone,start,end columns in seconds."""
    with path.open(newline="", encoding="utf-8") as csvfile:
        rows = []
        for row in csv.DictReader(csvfile):
            phone = row.get("phone") or row.get("label") or row.get("text")
            start = row.get("start") or row.get("xmin")
            end = row.get("end") or row.get("xmax")
            if phone is None or start is None or end is None:
                raise ValueError(f"Alignment CSV must include phone/start/end columns: {path}")
            rows.append({"phone": phone.strip(), "start": float(start), "end": float(end)})
    return rows


def read_alignment_textgrid(path: Path, tier_name: str = "phones") -> List[dict]:
    """Read phone intervals from a simple Praat TextGrid phone tier."""
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    in_tier = False
    current_start = None
    current_end = None

    for raw_line in lines:
        line = raw_line.strip()
        if line.startswith("name ="):
            in_tier = line.split("=", 1)[1].strip().strip('"') == tier_name
            continue
        if not in_tier:
            continue
        if line.startswith("xmin ="):
            current_start = float(line.split("=", 1)[1].strip())
        elif line.startswith("xmax ="):
            current_end = float(line.split("=", 1)[1].strip())
        elif line.startswith("text ="):
            phone = line.split("=", 1)[1].strip().strip('"')
            if current_start is not None and current_end is not None and phone:
                rows.append({"phone": phone, "start": current_start, "end": current_end})
            current_start = None
            current_end = None
    return rows


def read_phone_alignment(path: Path, tier_name: str = "phones") -> List[dict]:
    """Read phone alignment intervals from .csv or .TextGrid."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_alignment_csv(path)
    if suffix in {".textgrid", ".textgrid.txt"} or path.name.lower().endswith(".textgrid"):
        return read_alignment_textgrid(path, tier_name=tier_name)
    raise ValueError(f"Unsupported alignment format for {path}. Use CSV or TextGrid.")


def find_aligned_phone_interval(
    intervals: Sequence[dict],
    target_phone: str,
    *,
    occurrence: int = 1,
    aliases: Optional[Dict[str, Sequence[str]]] = None,
) -> dict:
    """Return the nth matching phone interval from forced-alignment output."""
    if occurrence < 1:
        raise ValueError("occurrence must be 1-based.")
    candidates = {target_phone}
    if aliases and target_phone in aliases:
        candidates.update(aliases[target_phone])
    normalized = {safe_name(phone).upper() for phone in candidates}
    seen = 0
    for interval in intervals:
        label = str(interval["phone"]).strip()
        if label in candidates or safe_name(label).upper() in normalized:
            seen += 1
            if seen == occurrence:
                return interval
    raise ValueError(f"Could not find occurrence {occurrence} of phone {target_phone!r} in alignment.")


def crop_wav_segment(
    source_wav: Path,
    output_wav: Path,
    start_seconds: float,
    end_seconds: float,
    *,
    padding_seconds: float = DEFAULT_CROP_PADDING_SECONDS,
) -> Tuple[float, float]:
    """Crop a WAV segment by time interval and return the clamped start/end seconds."""
    if end_seconds <= start_seconds:
        raise ValueError(f"Invalid crop interval: start={start_seconds} end={end_seconds}")

    with wave.open(str(source_wav), "rb") as reader:
        params = reader.getparams()
        frame_rate = reader.getframerate()
        total_frames = reader.getnframes()
        start_frame = max(0, int(round((start_seconds - padding_seconds) * frame_rate)))
        end_frame = min(total_frames, int(round((end_seconds + padding_seconds) * frame_rate)))
        if end_frame <= start_frame:
            raise ValueError(f"Crop interval is empty after clamping: {source_wav}")
        reader.setpos(start_frame)
        audio = reader.readframes(end_frame - start_frame)

    output_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output_wav), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(audio)

    return start_frame / frame_rate, end_frame / frame_rate


def clip_output_path(source_wav: Path) -> Path:
    return source_wav.parent / "clip" / source_wav.name


def target_ipa_span(rendered_ipa: str, target_ipa: str) -> Optional[Tuple[int, int]]:
    if not rendered_ipa or not target_ipa:
        return None
    start = rendered_ipa.find(target_ipa)
    if start < 0:
        return None
    return start, start + len(target_ipa)


def is_auto_clippable_asset(asset: dict) -> bool:
    if target_ipa_span(asset["rendered_ipa"], asset["target_ipa"]) is None:
        return False
    if asset["context"] == "word" and asset["rendered_ipa"] == asset["target_ipa"]:
        return False
    return True


def proportional_phone_interval(asset: dict, source_wav: Path) -> Optional[Tuple[float, float]]:
    """Estimate target-phone interval from its known position in rendered IPA."""
    if not is_auto_clippable_asset(asset):
        return None

    span = target_ipa_span(asset["rendered_ipa"], asset["target_ipa"])
    if span is None:
        return None

    rendered_len = len(asset["rendered_ipa"])
    if rendered_len <= 0:
        return None

    with wave.open(str(source_wav), "rb") as reader:
        duration = reader.getnframes() / float(reader.getframerate())

    if duration <= 0:
        return None

    start_idx, end_idx = span
    start_seconds = duration * (start_idx / rendered_len)
    end_seconds = duration * (end_idx / rendered_len)
    min_duration = min(0.08, duration)
    if end_seconds - start_seconds < min_duration:
        center = (start_seconds + end_seconds) / 2.0
        start_seconds = max(0.0, center - min_duration / 2.0)
        end_seconds = min(duration, center + min_duration / 2.0)
    return start_seconds, end_seconds


def crop_asset_phone_clip(
    asset: dict,
    source_wav: Path,
    *,
    padding_seconds: float = DEFAULT_AUTO_CLIP_PADDING_SECONDS,
    overwrite: bool = False,
) -> Optional[dict]:
    interval = proportional_phone_interval(asset, source_wav)
    if interval is None:
        return None

    output_wav = clip_output_path(source_wav)
    if output_wav.exists() and not overwrite:
        return {
            "clip_path": str(output_wav),
            "clip_status": "exists",
            "clip_method": "proportional_ipa",
        }

    crop_start, crop_end = crop_wav_segment(
        source_wav,
        output_wav,
        interval[0],
        interval[1],
        padding_seconds=padding_seconds,
    )
    return {
        "clip_path": str(output_wav),
        "clip_status": "created",
        "clip_method": "proportional_ipa",
        "clip_start": crop_start,
        "clip_end": crop_end,
    }


def crop_phone_from_alignment(
    source_wav: Path,
    alignment_path: Path,
    target_phone: str,
    output_wav: Path,
    *,
    occurrence: int = 1,
    padding_seconds: float = DEFAULT_CROP_PADDING_SECONDS,
    tier_name: str = "phones",
    aliases: Optional[Dict[str, Sequence[str]]] = None,
) -> dict:
    """Crop one phone from a WAV using external phone alignment intervals."""
    intervals = read_phone_alignment(alignment_path, tier_name=tier_name)
    interval = find_aligned_phone_interval(intervals, target_phone, occurrence=occurrence, aliases=aliases)
    crop_start, crop_end = crop_wav_segment(
        source_wav,
        output_wav,
        float(interval["start"]),
        float(interval["end"]),
        padding_seconds=padding_seconds,
    )
    return {
        "source_wav": str(source_wav),
        "alignment_path": str(alignment_path),
        "output_wav": str(output_wav),
        "target_phone": target_phone,
        "matched_phone": interval["phone"],
        "alignment_start": float(interval["start"]),
        "alignment_end": float(interval["end"]),
        "crop_start": crop_start,
        "crop_end": crop_end,
    }


def make_base_context(target_ipa: str, name: str, word: str) -> List[dict]:
    return [
        {
            "context": "word",
            "position": "lexical",
            "target_ipa": target_ipa,
            "rendered_ipa": target_ipa,
            "text": word,
            "natural_word": word,
            "prompt_type": "word",
            "description": f"Canonical word example for {name}: {word}",
        },
        {
            "context": "carrier_phrase",
            "position": "lexical",
            "target_ipa": target_ipa,
            "rendered_ipa": target_ipa,
            "text": word,
            "natural_word": word,
            "prompt_type": "carrier_phrase",
            "description": f"Carrier phrase for {name}: Say {word} again",
        },
    ]


def generate_contexts(accent: str, target_ipa: str, name: str, word: str, category: str) -> List[dict]:
    contexts = make_base_context(target_ipa, name, word)

    if category == "consonants":
        table = CONSONANT_CONTEXTS.get(target_ipa)
        if not table:
            return contexts
        for context_name, value in table.items():
            if value is None:
                continue
            rendered_ipa, syllable_text, natural_word = value
            position = CONTEXT_POSITION.get(context_name, "other")
            contexts.append({
                "context": context_name,
                "position": position,
                "target_ipa": target_ipa,
                "rendered_ipa": rendered_ipa,
                "text": natural_word,
                "natural_word": natural_word,
                "prompt_type": "natural_word",
                "description": f"{context_name} natural-word example for {name}: {natural_word}",
            })
            contexts.append({
                "context": f"{context_name}_syllable",
                "position": position,
                "target_ipa": target_ipa,
                "rendered_ipa": rendered_ipa,
                "text": syllable_text,
                "natural_word": natural_word,
                "prompt_type": "syllable_like",
                "description": f"{context_name} syllable-like prompt for {name}: {syllable_text}",
            })
        return contexts

    if category == "vowels":
        table = VOWEL_CONTEXTS.get(accent, {}).get(target_ipa)
        if not table:
            return contexts
        for context_name, value in table.items():
            rendered_ipa, text = value
            contexts.append({
                "context": context_name,
                "position": CONTEXT_POSITION.get(context_name, "vowel_nucleus"),
                "target_ipa": target_ipa,
                "rendered_ipa": rendered_ipa,
                "text": text,
                "natural_word": text,
                "prompt_type": "vowel_frame" if context_name == "hvd" else "natural_word",
                "description": f"{context_name} vowel example for {name}: {text}",
            })
        return contexts

    return contexts


def iter_assets(selected_accents: Iterable[str]) -> Iterable[dict]:
    for accent in selected_accents:
        inventory = INVENTORIES[accent]
        voice_config = VOICES[accent]
        language_code = voice_config["language_code"]
        for category, phonemes in inventory.items():
            for target_ipa, name, word in phonemes:
                for ctx in generate_contexts(accent, target_ipa, name, word, category):
                    for gender, voice_name in voice_config.items():
                        if gender == "language_code":
                            continue
                        arpa = IPA_TO_ARPABET.get(target_ipa, "")
                        yield {
                            "accent": accent,
                            "language_code": language_code,
                            "gender": gender,
                            "voice": voice_name,
                            "category": category,
                            "phoneme_name": name,
                            "arpabet": arpa,
                            **ctx,
                        }


def dedupe_key(asset: dict) -> Tuple[str, str, str, str, str, str]:
    return (
        asset["accent"],
        asset["gender"],
        asset["category"],
        asset["phoneme_name"],
        asset["rendered_ipa"],
        asset["text"],
    )


def keep_asset_for_profile(asset: dict, profile: str, seen_keys: set[Tuple[str, str, str, str, str, str]]) -> Tuple[bool, str]:
    if profile == ASSET_PROFILE_EXHAUSTIVE:
        return True, ""
    if profile != ASSET_PROFILE_RICH_UNIQUE:
        raise ValueError(f"Unsupported asset profile: {profile}")

    if asset["context"] == "carrier_phrase":
        return False, "carrier_phrase"

    allowed_contexts = RICH_UNIQUE_CONTEXTS.get(asset["category"])
    if allowed_contexts is not None and asset["context"] not in allowed_contexts:
        return False, "profile_context"

    key = dedupe_key(asset)
    if key in seen_keys:
        return False, "duplicate_prompt"
    seen_keys.add(key)
    return True, ""


def validate_context_coverage() -> List[str]:
    warnings = []
    for accent, inventory in INVENTORIES.items():
        for category, phonemes in inventory.items():
            for target_ipa, name, word in phonemes:
                contexts = generate_contexts(accent, target_ipa, name, word, category)
                if len(contexts) <= 2:
                    warnings.append(f"Only base contexts for {accent}/{category}/{name}/{target_ipa}")
                if target_ipa not in IPA_TO_ARPABET:
                    warnings.append(f"No ARPAbet mapping for {accent}/{category}/{name}/{target_ipa}")
    return warnings


def main():
    parser = argparse.ArgumentParser(description="Generate accent-specific English phoneme audio assets with a CSV manifest.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--format", choices=["wav", "mp3"], default=DEFAULT_AUDIO_FORMAT)
    parser.add_argument("--speaking-rate", type=float, default=DEFAULT_SPEAKING_RATE)
    parser.add_argument("--accent", action="append", choices=sorted(INVENTORIES.keys()), help="Accent to generate. Repeatable. Defaults to all accents.")
    parser.add_argument("--copy-verified-inventory", action="store_true", help="Copy a verified <source-root>/<voice>/<PHONE>.wav ARPABET inventory into --output-dir.")
    parser.add_argument("--review-inventory", action="store_true", help="Review --output-dir/phoneme_inventory_manifest.json for missing or unsafe assets.")
    parser.add_argument("--source-root", type=Path, default=None, help="Source root for --copy-verified-inventory, shaped as <root>/<voice>/<PHONE>.wav.")
    parser.add_argument("--inventory-voices", default=",".join(DEFAULT_INVENTORY_VOICES), help="Comma-separated voices for --copy-verified-inventory.")
    parser.add_argument("--phones", default=None, help="Comma-separated ARPABET phones for --copy-verified-inventory. Default: all supported.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest only; do not call Google TTS.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files even if they already exist.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of assets for testing.")
    parser.add_argument("--asset-profile", choices=ASSET_PROFILES, default=ASSET_PROFILE_RICH_UNIQUE, help="rich_unique removes carrier phrases and duplicate prompts; exhaustive preserves all contexts.")
    parser.add_argument("--no-auto-clips", action="store_true", help="Disable automatic target-phone clips under each category's clip/ directory.")
    parser.add_argument("--clips-only", action="store_true", help="Create missing clip/ WAVs from existing rendered assets without rendering audio.")
    parser.add_argument("--clip-padding", type=float, default=DEFAULT_AUTO_CLIP_PADDING_SECONDS, help="Seconds of padding around automatic proportional phone clips.")
    parser.add_argument("--tts-engine", choices=TTS_ENGINES, default="google", help="Renderer backend. google uses SSML; other engines render plain prompt text.")
    parser.add_argument("--edge-female-voice", default=None, help="Override Edge female voice for all accents.")
    parser.add_argument("--edge-male-voice", default=None, help="Override Edge male voice for all accents.")
    parser.add_argument("--edge-rate", default="+0%")
    parser.add_argument("--edge-pitch", default="+0Hz")
    parser.add_argument("--piper-exe", default="piper")
    parser.add_argument("--piper-use-python-module", action="store_true")
    parser.add_argument("--piper-model", default=None, help="Path to Piper ONNX model. Required for --tts-engine piper.")
    parser.add_argument("--piper-config", default=None, help="Optional Piper model config JSON.")
    parser.add_argument("--piper-speaker", type=int, default=None)
    parser.add_argument("--piper-length-scale", type=float, default=1.2)
    parser.add_argument("--piper-noise-scale", type=float, default=None)
    parser.add_argument("--piper-noise-w", type=float, default=None)
    parser.add_argument("--piper-sentence-silence", type=float, default=None)
    parser.add_argument("--kokoro-female-voice", default="bf_emma")
    parser.add_argument("--kokoro-male-voice", default="bm_lewis")
    parser.add_argument("--kokoro-lang-code", default="b")
    parser.add_argument("--kokoro-speed", type=float, default=1.0)
    parser.add_argument("--coqui-model", default="tts_models/en/ljspeech/tacotron2-DDC")
    parser.add_argument("--coqui-speaker", default=None)
    parser.add_argument("--coqui-language", default=None)
    parser.add_argument("--speaker-wav", default=None)
    parser.add_argument("--crop-phone", action="store_true", help="Crop one aligned phone from an existing WAV instead of generating assets.")
    parser.add_argument("--source-wav", type=Path, help="Input WAV for --crop-phone.")
    parser.add_argument("--alignment", type=Path, help="Phone alignment CSV/TextGrid for --crop-phone.")
    parser.add_argument("--target-phone", help="Phone label to crop for --crop-phone.")
    parser.add_argument("--crop-output", type=Path, help="Output WAV path for --crop-phone.")
    parser.add_argument("--phone-occurrence", type=int, default=1, help="1-based occurrence of target phone to crop.")
    parser.add_argument("--crop-padding", type=float, default=DEFAULT_CROP_PADDING_SECONDS, help="Seconds of padding around cropped phone.")
    parser.add_argument("--alignment-tier", default="phones", help="TextGrid tier name for phone intervals.")
    args = parser.parse_args()

    if args.review_inventory:
        raise SystemExit(review_inventory_manifest(args.output_dir))

    if args.copy_verified_inventory:
        copy_verified_inventory(args)
        return

    if args.crop_phone:
        missing = [
            name
            for name, value in {
                "--source-wav": args.source_wav,
                "--alignment": args.alignment,
                "--target-phone": args.target_phone,
                "--crop-output": args.crop_output,
            }.items()
            if not value
        ]
        if missing:
            raise SystemExit(f"--crop-phone requires: {', '.join(missing)}")
        result = crop_phone_from_alignment(
            source_wav=args.source_wav,
            alignment_path=args.alignment,
            target_phone=args.target_phone,
            output_wav=args.crop_output,
            occurrence=args.phone_occurrence,
            padding_seconds=args.crop_padding,
            tier_name=args.alignment_tier,
        )
        print("Cropped phone:")
        for key, value in result.items():
            print(f"  {key}: {value}")
        return

    selected_accents = args.accent or sorted(INVENTORIES.keys())
    if args.tts_engine in {"piper", "kokoro", "coqui"} and args.format != "wav":
        raise SystemExit(f"--tts-engine {args.tts_engine} currently supports --format wav only.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.csv"

    warnings = validate_context_coverage()
    if warnings:
        print("Validation warnings:")
        for warning in warnings:
            print(f"  - {warning}")

    fieldnames = [
        "asset_id", "tts_engine", "accent", "language_code", "gender", "voice", "category",
        "phoneme_name", "arpabet", "target_ipa", "rendered_ipa", "context", "position",
        "prompt_type", "text", "render_text", "natural_word", "ssml", "filename",
        "clip_filename", "clip_method", "description",
    ]

    renderer = None if args.dry_run or args.clips_only else make_renderer(args)
    count = 0
    candidates = 0
    skipped_by_reason = {
        "carrier_phrase": 0,
        "profile_context": 0,
        "duplicate_prompt": 0,
    }
    clip_created = 0
    clip_exists = 0
    clip_skipped = 0
    seen_keys: set[Tuple[str, str, str, str, str, str]] = set()

    with manifest_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for asset in iter_assets(selected_accents):
            candidates += 1
            keep, skip_reason = keep_asset_for_profile(asset, args.asset_profile, seen_keys)
            if not keep:
                skipped_by_reason[skip_reason] = skipped_by_reason.get(skip_reason, 0) + 1
                continue

            if args.limit is not None and count >= args.limit:
                break

            asset_id = "_".join([
                asset["accent"], asset["gender"], asset["category"], safe_name(asset["phoneme_name"]),
                safe_name(asset["target_ipa"]), asset["context"], safe_name(asset["rendered_ipa"]),
            ])
            filename = f"{asset_id}.{args.format}"
            out_file = args.output_dir / asset["accent"] / asset["gender"] / asset["category"] / filename
            clip_file = clip_output_path(out_file)
            ssml = build_ssml(asset["rendered_ipa"], asset["text"], asset["context"])
            render_text = normalize_render_text(asset["text"])
            render_voice = resolve_asset_voice(args, asset)

            row = {
                "asset_id": asset_id,
                "tts_engine": args.tts_engine,
                "accent": asset["accent"],
                "language_code": asset["language_code"],
                "gender": asset["gender"],
                "voice": render_voice,
                "category": asset["category"],
                "phoneme_name": asset["phoneme_name"],
                "arpabet": asset["arpabet"],
                "target_ipa": asset["target_ipa"],
                "rendered_ipa": asset["rendered_ipa"],
                "context": asset["context"],
                "position": asset["position"],
                "prompt_type": asset["prompt_type"],
                "text": asset["text"],
                "render_text": render_text,
                "natural_word": asset["natural_word"],
                "ssml": ssml,
                "filename": str(out_file),
                "clip_filename": "",
                "clip_method": "",
                "description": asset["description"],
            }

            if args.dry_run:
                if not args.no_auto_clips and is_auto_clippable_asset(asset):
                    row["clip_filename"] = str(clip_file)
                    row["clip_method"] = "proportional_ipa"
                print(f"DRY RUN: {out_file}")
            elif args.clips_only:
                if out_file.exists():
                    print(f"Clip source: {out_file}")
                else:
                    print(f"Missing clip source: {out_file}")
            elif out_file.exists() and not args.overwrite:
                print(f"Skip existing: {out_file}")
            else:
                try:
                    renderer.render(
                        ssml=ssml,
                        text=render_text,
                        language_code=asset["language_code"],
                        voice_name=render_voice,
                        filepath=out_file,
                    )
                    print(f"Saved: {out_file}")
                except Exception as exc:
                    print(f"Error: {out_file}: {exc}")
            if not args.dry_run and not args.no_auto_clips and out_file.exists():
                try:
                    clip_result = crop_asset_phone_clip(
                        asset,
                        out_file,
                        padding_seconds=args.clip_padding,
                        overwrite=args.overwrite,
                    )
                    if clip_result is None:
                        clip_skipped += 1
                    else:
                        row["clip_filename"] = clip_result["clip_path"]
                        row["clip_method"] = clip_result["clip_method"]
                        if clip_result["clip_status"] == "created":
                            clip_created += 1
                            print(f"Clipped: {clip_result['clip_path']}")
                        else:
                            clip_exists += 1
                            print(f"Skip existing clip: {clip_result['clip_path']}")
                except Exception as exc:
                    clip_skipped += 1
                    print(f"Clip error: {out_file}: {exc}")
            elif args.no_auto_clips:
                clip_skipped += 1

            writer.writerow(row)
            count += 1

    print(f"\nDone. Manifest saved to: {manifest_path}")
    print(f"Asset profile: {args.asset_profile}")
    print(f"Candidate assets scanned: {candidates}")
    print(f"Assets described: {count}")
    print(f"Clips created: {clip_created}")
    print(f"Clips already existed: {clip_exists}")
    print(f"Clips skipped: {clip_skipped}")
    if args.asset_profile != ASSET_PROFILE_EXHAUSTIVE:
        print("Skipped assets:")
        for reason, skipped_count in skipped_by_reason.items():
            print(f"  {reason}: {skipped_count}")


if __name__ == "__main__":
    main()
