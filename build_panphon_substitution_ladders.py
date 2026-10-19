import json
from itertools import product
import asyncio
import re
import subprocess
import tempfile
from pathlib import Path

import panphon.distance
import edge_tts

EDGE_TTS_VOICES = {
    "man": "en-AU-WilliamNeural",
    "woman": "en-AU-NatashaNeural",
}

EDGE_TTS_OUTPUT_ROOTS = {
    "man": "man",
    "woman": "woman",
}

VOCAB_PHONE_DICT = {
    "bath": ["B", "AE", "TH"],
    "food": ["F", "UW", "D"],
    "go": ["G", "OW"],
    "stop": ["S", "T", "O", "P"],
    "no": ["N", "OW"],
    "pain": ["P", "EY", "N"],
    "yes": ["Y", "EH", "S"],
    "help": ["HH", "EH", "L", "P"],
}

# Use one IPA representative per ARPABET phone.
# These are simplified Australian/General-English-friendly symbols.
ARPABET_TO_IPA = {
    "P": "p",
    "B": "b",
    "T": "t",
    "D": "d",
    "K": "k",
    "G": "ɡ",

    "F": "f",
    "V": "v",
    "TH": "θ",
    "DH": "ð",
    "S": "s",
    "Z": "z",
    "SH": "ʃ",
    "ZH": "ʒ",
    "HH": "h",

    "CH": "t͡ʃ",
    "JH": "d͡ʒ",

    "M": "m",
    "N": "n",
    "NG": "ŋ",

    "L": "l",
    "R": "ɹ",
    "Y": "j",
    "W": "w",

    "IY": "i",
    "IH": "ɪ",
    "EH": "e",
    "AE": "æ",
    "AH": "ʌ",
    "AX": "ə",
    "AA": "ɑ",
    "AO": "ɔ",
    "UH": "ʊ",
    "UW": "u",
    "ER": "ɜ",
    "OW": "o",
    "EY": "e",
    "AY": "aɪ",
    "AW": "aʊ",
    "OY": "ɔɪ",
    "O": "ɒ",
}

VOWEL_PHONES = {
    "IY", "IH", "EH", "AE", "AH", "AX", "AA", "AO",
    "UH", "UW", "ER", "OW", "EY", "AY", "AW", "OY", "O",
}

PHONE_TO_GRAPHEME = {
    "P": "p", "B": "b", "T": "t", "D": "d", "K": "k", "G": "g",
    "S": "s", "Z": "z", "SH": "sh", "ZH": "zh", "CH": "ch", "JH": "j",
    "TH": "th", "DH": "dh", "F": "f", "V": "v", "HH": "h",
    "M": "m", "N": "n", "NG": "ng", "L": "l", "R": "r", "Y": "y", "W": "w",

    "IH": "i", "IY": "ee", "EH": "e", "AE": "a",
    "AH": "uh", "AX": "uh", "AA": "ah", "AO": "aw",
    "UH": "oo", "UW": "oo", "ER": "er",
    "OW": "oh", "EY": "ay", "AY": "eye", "AW": "ow",
    "OY": "oy", "O": "o",
}

PHONE_TO_GRAPHEME.update({
    "OW": "oh",
    "O": "o",
    "AO": "aw",
    "AA": "ah",
    "AH": "uh",
    "AX": "uhx",
    "ER": "er",
    "UH": "u",
    "UW": "oo",
    "IY": "ee",
    "IH": "ih",
    "EH": "eh",
    "AE": "a",
    "EY": "ay",
    "AY": "eye",
    "AW": "ow",
    "OY": "oy",
})

WORD_PHONE_GRAPHEME_OVERRIDES = {
    ("go", ("G", "OW")): "go",
    ("go", ("K", "OW")): "koh",
    ("go", ("D", "OW")): "doh",
    ("go", ("W", "OW")): "woh",

    ("bath", ("B", "AE", "TH")): "bath",
    ("bath", ("B", "AE", "DH")): "badh",
    ("bath", ("B", "AE", "F")): "baf",
    ("bath", ("B", "AE", "S")): "bas",
    ("bath", ("B", "AE", "HH")): "bah",

    ("yes", ("Y", "EH", "S")): "yes",
    ("yes", ("Y", "EH", "Z")): "yez",
    ("yes", ("Y", "EH", "TH")): "yeth",
    ("yes", ("Y", "EH", "SH")): "yesh",

    ("no", ("N", "OW")): "no",
    ("no", ("N", "UW")): "noo",
    ("no", ("N", "AO")): "naw",
    ("no", ("N", "AX")): "nuh",

    ("help", ("HH", "EH", "L", "P")): "help",
}

CONSONANT_SUBSTITUTION_POOL = [
    "P", "B", "T", "D", "K", "G",
    "F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH",
    "CH", "JH",
    "M", "N", "NG",
    "L", "R", "Y", "W",
]

VOWEL_SUBSTITUTION_POOL = [
    "IY", "IH", "EH", "AE",
    "AH", "AX", "AA", "AO",
    "UH", "UW", "ER",
    "OW", "EY", "AY", "AW", "OY", "O",
]

CONSONANT_MANNER = {
    "P": "stop", "B": "stop", "T": "stop", "D": "stop", "K": "stop", "G": "stop",
    "F": "fricative", "V": "fricative", "TH": "fricative", "DH": "fricative",
    "S": "fricative", "Z": "fricative", "SH": "fricative", "ZH": "fricative",
    "HH": "fricative",
    "CH": "affricate", "JH": "affricate",
    "M": "nasal", "N": "nasal", "NG": "nasal",
    "L": "liquid", "R": "liquid",
    "Y": "glide", "W": "glide",
}

CONSONANT_PLACE = {
    "P": "labial", "B": "labial", "M": "labial", "F": "labiodental", "V": "labiodental",
    "TH": "dental", "DH": "dental",
    "T": "alveolar", "D": "alveolar", "S": "alveolar", "Z": "alveolar", "N": "alveolar",
    "L": "alveolar", "R": "postalveolar",
    "SH": "postalveolar", "ZH": "postalveolar", "CH": "postalveolar", "JH": "postalveolar",
    "Y": "palatal",
    "K": "velar", "G": "velar", "NG": "velar", "W": "labial_velar",
    "HH": "glottal",
}

VOICELESS_STOPS = {"P", "T", "K"}
STOPS = {"P", "B", "T", "D", "K", "G"}
FRICATIVES = {"F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"}
SIBILANTS = {"S", "Z", "SH", "ZH"}
NASALS = {"M", "N", "NG"}
LIQUIDS = {"L", "R"}
GLIDES = {"Y", "W"}

# Approximate sonority hierarchy.
SONORITY = {
    "P": 1, "B": 1, "T": 1, "D": 1, "K": 1, "G": 1,
    "CH": 1.5, "JH": 1.5,
    "F": 2, "V": 2, "TH": 2, "DH": 2, "S": 2, "Z": 2, "SH": 2, "ZH": 2, "HH": 2,
    "M": 3, "N": 3, "NG": 3,
    "L": 4, "R": 4,
    "Y": 5, "W": 5,
}

VOICELESS_OBSTRUENTS = {"P", "T", "K", "F", "TH", "S", "SH", "CH", "HH"}
VOICED_OBSTRUENTS = {"B", "D", "G", "V", "DH", "Z", "ZH", "JH"}
CORONAL_FRICATIVES = {"TH", "DH", "S", "Z", "SH", "ZH"}
VOICELESS_CORONAL_FRICATIVES = {"TH", "S", "SH"}

MAX_LADDER_STATES_PER_PHONE = 3
MAX_RELAXATION_LEVELS = 3
CANDIDATE_BUFFER_MULTIPLIER = 3

def safe_filename(text: str) -> str:
    """
    Convert a generated grapheme prompt into a filesystem-safe filename stem.
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^a-z0-9_\-]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "empty"


def voice_suffix(voice_name: str) -> str:
    """
    Keep filenames readable while still preserving which Edge voice made them.
    Example:
        en-AU-WilliamNeural -> en-AU-WilliamNeural
    """
    return safe_filename(voice_name)


async def synthesize_edge_tts_to_wav(
    text: str,
    voice: str,
    wav_path: Path,
    rate: str = "+0%",
    pitch: str = "+0Hz",
    overwrite: bool = False,
):
    """
    Synthesize text with edge-tts and save as WAV.

    edge-tts natively writes compressed audio, usually MP3.
    This function writes a temporary MP3 and converts it to WAV with ffmpeg.

    Requires:
        pip install edge-tts
        ffmpeg available on PATH
    """
    wav_path.parent.mkdir(parents=True, exist_ok=True)

    if wav_path.exists() and not overwrite:
        return False

    with tempfile.TemporaryDirectory() as tmpdir:
        mp3_path = Path(tmpdir) / "edge_tts_tmp.mp3"

        communicate = edge_tts.Communicate(
            text=text,
            voice=voice,
            rate=rate,
            pitch=pitch,
        )

        await communicate.save(str(mp3_path))

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(mp3_path),
            "-ar",
            "24000",
            "-ac",
            "1",
            str(wav_path),
        ]

        subprocess.run(cmd, check=True)

    return True


async def synthesize_substitution_json_with_edge_tts(
    word: str,
    manifest: list[dict],
    output_root: str | Path = ".",
    voices: dict[str, str] = EDGE_TTS_VOICES,
    overwrite: bool = False,
    rate: str = "+0%",
    pitch: str = "+0Hz",
):
    """
    Synthesize all unique grapheme prompts from one word substitution manifest.

    Output:
        man/{word}/{new_word}_{voice_selected}.wav
        woman/{word}/{new_word}_{voice_selected}.wav
    """
    output_root = Path(output_root)

    unique_prompts = sorted({item["new_word"] for item in manifest})

    made = 0
    skipped = 0

    for gender_dir, voice in voices.items():
        for prompt in unique_prompts:
            wav_path = output_root / audio_path_for_prompt(
                word=word,
                prompt=prompt,
                gender_dir=gender_dir,
                voice=voice,
            )

            did_make = await synthesize_edge_tts_to_wav(
                text=prompt,
                voice=voice,
                wav_path=wav_path,
                rate=rate,
                pitch=pitch,
                overwrite=overwrite,
            )

            if did_make:
                made += 1
                print(f"Saved {wav_path}")
            else:
                skipped += 1

    return {"made": made, "skipped": skipped}


async def synthesize_all_substitution_jsons_with_edge_tts(
    transcript_root: str | Path = "tests",
    output_root: str | Path = ".",
    voices: dict[str, str] = EDGE_TTS_VOICES,
    overwrite: bool = False,
    rate: str = "+0%",
    pitch: str = "+0Hz",
):
    """
    Load every {word}_substitution_transcripts.json from transcript_root and
    synthesize audio files for both Australian man/woman Edge TTS voices.
    """
    transcript_root = Path(transcript_root)
    json_paths = sorted(transcript_root.glob("*_substitution_transcripts.json"))

    if not json_paths:
        raise FileNotFoundError(
            f"No *_substitution_transcripts.json files found in {transcript_root}"
        )

    total_made = 0
    total_skipped = 0

    for json_path in json_paths:
        word = json_path.name.replace("_substitution_transcripts.json", "")

        with open(json_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        print(
            "Synthesizing ",
            f"{word}: {len({item['new_word'] for item in manifest})}",
            " unique prompts"
        )

        result = await synthesize_substitution_json_with_edge_tts(
            word=word,
            manifest=manifest,
            output_root=output_root,
            voices=voices,
            overwrite=overwrite,
            rate=rate,
            pitch=pitch,
        )

        total_made += result["made"]
        total_skipped += result["skipped"]

    print(f"Edge TTS complete. Made={total_made}, skipped={total_skipped}")
    return {"made": total_made, "skipped": total_skipped}

def audio_path_for_prompt(
    word: str,
    prompt: str,
    gender_dir: str,
    voice: str,
) -> str:
    """
    Return the local relative path where the synthesized WAV should be saved.

    Example:
        man/stop/thtop_en-AU-WilliamNeural.wav
    """
    prompt_stem = safe_filename(prompt)
    voice_tag = voice_suffix(voice)
    return f"{gender_dir}/{word}/{prompt_stem}_{voice_tag}.wav"


def substitution_record(
    word: str,
    distance_key: str,
    new_word: str,
    voices: dict[str, str] = EDGE_TTS_VOICES,
) -> dict:
    """
    Build one JSON manifest record with audio paths.
    """
    return {
        "distance": distance_key,
        "word": word,
        "new_word": new_word,
        "path": {
            gender_dir: audio_path_for_prompt(
                word=word,
                prompt=new_word,
                gender_dir=gender_dir,
                voice=voice,
            )
            for gender_dir, voice in voices.items()
        },
    }

def is_vowel(phone: str) -> bool:
    return phone in VOWEL_PHONES


def same_cv_class(a: str, b: str) -> bool:
    return is_vowel(a) == is_vowel(b)


def distance_label(x: float) -> str:
    x = round(float(x), 2)
    if abs(x) < 1e-9:
        return "0.0"
    if abs(x - 1.0) < 1e-9:
        return "1.0"
    return f"{x:.2f}".rstrip("0").rstrip(".")

def relaxed_candidate_pool_for_phone(
    source_phone: str,
    role: str,
    relaxation_level: int,
) -> list[str]:
    """
    Linguistically staged candidate expansion.

    Level 0 = current strict candidate pool.
    Level 1 = nearby natural class expansion.
    Level 2 = broader but still pronounceable sonority-compatible expansion.

    This avoids missing ladder coordinates while not allowing arbitrary C -> C.
    """
    if is_vowel(source_phone):
        return VOWEL_SUBSTITUTION_POOL

    if relaxation_level == 0:
        return candidate_pool_for_phone(source_phone)

    # Glides in onset position: allow other high-sonority onset-like consonants.
    if source_phone in GLIDES:
        if relaxation_level == 1:
            return list(GLIDES | LIQUIDS)
        return list(GLIDES | LIQUIDS | {"HH"})

    # Nasals in onset position: allow other nasals first, then nearby sonorants.
    if source_phone in NASALS:
        if role.startswith("onset"):
            if relaxation_level == 1:
                return ["M", "N"]          # still block onset NG
            return ["M", "N", "L", "R"]    # sonorant fallback
        else:
            if relaxation_level == 1:
                return ["M", "N", "NG"]
            return ["M", "N", "NG", "L", "R"]

    # Liquids: allow liquid/glide interchange.
    if source_phone in LIQUIDS:
        return list(LIQUIDS | GLIDES)

    # Obstruents: keep the existing full consonant pool, filtered downstream.
    return candidate_pool_for_phone(source_phone)

def candidate_pool_for_phone(source_phone: str) -> list[str]:
    """
    Fill ladders from the full C/V-compatible phone pool.

    Vowels may substitute with any other vowel that PanPhon can score.
    Consonants may substitute with any other consonant that PanPhon can score.
    """
    if is_vowel(source_phone):
        return VOWEL_SUBSTITUTION_POOL

    return CONSONANT_SUBSTITUTION_POOL

def phones_to_grapheme(word: str, phones: list[str]) -> str:
    key = (word, tuple(phones))
    if key in WORD_PHONE_GRAPHEME_OVERRIDES:
        return WORD_PHONE_GRAPHEME_OVERRIDES[key]
    return "".join(PHONE_TO_GRAPHEME.get(p, p.lower()) for p in phones)

def phone_position_role(phonemes: list[str], position: int) -> str:
    """
    Approximate role for monosyllabic C/V words.

    This is not a full syllabifier, but works for your current word list:
    bath CVC, food CVC, go CV, stop CCVC, no CV, pain CVC, yes CVC, help CVCC.
    """
    vowel_positions = [i for i, p in enumerate(phonemes) if is_vowel(p)]
    if not vowel_positions:
        return "unknown"

    nucleus = vowel_positions[0]

    if position == nucleus:
        return "nucleus"

    if position < nucleus:
        if nucleus >= 2:
            if position == 0:
                return "onset_cluster_first"
            return "onset_cluster_later"
        return "onset_single"

    if position > nucleus:
        if len(phonemes) - nucleus > 2:
            return "coda_cluster"
        return "coda_single"

    return "unknown"

def make_panphon_distance():
    """
    PanPhon supports several distance functions.

    weighted_feature_edit_distance(source, target)
    computes a feature-aware edit distance using PanPhon's built-in feature
    geometry/weights. For single-segment strings this behaves like a weighted
    feature substitution cost.
    """
    return panphon.distance.Distance()

def is_english_plausible_wordform(phones: list[str]) -> bool:
    """
    Conservative phonotactic filter for this augmentation set.

    Blocks impossible-looking English forms such as DH T O P from stop,
    while allowing common CVC/CV/CCVC/CVCC shapes.
    """
    vowel_positions = [i for i, p in enumerate(phones) if is_vowel(p)]
    if len(vowel_positions) != 1:
        return False

    nucleus = vowel_positions[0]
    onset = phones[:nucleus]
    coda = phones[nucleus + 1:]

    # No word-initial NG in ordinary English.
    if onset and onset[0] == "NG":
        return False

    # Single onsets: most consonants okay, but avoid DH/NG as generated pseudo-word starts.
    if len(onset) == 1:
        if onset[0] in {"NG", "DH"}:
            return False

        # Two-consonant onsets.
    if len(onset) == 2:
        c1, c2 = onset

        # English has a special /s/+voiceless-stop onset pattern:
        #   stop, spill, skill.
        # For substitution-feedback generation, keep this more general as a
        # voiceless/front-fricative-compatible + voiceless-stop cluster, with Z
        # allowed as a lower-ranked voiced-fricative candidate. This allows
        # speech-like distortions such as thtop/shtop/ztop, but still blocks
        # dhtop/ngtop/ltop.
        if c1 in {"S", "TH", "SH", "F", "Z"} and c2 in VOICELESS_STOPS:
            pass

        # Other common rising-sonority clusters: stop/fricative + liquid/glide.
        elif CONSONANT_MANNER.get(c1) in {"stop", "fricative"} and c2 in (LIQUIDS | GLIDES):
            if SONORITY.get(c1, 0) >= SONORITY.get(c2, 0):
                return False

        else:
            return False

    # Avoid longer onsets for this simple generator.
    if len(onset) > 2:
        return False

    # Conservative coda checks.
    if len(coda) >= 1:
        # H, Y, W are generally not useful final consonant substitutions here.
        if any(p in {"HH", "Y", "W"} for p in coda):
            return False

    if len(coda) == 2:
        c1, c2 = coda

        # Help-like final clusters: liquid/nasal/fricative + stop are common enough.
        if c1 in LIQUIDS | NASALS | FRICATIVES and c2 in STOPS:
            pass
        # Fricative + stop codas such as /st/, /ft/ are also plausible.
        elif c1 in FRICATIVES and c2 in STOPS:
            pass
        else:
            return False

    if len(coda) > 2:
        return False

    return True

def is_process_plausible_substitution(
    source_phone: str,
    target_phone: str,
    role: str,
    phonemes: list[str],
    position: int,
    relaxation_level: int = 0,
) -> bool:
    """
    Allows substitutions that match common articulatory/phonological patterns,
    then leaves ranking to PanPhon distance.

    This is a conservative gate, not a probability model.
    """
    if source_phone == target_phone:
        return True

    if is_vowel(source_phone) and is_vowel(target_phone):
        return True

    if is_vowel(source_phone) != is_vowel(target_phone):
        return False

    src_manner = CONSONANT_MANNER.get(source_phone)
    tgt_manner = CONSONANT_MANNER.get(target_phone)
    src_place = CONSONANT_PLACE.get(source_phone)
    tgt_place = CONSONANT_PLACE.get(target_phone)

    next_phone = phonemes[position + 1] if position + 1 < len(phonemes) else None
    prev_phone = phonemes[position - 1] if position - 1 >= 0 else None

    # Word-initial cluster first member, e.g. S in stop.
    #
    # This is not simply "any consonant". It is a pre-stop fricative/obstruent
    # position. For speech-like distortions, preserve:
    #   1. voicelessness when the following consonant is voiceless
    #   2. coronal/front fricative character
    #   3. no sonorants/nasals/glides in this slot
    #
    # This allows TH/SH/F/Z before T. Z is allowed as a candidate but is
    # penalized by context_transition_penalty because it is voiced before
    # a voiceless stop.
    if role == "onset_cluster_first":
        if source_phone == "S" and next_phone in VOICELESS_STOPS:
            return target_phone in {"S", "TH", "SH", "F", "Z"}
        return False

    # The second member of an /s/+stop onset should remain stop-like.
    if role == "onset_cluster_later":
        if source_phone in STOPS:
            return target_phone in STOPS
        return False

    # Voicing/devoicing at same place and manner: s/z, t/d, p/b.
    if src_place == tgt_place and src_manner == tgt_manner:
        return True

    # Place substitutions within same manner: fronting/backing-like.
    if src_manner == tgt_manner:
        return True

    # Stopping/deaffrication/affrication.
    if src_manner == "fricative" and tgt_manner in {"stop", "affricate"}:
        return True
    if src_manner == "affricate" and tgt_manner in {"fricative", "stop"}:
        return True
    if src_manner == "stop" and tgt_manner == "affricate":
        return True

    # Relaxed sonorant fallback:
    # Use only when the strict speech-pattern filter cannot fill the ladder.
    # This keeps replacements within onset-pronounceable sonorant classes rather
    # than allowing arbitrary consonants.
    if relaxation_level > 0:
        if source_phone in GLIDES and target_phone in (GLIDES | LIQUIDS | {"HH"}):
            return True

        if source_phone in NASALS:
            if role.startswith("onset"):
                return target_phone in {"M", "N", "L", "R"}
            return target_phone in {"M", "N", "NG", "L", "R"}

        if source_phone in LIQUIDS and target_phone in (LIQUIDS | GLIDES):
            return True

    # Nasal place changes, but do not create initial NG.
    if source_phone in NASALS and target_phone in NASALS:
        if role.startswith("onset") and target_phone == "NG":
            return False
        return True

    return False

def panphon_phone_distance(distance_obj, source_phone: str, target_phone: str) -> float:
    if source_phone == target_phone:
        return 0.0

    if source_phone not in ARPABET_TO_IPA or target_phone not in ARPABET_TO_IPA:
        return float("inf")

    if not same_cv_class(source_phone, target_phone):
        return float("inf")

    source_ipa = ARPABET_TO_IPA[source_phone]
    target_ipa = ARPABET_TO_IPA[target_phone]

    try:
        return float(distance_obj.weighted_feature_edit_distance(source_ipa, target_ipa))
    except Exception:
        return float("inf")


def context_transition_penalty(
    phonemes: list[str],
    position: int,
    target_phone: str,
) -> float:
    penalty = 0.0

    prev_phone = phonemes[position - 1] if position > 0 else None
    next_phone = phonemes[position + 1] if position + 1 < len(phonemes) else None

    # If target is directly before a voiceless stop, avoid voicing mismatch.
    if next_phone in VOICELESS_STOPS:
        if target_phone in VOICED_OBSTRUENTS:
            penalty += 0.50

        if target_phone in NASALS | LIQUIDS | GLIDES:
            penalty += 0.75

    # If target is directly after /s/ in an /s/+stop cluster, keep it stop-like.
    if prev_phone == "S" and target_phone not in STOPS:
        penalty += 0.75

    # Block initial NG strongly.
    if position == 0 and target_phone == "NG":
        penalty += 10.0

    return penalty

_WORD_POSITION_LADDER_CACHE = {}


def ranked_panphon_ladder_for_word_position(
    word: str,
    phonemes: list[str],
    position: int,
    distance_obj,
) -> list[tuple[str, float]]:
    cache_key = (word, tuple(phonemes), position)
    if cache_key in _WORD_POSITION_LADDER_CACHE:
        return _WORD_POSITION_LADDER_CACHE[cache_key]

    source_phone = phonemes[position]
    role = phone_position_role(phonemes, position)

    candidates = []
    seen_candidate_phones = set()

    # Stop early once we have enough possible phones.
    candidate_buffer_target = (MAX_LADDER_STATES_PER_PHONE - 1) *\
            CANDIDATE_BUFFER_MULTIPLIER

    for relaxation_level in range(MAX_RELAXATION_LEVELS):
        for target_phone in relaxed_candidate_pool_for_phone(source_phone, role, relaxation_level):
            if target_phone == source_phone:
                continue

            if target_phone in seen_candidate_phones:
                continue

            if not same_cv_class(source_phone, target_phone):
                continue

            if not is_process_plausible_substitution(
                source_phone,
                target_phone,
                role,
                phonemes,
                position,
                relaxation_level=relaxation_level,
            ):
                continue

            test_phones = list(phonemes)
            test_phones[position] = target_phone

            if not is_english_plausible_wordform(test_phones):
                continue

            raw_distance = panphon_phone_distance(distance_obj, source_phone, target_phone)
            if raw_distance == float("inf"):
                continue

            penalty = context_transition_penalty(phonemes, position, target_phone)
            adjusted_distance = raw_distance + penalty + (0.25 * relaxation_level)

            candidates.append((target_phone, adjusted_distance, raw_distance, penalty, relaxation_level))
            seen_candidate_phones.add(target_phone)

        if len(candidates) >= candidate_buffer_target:
            break

    candidates.sort(key=lambda item: (item[4], item[1], item[0]))

    selected_phones = [source_phone]
    seen_single_position_graphemes = {phones_to_grapheme(word, phonemes)}

    for target_phone, adjusted_distance, raw_distance, penalty, relaxation_level in candidates:
        test_phones = list(phonemes)
        test_phones[position] = target_phone
        test_grapheme = phones_to_grapheme(word, test_phones)

        if test_grapheme in seen_single_position_graphemes:
            continue

        selected_phones.append(target_phone)
        seen_single_position_graphemes.add(test_grapheme)

        if len(selected_phones) >= MAX_LADDER_STATES_PER_PHONE:
            break

    if len(selected_phones) <= 1:
        ladder = [(source_phone, 0.0)]
    else:
        denom = len(selected_phones) - 1
        ladder = [(phone, i / denom) for i, phone in enumerate(selected_phones)]

    _WORD_POSITION_LADDER_CACHE[cache_key] = ladder
    return ladder


def ranked_panphon_ladder_for_word_position_debug(
    word: str,
    phonemes: list[str],
    position: int,
    distance_obj,
) -> list[dict]:
    source_phone = phonemes[position]
    role = phone_position_role(phonemes, position)

    candidates = []
    seen_candidate_phones = set()

    candidate_buffer_target = (MAX_LADDER_STATES_PER_PHONE - 1) * \
            CANDIDATE_BUFFER_MULTIPLIER

    for relaxation_level in range(MAX_RELAXATION_LEVELS):
        for target_phone in relaxed_candidate_pool_for_phone(
            source_phone,
            role,
            relaxation_level,
        ):
            if target_phone == source_phone:
                continue

            if target_phone in seen_candidate_phones:
                continue

            if not same_cv_class(source_phone, target_phone):
                continue

            if not is_process_plausible_substitution(
                source_phone,
                target_phone,
                role,
                phonemes,
                position,
                relaxation_level=relaxation_level,
            ):
                continue

            test_phones = list(phonemes)
            test_phones[position] = target_phone

            if not is_english_plausible_wordform(test_phones):
                continue

            raw_distance = panphon_phone_distance(distance_obj, source_phone, target_phone)
            if raw_distance == float("inf"):
                continue

            penalty = context_transition_penalty(phonemes, position, target_phone)
            adjusted_distance = raw_distance + penalty + (0.25 * relaxation_level)
            grapheme = phones_to_grapheme(word, test_phones)

            candidates.append(
                (
                    target_phone,
                    adjusted_distance,
                    raw_distance,
                    penalty,
                    relaxation_level,
                    grapheme,
                )
            )
            seen_candidate_phones.add(target_phone)

        if len(candidates) >= candidate_buffer_target:
            break

    candidates.sort(key=lambda item: (item[4], item[1], item[0]))

    selected = [
        (source_phone, 0.0, 0.0, 0.0, 0, phones_to_grapheme(word, phonemes))
    ]

    seen_single_position_graphemes = {phones_to_grapheme(word, phonemes)}

    for (
        target_phone,
        adjusted_distance,
        raw_distance,
        penalty,
        relaxation_level,
        grapheme,
    ) in candidates:
        if grapheme in seen_single_position_graphemes:
            continue

        selected.append(
            (
                target_phone,
                adjusted_distance,
                raw_distance,
                penalty,
                relaxation_level,
                grapheme,
            )
        )
        seen_single_position_graphemes.add(grapheme)

        if len(selected) >= MAX_LADDER_STATES_PER_PHONE:
            break

    if len(selected) == 1:
        return [{
            "phone": source_phone,
            "role": role,
            "rank_distance": 0.0,
            "adjusted_distance": 0.0,
            "raw_panphon_distance": 0.0,
            "context_penalty": 0.0,
            "grapheme": phones_to_grapheme(word, phonemes),
        }]

    denom = len(selected) - 1

    return [
        {
            "phone": phone,
            "role": role,
            "rank_distance": i / denom,
            "adjusted_distance": adjusted_distance,
            "raw_panphon_distance": raw_distance,
            "context_penalty": penalty,
            "relaxation_level": relaxation_level,
            "grapheme": grapheme,
        }
        for i, (
            phone,
            adjusted_distance,
            raw_distance,
            penalty,
            relaxation_level,
            grapheme,
        )
        in enumerate(selected)
    ]

def build_word_substitution_ladder(word: str, phonemes: list[str], distance_obj) -> list[dict]:
    ladders = [
        ranked_panphon_ladder_for_word_position(
            word=word,
            phonemes=phonemes,
            position=i,
            distance_obj=distance_obj,
        )
        for i in range(len(phonemes))
    ]

    records = []

    for combo in product(*ladders):
        candidate_phones = [phone for phone, _ in combo]
        candidate_distances = [d for _, d in combo]

        if len(candidate_phones) != len(phonemes):
            continue

        if any(not same_cv_class(src, dst) for src, dst in zip(phonemes, candidate_phones)):
            continue

        if not is_english_plausible_wordform(candidate_phones):
            continue

        distance_key = "_".join(distance_label(x) for x in candidate_distances)
        new_word = phones_to_grapheme(word, candidate_phones)

        records.append(
            substitution_record(
                word=word,
                distance_key=distance_key,
                new_word=new_word,
            )
        )

    return records


def write_per_word_substitution_transcript_jsons(
    output_root="tests",
    vocab_phone_dict=VOCAB_PHONE_DICT,
):
    import os

    os.makedirs(output_root, exist_ok=True)
    distance_obj = make_panphon_distance()

    all_ladders = {}

    # Useful audit file: shows what ladder each source phone received.
    phone_ladder_debug = {}

    for word, phonemes in vocab_phone_dict.items():
        manifest = build_word_substitution_ladder(word, phonemes, distance_obj)
        all_ladders[word] = manifest

        for i, phone in enumerate(phonemes):
            debug_key = f"{word}:{i}:{phone}:{phone_position_role(phonemes, i)}"
            phone_ladder_debug[debug_key] = ranked_panphon_ladder_for_word_position_debug(
                word,
                phonemes,
                i,
                distance_obj,
            )

        output_path = os.path.join(output_root, f"{word}_substitution_transcripts.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        print(f"Saved {output_path} with {len(manifest)} candidates")

    debug_path = os.path.join(output_root, "panphon_phone_ladders_debug.json")
    with open(debug_path, "w", encoding="utf-8") as f:
        json.dump(phone_ladder_debug, f, ensure_ascii=False, indent=2)

    print(f"Saved {debug_path}")
    return all_ladders


def main():
    write_per_word_substitution_transcript_jsons(output_root="tests")

    asyncio.run(
        synthesize_all_substitution_jsons_with_edge_tts(
            transcript_root="tests",
            output_root=".",
            overwrite=False,
        )
    )


if __name__ == "__main__":
    main()

