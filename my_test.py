"""
Guide:
    * selection first filters candidates by voice mode (female vs male), word/label,
    and candidate (global vs per-phoneme).
    * For triphone global search, collapse the EEG temporal mismatch wave to one
    whole-word target level, then search the word's triphone library.
    * `TRIPHONE_LEVEL_WEIGHT` is a selector-time ranking weight, not a generator
    level control. It scales how strongly `target_level` proximity affects ranking
    in `select_blabber_asset_triphone.select_assets`.

Distance-to-selection mapping in the current flow is:
  1. EEG mismatch signal to one target level
     target_level = clamp_unit(mean(per_time_l2))
  2. Selector level match per asset
     In select_blabber_asset_triphone.py, for each candidate asset:
     level_score = max(0, 1 - abs(asset_level - target_level))
  3. Add weighted level contribution
     score += level_weight * level_score
     In your test, level_weight = 6.0, so level proximity is strongly prioritized.
  4. Plus phonetic/similarity terms
     In global_search=True, total score also includes phone/triphone overlap + suitability/neural/phone similarity, but with lighter weights
     than normal mode.

  So ground truth will be selected when its total score is best. At high target levels (near 1), augmented assets often win on level
  proximity; at low target levels (near 0), ground truth is more likely to win. If you want ground truth to win more often regardless of
  distance, lower TRIPHONE_LEVEL_WEIGHT or add --max-level/--min-level constraints.


"""
from pathlib import Path
import os
import pprint
import sys

from Template_l2_compare_v2 import (
    compare_signal_to_prebuilt_template,
    build_template_from_dataset,
    get_one_trial_from_dataset,
)
from select_blabber_asset import (
    adapt_comparison_result,
    normalize_voice_mode,
)

from select_blabber_asset_triphone import load_assets, select_assets


REQUESTED_VOICE = "female"
TRIPHONE_ASSET_ROOT = Path(
    os.environ.get(
        "TRIPHONE_ASSET_ROOT",
        r"/data/raqchia/audio-speech/speech-assets-triphone",
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
    return max(0.0, min(1.0, float(value)))


def global_target_level(comparison: dict) -> float:
    """
    Map per-time L2 mismatch values to a single [0,1] target level.

    This target level is passed into selector scoring (not generation). During
    generation, candidate suitability already includes its own internal level
    match term based on phone-distance proximity to that candidate's level.
    """
    values = [float(value) for value in comparison["per_time_l2"]]
    if not values:
        return 0.0
    return clamp_unit(sum(values) / float(len(values)))


def library_json_path(word: str, voice_mode: str) -> Path:
    return TRIPHONE_ASSET_ROOT / voice_mode / GENERATION_MODE / word / f"{word}_blabber_scored.json"


def select_triphone_global_asset(comparison: dict, requested_voice: str) -> dict:
    word = str(comparison.get("label_name") or "").strip().lower()
    if word not in LEXICON:
        raise ValueError(f"No canonical phone sequence configured for word '{word}'.")

    voice_mode = normalize_voice_mode(requested_voice)
    library_path = library_json_path(word, voice_mode)
    if not library_path.is_file():
        raise FileNotFoundError(f"Missing triphone library JSON: {library_path}")

    target_level = global_target_level(comparison)
    assets, _payload = load_assets(library_path)
    selected = select_assets(
        assets,
        word=word,
        phones=LEXICON[word],
        triphones=[],
        voice=voice_mode,
        target_level=target_level,
        min_level=None,
        max_level=None,
        top_k=1,
        level_weight=TRIPHONE_LEVEL_WEIGHT,
        global_search=True,
    )
    if not selected:
        raise ValueError(f"No triphone assets matched word='{word}' voice='{voice_mode}'.")

    best = selected[0]
    asset = best["asset"]
    return {
        "word": word,
        "voice_mode": voice_mode,
        "generation_mode": GENERATION_MODE,
        "library_json": str(library_path),
        "audio_path": asset.get("audio_path") or asset.get("wav_path") or asset.get("output_audio_path"),
        "asset_id": asset.get("asset_id"),
        "target_level": target_level,
        "requested_generation_signature": [target_level],
        "selected_phoneme_grades": asset.get("phoneme_values", [asset.get("level")]),
        "score": best["score"],
        "score_details": best["score_details"],
        "asset_metadata": asset,
        "times": list(comparison["times"]),
        "per_time_l2": list(comparison["per_time_l2"]),
    }


if __name__ == "__main__":
    """
    Take from `Function call copy_v2.ipynb`
    """
    template_days = [1]
    template_sessions = range(1, 8)
    label = 0

    template_object = build_template_from_dataset(
        template_days=template_days,
        template_sessions=template_sessions,
        nperseg=128,
        noverlap=96,
        eps=1e-8,
        fmax=50,
    )

    trial, label_name, dataset = get_one_trial_from_dataset(
        day=1,
        sess=8,
        label=label,
        trial_index_within_label=0,
    )
    print("Loaded trial label:", label_name)

    result = compare_signal_to_prebuilt_template(
        template_object=template_object,
        input_signal=trial,
        input_label=label,
        output_mode="both",
        sr=250,
    )
    pprint.pprint(result)

    comparison = adapt_comparison_result(result)
    pprint.pprint(comparison)
    selection = select_triphone_global_asset(
        comparison=comparison,
        requested_voice=REQUESTED_VOICE,
    )

    print("Selected asset:", selection["audio_path"])
    print("Selected asset id:", selection["asset_id"])
    print("Generation mode:", selection["generation_mode"])
    print("Target level:", selection["target_level"])
    print("Score details:", selection["score_details"])
