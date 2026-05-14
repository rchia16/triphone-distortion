#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_ASSET_INDEX_PATH = ROOT / "blabber_asset_index.json"
ASSET_MODE_AUTO = "auto"
ASSET_MODE_GLOBAL = "global"
ASSET_MODE_PER_PHONEME = "per_phoneme"
ASSET_MODES = (ASSET_MODE_AUTO, ASSET_MODE_GLOBAL, ASSET_MODE_PER_PHONEME)


@dataclass(frozen=True)
class AssetCandidate:
    word: str
    gender: str
    voice_mode: str
    generation_mode: str
    phoneme_values: tuple[float, ...]
    audio_path: Path
    sidecar_path: Path
    source_phonemes: tuple[str, ...]
    label_id: int | None
    raw: dict[str, Any]


def clamp_unit(value: float) -> float:
    return max(0.0, min(1.0, value))


def round_grade(value: float) -> float:
    return round(clamp_unit(float(value)) + 1e-8, 1)


def normalize_word(value: str) -> str:
    return str(value).strip().lower()


def normalize_voice_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "female": "woman",
        "woman": "woman",
        "male": "man",
        "man": "man",
    }
    if normalized not in aliases:
        raise ValueError("Voice must be one of: female, woman, male, man.")
    return aliases[normalized]


def normalize_gender(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "female": "female",
        "woman": "female",
        "male": "male",
        "man": "male",
    }
    if normalized not in aliases:
        raise ValueError("Gender must be one of: female, male.")
    return aliases[normalized]


def gender_to_voice_mode(gender: str) -> str:
    return "woman" if normalize_gender(gender) == "female" else "man"


def _infer_voice_mode_from_paths(*paths: Path) -> str | None:
    aliases = {
        "female": "woman",
        "woman": "woman",
        "male": "man",
        "man": "man",
    }
    for path in paths:
        for part in path.parts:
            normalized = str(part).strip().lower()
            if normalized in aliases:
                return aliases[normalized]
    return None


def _resolve_effective_voice_mode(
    explicit_voice_value: Any,
    *paths: Path,
) -> str:
    inferred = _infer_voice_mode_from_paths(*paths)
    if explicit_voice_value is not None and str(explicit_voice_value).strip():
        explicit = normalize_voice_mode(str(explicit_voice_value))
        if inferred is not None and inferred != explicit:
            return inferred
        return explicit
    if inferred is not None:
        return inferred
    raise ValueError("Voice must be one of: female, woman, male, man.")


def _resolve_effective_gender(
    explicit_gender_value: Any,
    explicit_voice_value: Any,
    *paths: Path,
) -> str:
    if explicit_gender_value is not None and str(explicit_gender_value).strip():
        return normalize_gender(str(explicit_gender_value))
    if explicit_voice_value is not None and str(explicit_voice_value).strip():
        return normalize_gender(str(explicit_voice_value))
    inferred_voice_mode = _infer_voice_mode_from_paths(*paths)
    if inferred_voice_mode is not None:
        return normalize_gender(inferred_voice_mode)
    raise ValueError("Gender must be one of: female, male.")


def normalize_asset_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized == "global_soft":
        return ASSET_MODE_GLOBAL
    if normalized not in ASSET_MODES:
        raise ValueError(f"Generation mode must be one of: {', '.join(ASSET_MODES)}.")
    return normalized


def _resolve_path(path_value: str | Path, base_dir: Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _coerce_float_list(values: Sequence[Any], field_name: str) -> tuple[float, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"'{field_name}' must be a sequence of numbers.")
    return tuple(float(value) for value in values)


def _coerce_string_list(values: Sequence[Any], field_name: str) -> tuple[str, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"'{field_name}' must be a sequence of strings.")
    return tuple(str(value) for value in values)


def _flatten_asset_index_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        flattened: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("Asset index entries must be JSON objects.")
            nested_items = item.get("items")
            if isinstance(nested_items, list):
                for nested in nested_items:
                    if not isinstance(nested, dict):
                        raise ValueError("Nested asset index items must be JSON objects.")
                    merged = dict(item)
                    merged.update(nested)
                    merged.pop("items", None)
                    flattened.append(merged)
            else:
                flattened.append(dict(item))
        return flattened
    raise ValueError("Asset index JSON must contain a top-level array.")


def load_asset_index(index_path: Path) -> list[AssetCandidate]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    rows = _flatten_asset_index_payload(payload)
    base_dir = index_path.parent
    candidates: list[AssetCandidate] = []

    for row in rows:
        word_value = row.get("word") or row.get("transcript") or row.get("label_name")
        gender_value = row.get("gender")
        voice_value = row.get("voice_mode") or row.get("voice") or row.get("gender")
        generation_mode = row.get("generation_mode") or row.get("mode") or row.get("bank") or ASSET_MODE_PER_PHONEME
        audio_value = row.get("audio_path") or row.get("output_audio_path") or row.get("file")
        sidecar_value = row.get("sidecar_path") or row.get("json_path") or row.get("metadata_path")
        phoneme_values = row.get("phoneme_values")
        if phoneme_values is None and normalize_asset_mode(str(generation_mode)) == ASSET_MODE_GLOBAL:
            quality_value = row.get("quality")
            if quality_value is not None:
                phoneme_values = [quality_value]
        source_phonemes = row.get("source_phonemes")

        if not word_value:
            raise ValueError("Each asset index row must define 'word', 'transcript', or 'label_name'.")
        if not audio_value:
            raise ValueError(f"Asset index row for '{word_value}' is missing 'audio_path'.")
        if not sidecar_value:
            raise ValueError(f"Asset index row for '{word_value}' is missing 'sidecar_path' or 'json_path'.")
        if phoneme_values is None:
            raise ValueError(f"Asset index row for '{word_value}' is missing 'phoneme_values'.")
        if source_phonemes is None:
            raise ValueError(f"Asset index row for '{word_value}' is missing 'source_phonemes'.")

        label_id = row.get("label_id")
        resolved_audio_path = _resolve_path(audio_value, base_dir)
        resolved_sidecar_path = _resolve_path(sidecar_value, base_dir)
        resolved_gender = _resolve_effective_gender(
            gender_value,
            voice_value,
            index_path,
            resolved_audio_path,
            resolved_sidecar_path,
        )
        resolved_voice_mode = gender_to_voice_mode(resolved_gender)
        candidates.append(
            AssetCandidate(
                word=normalize_word(str(word_value)),
                gender=resolved_gender,
                voice_mode=resolved_voice_mode,
                generation_mode=normalize_asset_mode(str(generation_mode)),
                phoneme_values=_coerce_float_list(phoneme_values, "phoneme_values"),
                audio_path=resolved_audio_path,
                sidecar_path=resolved_sidecar_path,
                source_phonemes=_coerce_string_list(source_phonemes, "source_phonemes"),
                label_id=int(label_id) if label_id is not None else None,
                raw=dict(row),
            )
        )
    return candidates


def scan_asset_root(asset_root: Path) -> list[AssetCandidate]:
    candidates: list[AssetCandidate] = []
    for generation_mode in (ASSET_MODE_GLOBAL, ASSET_MODE_PER_PHONEME):
        mode_root = asset_root / generation_mode
        if not mode_root.is_dir():
            continue
        for sidecar_path in sorted(mode_root.glob("*/*.json")):
            payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
            word_value = payload.get("word") or payload.get("transcript") or metadata.get("transcript")
            source_phonemes = payload.get("source_phonemes")
            phoneme_values = payload.get("phoneme_values")
            if not word_value or source_phonemes is None or phoneme_values is None:
                continue
            audio_value = payload.get("output_audio_path") or sidecar_path.with_suffix(".wav").name
            label_id = payload.get("label_id", metadata.get("label_id"))
            resolved_audio_path = _resolve_path(audio_value, sidecar_path.parent)
            resolved_gender = _resolve_effective_gender(
                payload.get("gender", metadata.get("gender")),
                metadata.get("voice_mode"),
                asset_root,
                resolved_audio_path,
                sidecar_path.resolve(),
            )
            resolved_voice_mode = gender_to_voice_mode(resolved_gender)
            candidates.append(
                AssetCandidate(
                    word=normalize_word(str(word_value)),
                    gender=resolved_gender,
                    voice_mode=resolved_voice_mode,
                    generation_mode=generation_mode,
                    phoneme_values=_coerce_float_list(phoneme_values, "phoneme_values"),
                    audio_path=resolved_audio_path,
                    sidecar_path=sidecar_path.resolve(),
                    source_phonemes=_coerce_string_list(source_phonemes, "source_phonemes"),
                    label_id=int(label_id) if label_id is not None else None,
                    raw={
                        "word": str(word_value),
                        "transcript": str(word_value),
                        "gender": resolved_gender,
                        "voice_mode": resolved_voice_mode,
                        "generation_mode": generation_mode,
                        "audio_path": str(resolved_audio_path),
                        "sidecar_path": str(sidecar_path.resolve()),
                        "source_phonemes": list(source_phonemes),
                        "phoneme_values": list(phoneme_values),
                    },
                )
            )
    return candidates


def load_comparison_result(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".json", ".jsonl"}:
        return json.loads(path.read_text(encoding="utf-8"))
    if suffix in {".pkl", ".pickle"}:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("Pickle comparison result must contain a dictionary payload.")
        return payload
    raise ValueError("Comparison result must be .json, .jsonl, .pkl, or .pickle.")


def _coerce_series(values: Any, field_name: str) -> list[float]:
    if values is None:
        raise ValueError(f"Comparison result is missing '{field_name}'.")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"Comparison field '{field_name}' must be a sequence.")
    return [float(value) for value in values]


def adapt_comparison_result(payload: dict[str, Any]) -> dict[str, Any]:
    times = _coerce_series(payload.get("times"), "times")
    per_time_l2 = _coerce_series(payload.get("per_time_l2"), "per_time_l2")
    if len(times) != len(per_time_l2):
        raise ValueError(
            f"Comparison result length mismatch: len(times)={len(times)} vs len(per_time_l2)={len(per_time_l2)}."
        )
    label_name = payload.get("label_name")
    label_id = payload.get("label_id")
    return {
        "times": times,
        "per_time_l2": per_time_l2,
        "label_name": str(label_name).strip() if label_name is not None else None,
        "label_id": int(label_id) if label_id is not None else None,
        "dtw_cost": float(payload["dtw_cost"]) if payload.get("dtw_cost") is not None else None,
        "source": "template_l2_compare_v2" if "dtw_path" in payload or "warped_input_seq" in payload else "template_l2_compare",
        "raw": payload,
    }


def _time_bin_edges(times: Sequence[float]) -> list[float]:
    if not times:
        return [0.0, 0.0]
    if len(times) == 1:
        center = float(times[0])
        return [max(0.0, center - 0.5), center + 0.5]

    edges: list[float] = [max(0.0, float(times[0]) - ((float(times[1]) - float(times[0])) * 0.5))]
    for left, right in zip(times, times[1:]):
        edges.append((float(left) + float(right)) * 0.5)
    final_step = float(times[-1]) - float(times[-2])
    edges.append(float(times[-1]) + (final_step * 0.5))
    return edges


def _overlap(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _weighted_span_mean(times: Sequence[float], values: Sequence[float], start_sec: float, end_sec: float) -> float:
    if not times:
        return 0.0

    edges = _time_bin_edges(times)
    weighted_sum = 0.0
    total_weight = 0.0

    for index, value in enumerate(values):
        bin_start = edges[index]
        bin_end = edges[index + 1]
        weight = _overlap(bin_start, bin_end, start_sec, end_sec)
        if weight > 0.0:
            weighted_sum += float(value) * weight
            total_weight += weight

    if total_weight > 0.0:
        return weighted_sum / total_weight

    span_center = (start_sec + end_sec) * 0.5
    nearest_index = min(range(len(times)), key=lambda idx: abs(float(times[idx]) - span_center))
    return float(values[nearest_index])


def compute_per_phoneme_mismatch(
    times: Sequence[float],
    per_time_l2: Sequence[float],
    source_alignment: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    phoneme_scores: list[dict[str, Any]] = []
    for index, span in enumerate(source_alignment):
        start_sec = float(span["start_sec"])
        end_sec = float(span["end_sec"])
        mean_mismatch = _weighted_span_mean(times, per_time_l2, start_sec, end_sec)
        phoneme_scores.append(
            {
                "index": index,
                "phone": str(span["phone"]),
                "start_sec": start_sec,
                "end_sec": end_sec,
                "mean_mismatch": mean_mismatch,
                "rounded_grade": round_grade(mean_mismatch),
            }
        )
    return phoneme_scores


def _selection_distance(requested: Sequence[float], candidate: Sequence[float]) -> tuple[float, float]:
    deltas = [abs(float(left) - float(right)) for left, right in zip(requested, candidate)]
    return (sum(deltas), max(deltas) if deltas else 0.0)


def _requested_grade_signature(
    phoneme_scores: Sequence[dict[str, Any]],
    generation_mode: str,
) -> tuple[float, ...]:
    if normalize_asset_mode(generation_mode) == ASSET_MODE_GLOBAL:
        if not phoneme_scores:
            return (0.0,)
        mean_grade = sum(float(item["mean_mismatch"]) for item in phoneme_scores) / float(len(phoneme_scores))
        return (clamp_unit(mean_grade),)
    return tuple(float(item["rounded_grade"]) for item in phoneme_scores)


def _build_selection_result(
    comparison: dict[str, Any],
    candidate: AssetCandidate,
    sidecar: dict[str, Any],
    requested_word: str,
    requested_voice: str,
    phoneme_scores: Sequence[dict[str, Any]],
    exact_match: bool,
) -> dict[str, Any]:
    requested_grades = [float(item["rounded_grade"]) for item in phoneme_scores]
    requested_signature = _requested_grade_signature(phoneme_scores, candidate.generation_mode)
    distance_sum, distance_max = _selection_distance(requested_signature, candidate.phoneme_values)
    return {
        "word": requested_word,
        "gender": candidate.gender,
        "voice_mode": requested_voice,
        "generation_mode": candidate.generation_mode,
        "comparison_source": comparison["source"],
        "comparison_label_name": comparison.get("label_name"),
        "comparison_label_id": comparison.get("label_id"),
        "comparison_dtw_cost": comparison.get("dtw_cost"),
        "audio_path": str(candidate.audio_path),
        "sidecar_path": str(candidate.sidecar_path),
        "source_phonemes": list(candidate.source_phonemes),
        "requested_phoneme_grades": requested_grades,
        "requested_generation_signature": list(requested_signature),
        "selected_phoneme_grades": [float(value) for value in candidate.phoneme_values],
        "phoneme_scores": list(phoneme_scores),
        "per_time_l2": list(comparison["per_time_l2"]),
        "times": list(comparison["times"]),
        "exact_match": exact_match,
        "selection_distance_l1": distance_sum,
        "selection_distance_max": distance_max,
        "asset_metadata": candidate.raw,
        "asset_sidecar": sidecar,
    }


def select_asset(
    comparison: dict[str, Any],
    candidates: Sequence[AssetCandidate],
    requested_voice: str,
    target_word: str | None = None,
    target_label_id: int | None = None,
    generation_mode: str = ASSET_MODE_AUTO,
) -> dict[str, Any]:
    resolved_word = normalize_word(
        target_word or comparison.get("label_name") or ""
    )
    if not resolved_word and target_label_id is None and comparison.get("label_id") is None:
        raise ValueError("A target word, target label id, or comparison label must be available for selection.")

    normalized_generation_mode = normalize_asset_mode(generation_mode)
    requested_gender = normalize_gender(requested_voice)
    resolved_label_id = target_label_id if target_label_id is not None else comparison.get("label_id")
    filtered = [
        candidate
        for candidate in candidates
        if candidate.gender == requested_gender
        and (normalized_generation_mode == ASSET_MODE_AUTO or candidate.generation_mode == normalized_generation_mode)
        and (
            candidate.word == resolved_word
            or (resolved_label_id is not None and candidate.label_id == resolved_label_id)
        )
    ]
    if not filtered:
        raise ValueError(
            f"No assets found for word='{resolved_word or '<unknown>'}' voice='{requested_voice}' mode='{normalized_generation_mode}'."
        )

    sidecars: list[tuple[AssetCandidate, dict[str, Any]]] = []
    for candidate in filtered:
        sidecar = json.loads(candidate.sidecar_path.read_text(encoding="utf-8"))
        sidecars.append((candidate, sidecar))

    reference_candidate, reference_sidecar = sidecars[0]
    source_alignment = reference_sidecar.get("source_alignment")
    if not isinstance(source_alignment, list) or not source_alignment:
        raise ValueError(f"Asset sidecar '{reference_candidate.sidecar_path}' is missing 'source_alignment'.")

    phoneme_scores = compute_per_phoneme_mismatch(
        comparison["times"],
        comparison["per_time_l2"],
        source_alignment,
    )
    compatible: list[tuple[AssetCandidate, dict[str, Any]]] = []
    for candidate, sidecar in sidecars:
        requested_signature = _requested_grade_signature(phoneme_scores, candidate.generation_mode)
        if len(candidate.phoneme_values) != len(requested_signature):
            continue
        compatible.append((candidate, sidecar))
    if not compatible:
        raise ValueError(
            f"No assets for word='{resolved_word or '<unknown>'}' voice='{requested_voice}' match the phoneme count "
            f"{len(requested_grades)}."
        )

    exact = [
        (candidate, sidecar)
        for candidate, sidecar in compatible
        if candidate.phoneme_values == _requested_grade_signature(phoneme_scores, candidate.generation_mode)
    ]
    if exact:
        exact.sort(key=lambda item: str(item[0].audio_path))
        chosen_candidate, chosen_sidecar = exact[0]
        return _build_selection_result(
            comparison,
            chosen_candidate,
            chosen_sidecar,
            resolved_word,
            requested_voice,
            phoneme_scores,
            exact_match=True,
        )

    ranked = sorted(
        compatible,
        key=lambda item: (
            _selection_distance(
                _requested_grade_signature(phoneme_scores, item[0].generation_mode),
                item[0].phoneme_values,
            ),
            0 if item[0].generation_mode == ASSET_MODE_PER_PHONEME else 1,
            str(item[0].audio_path),
        ),
    )
    chosen_candidate, chosen_sidecar = ranked[0]
    return _build_selection_result(
        comparison,
        chosen_candidate,
        chosen_sidecar,
        resolved_word,
        requested_voice,
        phoneme_scores,
        exact_match=False,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select a local blabber asset for a known target word from a Template_l2_compare "
            "or Template_l2_compare_v2 comparison result."
        )
    )
    parser.add_argument(
        "--comparison-result",
        required=True,
        help="Path to a JSON or pickle payload returned by compare_signal_to_prebuilt_template().",
    )
    parser.add_argument(
        "--asset-index",
        default=str(DEFAULT_ASSET_INDEX_PATH),
        help="Path to a local JSON asset index. Defaults to blabber_asset_index.json beside this script.",
    )
    parser.add_argument(
        "--asset-root",
        default="",
        help="Optional asset-root directory to scan when the asset index is missing.",
    )
    parser.add_argument(
        "--voice",
        required=True,
        help="Requested voice bank: female/woman or male/man.",
    )
    parser.add_argument(
        "--generation-mode",
        choices=ASSET_MODES,
        default=ASSET_MODE_AUTO,
        help="Limit selection to a specific output bank or search across all assets.",
    )
    parser.add_argument(
        "--word",
        default="",
        help="Known target word. If omitted, label_name from the comparison result is used.",
    )
    parser.add_argument(
        "--label-id",
        type=int,
        default=None,
        help="Optional label id override used when the asset index includes label_id metadata.",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write the selection result as JSON.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    comparison_path = _resolve_path(args.comparison_result, ROOT)
    asset_index_path = _resolve_path(args.asset_index, ROOT)
    asset_root = _resolve_path(args.asset_root, ROOT) if args.asset_root.strip() else asset_index_path.parent
    requested_voice = normalize_voice_mode(args.voice)

    comparison = adapt_comparison_result(load_comparison_result(comparison_path))
    if asset_index_path.is_file():
        candidates = load_asset_index(asset_index_path)
    else:
        candidates = scan_asset_root(asset_root)
    result = select_asset(
        comparison=comparison,
        candidates=candidates,
        requested_voice=requested_voice,
        target_word=args.word.strip() or None,
        target_label_id=args.label_id,
        generation_mode=str(args.generation_mode),
    )

    output_json = json.dumps(result, indent=2)
    if args.output_json.strip():
        output_path = _resolve_path(args.output_json, ROOT)
        output_path.write_text(output_json + "\n", encoding="utf-8")
    print(output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
