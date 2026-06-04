#!/usr/bin/env python3
"""
Check that text-only triphone combination files capture all unique graphemes.

This validates files like:
  /projects/SSNFB/Ray/audio-assets/text-only/text/triphone/{word}/{word}_combinations.txt

against the unique grapheme tokens implied by a triphone WAV asset tree like:
  /data/raqchia/audio-assets/speech-assets-triphone/man/triphone/{word}/level_*/0001_L1.00_grapheme_piper.wav

Typical use:
  python3 check_text_only_unique_graphemes.py

Check the female/RayFem output instead:
  python3 check_text_only_unique_graphemes.py \
    --assets-root /data/raqchia/audio-assets/speech-assets-triphone/woman/triphone \
    --voice-suffix en-AU-RayFem

Write a CSV report:
  python3 check_text_only_unique_graphemes.py --report-csv text_only_grapheme_check.csv

Exit status:
  0 = no required graphemes missing and no unparsable asset filenames
  1 = missing graphemes, unparsable asset filenames, or selected strict checks failed
  2 = configuration/input error
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


DEFAULT_ASSETS_ROOT = Path("/data/raqchia/audio-assets/speech-assets-triphone/man/triphone")
DEFAULT_TEXT_ROOT = Path("/projects/SSNFB/Ray/audio-assets/text-only/text/triphone")


@dataclass
class WordCheck:
    word: str
    asset_unique: Set[str]
    text_lines: List[str]
    text_unique: Set[str]
    missing_from_text: Set[str]
    extra_in_text: Set[str]
    duplicate_text: Dict[str, int]
    asset_files_seen: int
    unmatched_asset_files: List[str]
    text_file: Optional[Path]


def normalize_grapheme(text: str, *, case_sensitive: bool) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if not case_sensitive:
        text = text.lower()
    return text


def build_asset_pattern(voice_suffix: Optional[str]) -> re.Pattern[str]:
    # Expected filename:
    #   0058_L1.00_yehz_piper.wav
    #   0058_L1.00_yehz_en-AU-RayFem.wav
    #
    # The grapheme capture is greedy so tokens containing underscores still work
    # as long as the voice suffix is known.
    if voice_suffix:
        escaped_voice = re.escape(voice_suffix)
        return re.compile(
            rf"^(?P<index>\d+)_L(?P<level>\d+(?:\.\d+)?)_(?P<grapheme>.+)_{escaped_voice}\.wav$",
            re.IGNORECASE,
        )

    # Fallback if checking all voices. This assumes the voice suffix is the final
    # underscore-separated component. Prefer --voice-suffix for exact checking.
    return re.compile(
        r"^(?P<index>\d+)_L(?P<level>\d+(?:\.\d+)?)_(?P<grapheme>.+)_(?P<voice>[^_]+)\.wav$",
        re.IGNORECASE,
    )


def discover_words(assets_root: Path, text_root: Path, requested_words: Optional[Sequence[str]]) -> List[str]:
    if requested_words:
        return sorted({w.strip().lower() for w in requested_words if w.strip()})

    words: Set[str] = set()
    if assets_root.exists():
        for p in assets_root.iterdir():
            if p.is_dir():
                words.add(p.name.lower())

    if text_root.exists():
        for p in text_root.iterdir():
            if p.is_dir():
                words.add(p.name.lower())
            elif p.name.endswith("_combinations.txt"):
                words.add(p.name[: -len("_combinations.txt")].lower())

    return sorted(words)


def text_file_candidates(text_root: Path, word: str) -> List[Path]:
    return [
        text_root / word / f"{word}_combinations.txt",
        text_root / f"{word}_combinations.txt",
    ]


def read_text_graphemes(
    text_root: Path,
    word: str,
    *,
    case_sensitive: bool,
    ignore_comments: bool,
) -> Tuple[Optional[Path], List[str]]:
    for path in text_file_candidates(text_root, word):
        if path.exists() and path.is_file():
            lines: List[str] = []
            for raw in path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line:
                    continue
                if ignore_comments and line.startswith("#"):
                    continue
                lines.append(normalize_grapheme(line, case_sensitive=case_sensitive))
            return path, lines
    return None, []


def collect_asset_graphemes(
    assets_root: Path,
    word: str,
    pattern: re.Pattern[str],
    *,
    case_sensitive: bool,
) -> Tuple[Set[str], int, List[str]]:
    word_dir = assets_root / word
    unique: Set[str] = set()
    seen = 0
    unmatched: List[str] = []

    if not word_dir.exists():
        return unique, seen, unmatched

    for wav_path in sorted(word_dir.glob("level_*/*.wav")):
        seen += 1
        m = pattern.match(wav_path.name)
        if not m:
            unmatched.append(str(wav_path))
            continue
        unique.add(normalize_grapheme(m.group("grapheme"), case_sensitive=case_sensitive))

    return unique, seen, unmatched


def check_word(
    word: str,
    assets_root: Path,
    text_root: Path,
    pattern: re.Pattern[str],
    *,
    case_sensitive: bool,
    ignore_comments: bool,
) -> WordCheck:
    asset_unique, asset_files_seen, unmatched_asset_files = collect_asset_graphemes(
        assets_root,
        word,
        pattern,
        case_sensitive=case_sensitive,
    )
    text_file, text_lines = read_text_graphemes(
        text_root,
        word,
        case_sensitive=case_sensitive,
        ignore_comments=ignore_comments,
    )
    counts = Counter(text_lines)
    text_unique = set(text_lines)
    duplicate_text = {k: v for k, v in counts.items() if v > 1}

    return WordCheck(
        word=word,
        asset_unique=asset_unique,
        text_lines=text_lines,
        text_unique=text_unique,
        missing_from_text=asset_unique - text_unique,
        extra_in_text=text_unique - asset_unique,
        duplicate_text=duplicate_text,
        asset_files_seen=asset_files_seen,
        unmatched_asset_files=unmatched_asset_files,
        text_file=text_file,
    )


def short_list(values: Iterable[str], limit: int) -> str:
    vals = sorted(values)
    if not vals:
        return ""
    shown = vals[:limit]
    suffix = "" if len(vals) <= limit else f" ... (+{len(vals) - limit} more)"
    return ", ".join(shown) + suffix


def print_report(results: Sequence[WordCheck], *, list_limit: int) -> None:
    total_missing = sum(len(r.missing_from_text) for r in results)
    total_extra = sum(len(r.extra_in_text) for r in results)
    total_dupes = sum(len(r.duplicate_text) for r in results)
    total_unmatched = sum(len(r.unmatched_asset_files) for r in results)

    print("\n=== text-only grapheme coverage check ===")
    print(f"words checked:          {len(results)}")
    print(f"missing graphemes:      {total_missing}")
    print(f"extra text-only lines:  {total_extra}")
    print(f"duplicate text entries: {total_dupes}")
    print(f"unparsable WAV names:   {total_unmatched}")
    print("")

    header = (
        f"{'word':<14} {'asset_unique':>12} {'text_unique':>11} "
        f"{'missing':>8} {'extra':>7} {'dupes':>7} {'wav_files':>9}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r.word:<14} {len(r.asset_unique):>12} {len(r.text_unique):>11} "
            f"{len(r.missing_from_text):>8} {len(r.extra_in_text):>7} "
            f"{len(r.duplicate_text):>7} {r.asset_files_seen:>9}"
        )

        if r.text_file is None:
            print("  TEXT FILE MISSING")
        if r.missing_from_text:
            print(f"  missing: {short_list(r.missing_from_text, list_limit)}")
        if r.extra_in_text:
            print(f"  extra:   {short_list(r.extra_in_text, list_limit)}")
        if r.duplicate_text:
            dupes = [f"{k} x{v}" for k, v in sorted(r.duplicate_text.items())]
            print(f"  dupes:   {short_list(dupes, list_limit)}")
        if r.unmatched_asset_files:
            print(f"  bad wav: {short_list(r.unmatched_asset_files, min(list_limit, 3))}")


def write_csv(path: Path, results: Sequence[WordCheck]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "word",
                "asset_unique_count",
                "text_unique_count",
                "asset_files_seen",
                "text_file",
                "missing_count",
                "missing_graphemes",
                "extra_count",
                "extra_graphemes",
                "duplicate_count",
                "duplicate_graphemes",
                "unmatched_wav_count",
                "unmatched_wav_files",
            ],
        )
        writer.writeheader()
        for r in results:
            writer.writerow(
                {
                    "word": r.word,
                    "asset_unique_count": len(r.asset_unique),
                    "text_unique_count": len(r.text_unique),
                    "asset_files_seen": r.asset_files_seen,
                    "text_file": "" if r.text_file is None else str(r.text_file),
                    "missing_count": len(r.missing_from_text),
                    "missing_graphemes": "|".join(sorted(r.missing_from_text)),
                    "extra_count": len(r.extra_in_text),
                    "extra_graphemes": "|".join(sorted(r.extra_in_text)),
                    "duplicate_count": len(r.duplicate_text),
                    "duplicate_graphemes": "|".join(f"{k}:{v}" for k, v in sorted(r.duplicate_text.items())),
                    "unmatched_wav_count": len(r.unmatched_asset_files),
                    "unmatched_wav_files": "|".join(r.unmatched_asset_files),
                }
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Verify text-only triphone combination files contain all unique graphemes from a WAV asset tree."
    )
    ap.add_argument("--assets-root", type=Path, default=DEFAULT_ASSETS_ROOT)
    ap.add_argument("--text-root", type=Path, default=DEFAULT_TEXT_ROOT)
    ap.add_argument(
        "--voice-suffix",
        default="piper",
        help=(
            "Filename suffix to parse after the grapheme, without .wav. "
            "Use piper for the original template tree, or en-AU-RayFem for the female output tree. "
            "Use an empty string to parse any final underscore suffix."
        ),
    )
    ap.add_argument("--word", dest="words", action="append", help="Check only this source word. Can be repeated.")
    ap.add_argument("--case-sensitive", action="store_true")
    ap.add_argument("--ignore-comments", action="store_true", default=True)
    ap.add_argument("--list-limit", type=int, default=25)
    ap.add_argument("--report-csv", type=Path, default=None)
    ap.add_argument("--fail-on-extra", action="store_true")
    ap.add_argument("--fail-on-duplicates", action="store_true")
    ap.add_argument("--allow-unmatched-wav", action="store_true")
    args = ap.parse_args(argv)

    if not args.assets_root.exists():
        print(f"ERROR: assets root does not exist: {args.assets_root}", file=sys.stderr)
        return 2
    if not args.text_root.exists():
        print(f"ERROR: text root does not exist: {args.text_root}", file=sys.stderr)
        return 2

    voice_suffix = args.voice_suffix if args.voice_suffix else None
    pattern = build_asset_pattern(voice_suffix)

    words = discover_words(args.assets_root, args.text_root, args.words)
    if not words:
        print("ERROR: no words discovered to check", file=sys.stderr)
        return 2

    results = [
        check_word(
            word,
            args.assets_root,
            args.text_root,
            pattern,
            case_sensitive=args.case_sensitive,
            ignore_comments=args.ignore_comments,
        )
        for word in words
    ]

    print_report(results, list_limit=args.list_limit)

    if args.report_csv:
        write_csv(args.report_csv, results)
        print(f"\nCSV report written to: {args.report_csv}")

    has_missing = any(r.missing_from_text for r in results)
    has_extra = any(r.extra_in_text for r in results)
    has_dupes = any(r.duplicate_text for r in results)
    has_unmatched = any(r.unmatched_asset_files for r in results)

    if has_missing:
        print("\nFAIL: at least one required asset grapheme is missing from text-only output.")
    if has_unmatched and not args.allow_unmatched_wav:
        print("\nFAIL: at least one WAV filename could not be parsed. Check --voice-suffix.")
    if has_extra and args.fail_on_extra:
        print("\nFAIL: extra text-only graphemes found and --fail-on-extra was set.")
    if has_dupes and args.fail_on_duplicates:
        print("\nFAIL: duplicate text-only graphemes found and --fail-on-duplicates was set.")

    if has_missing:
        return 1
    if has_unmatched and not args.allow_unmatched_wav:
        return 1
    if has_extra and args.fail_on_extra:
        return 1
    if has_dupes and args.fail_on_duplicates:
        return 1

    print("\nPASS: text-only output captures every unique grapheme found in the asset tree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
