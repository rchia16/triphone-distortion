#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_WORDS = ["go", "stop", "bath", "food", "yes", "no", "pain", "help"]
VOICE_MAP = {
    "woman": "en-AU-NatashaNeural",
    "man": "en-AU-WilliamNeural",
}


def parse_csv(text: str) -> list[str]:
    return [item.strip() for item in str(text).split(",") if item.strip()]


def build_candidate_payload(
    *,
    python_bin: str,
    processor: Path,
    word: str,
    levels: int,
    max_candidates: int,
    extra_args: list[str],
) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        json_path = Path(tmp.name)

    cmd = [
        python_bin,
        str(processor),
        "--word",
        word,
        "--levels",
        str(levels),
        "--max-candidates",
        str(max_candidates),
        "--skip-neural",
        "--json-out",
        str(json_path),
        *extra_args,
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return json.loads(json_path.read_text())
    finally:
        json_path.unlink(missing_ok=True)


def extract_candidate_lines(payload: dict) -> list[str]:
    assets = payload.get("assets") or payload.get("blabber_assets") or []
    if not isinstance(assets, list):
        return []

    lines: list[str] = []
    seen: set[tuple[str, float | None]] = set()
    for asset in assets:
        text = str(asset.get("candidate_text") or asset.get("grapheme") or "").strip()
        level = asset.get("level")
        key = (text, float(level) if level is not None else None)
        if not text or key in seen:
            continue
        seen.add(key)
        lines.append(text)
    return lines


def build_concat_text(word_to_lines: dict[str, list[str]], carrier: str | None) -> str:
    parts: list[str] = []
    for word, lines in word_to_lines.items():
        if not lines:
            continue
        joined = ". ".join(lines).strip()
        if not joined:
            continue
        if carrier:
            parts.append(carrier.format(word=word, candidates=joined))
        else:
            parts.append(joined + ".")
    return " ".join(parts).strip()


def render_edge_text(text: str, voice: str, out_path: Path, rate: str, pitch: str) -> None:
    try:
        import edge_tts
    except Exception as exc:
        raise RuntimeError("edge-tts is not installed. Install with: pip install edge-tts") from exc

    import asyncio

    async def _run() -> None:
        communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
        await communicate.save(str(out_path))

    asyncio.run(_run())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--processor", default="deep_phone_candidate_stack_blabber_triphone.py")
    parser.add_argument("--words", default=",".join(DEFAULT_WORDS))
    parser.add_argument("--voices", default="woman,man")
    parser.add_argument("--levels", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=64)
    parser.add_argument("--out-dir", default="edge_concat_test")
    parser.add_argument("--edge-rate", default="+0%")
    parser.add_argument("--edge-pitch", default="+0Hz")
    parser.add_argument(
        "--carrier",
        default="Say {word}. {candidates}.",
        help="Format string for each word block. Available fields: {word}, {candidates}. Use empty string to disable.",
    )
    parser.add_argument(
        "generator_args",
        nargs="*",
        help="Extra args forwarded to deep_phone_candidate_stack_blabber_triphone.py, e.g. --ground-truth-level 0.05",
    )
    args = parser.parse_args()

    processor = Path(args.processor)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    words = parse_csv(args.words)
    voices = parse_csv(args.voices)
    carrier = args.carrier if args.carrier else None

    word_to_lines: dict[str, list[str]] = {}
    for word in words:
        payload = build_candidate_payload(
            python_bin=args.python_bin,
            processor=processor,
            word=word,
            levels=args.levels,
            max_candidates=args.max_candidates,
            extra_args=args.generator_args,
        )
        word_to_lines[word] = extract_candidate_lines(payload)

    concat_text = build_concat_text(word_to_lines, carrier)
    if not concat_text:
        raise RuntimeError("No candidate text was generated for concatenated Edge TTS test.")

    (out_dir / "concat_text.txt").write_text(concat_text)
    (out_dir / "candidate_blocks.json").write_text(json.dumps(word_to_lines, indent=2))

    outputs: dict[str, str] = {}
    for voice_mode in voices:
        resolved_voice = VOICE_MAP.get(voice_mode, voice_mode)
        out_path = out_dir / f"concat_{voice_mode}.mp3"
        render_edge_text(
            text=concat_text,
            voice=resolved_voice,
            out_path=out_path,
            rate=args.edge_rate,
            pitch=args.edge_pitch,
        )
        outputs[voice_mode] = str(out_path)

    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "words": words,
                "voices": outputs,
                "text_path": str(out_dir / "concat_text.txt"),
                "candidate_blocks_path": str(out_dir / "candidate_blocks.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
