# Intuition Guide

This is the intuition for the two scripts:

- `deep_phone_candidate_stack_blabber_triphone.py`
- `my_test.py`

---

## 1) What `deep_phone_candidate_stack_blabber_triphone.py` does

Think of this script as a **word distortion generator**.

Input:
- A base word (`yes`, `no`, `help`, etc.)
- A level from `0` to `1`

Output:
- Many candidate pronunciations (phone sequences)
- Metadata for search/ranking
- Optional audio renderings

TTS backends currently supported by the generator:
- `piper`
- `edge` via `edge-tts` plus `ffmpeg`
- `kokoro`
- `coqui`

### Level intuition
- `level = 0` means very close to the original word.
- `level = 1` means stronger distortion (more changed phones allowed).

### Important behavior
- The base/canonical word is kept visible at each level (unless disabled).
- Each candidate stores:
  - `level`
  - `phone_distance` (how phonetically changed it is)
  - `phone_similarity`
  - optional `neural_similarity`

So this script builds the **candidate library** and describes how distorted each candidate is.

---

## 2) What `my_test.py` does

Think of this script as the template compare and asset selection **controller**:

1. Compare EEG sample vs template.
2. Turn mismatch into a target distortion strength.
3. Choose a candidate audio asset.

It does not generate candidates. It chooses from the prebuilt library.

---

## 3) How distance becomes a target strength

From EEG comparison, you get `per_time_l2`.

`my_test.py` does:

1. Compute global summary (`mean` and `p90` of L2).
2. Blend them (`L2_BLEND_MEAN`, `L2_BLEND_P90`).
3. Normalize into `[0, 1]` using (`L2_LEVEL_MIN`, `L2_LEVEL_MAX`).
4. Apply a deadband (`LOW_DISTANCE_DEADBAND`) so tiny errors map to near-zero strength.

Result:
- `target_strength` in `[0,1]`
- higher = allow stronger augmentation

---

## 4) How candidate choice is made (two-stage ranking)

### Stage A: base ranking (selector)
`select_blabber_asset_triphone.py` ranks candidates with word/voice constraints and distance-targeted scoring.

This gives top candidates.

### Stage B: temporal reranking (inside `my_test.py`)
From the top pool, `my_test.py` compares:

- `temporal_target_profile` (from EEG `per_time_l2` over time)
- `candidate_temporal_profile` (from candidate phone edits over time)

Then it computes:
- `temporal_mse` (lower is better)
- `temporal_match = 1 - clipped_mse` (higher is better)

Final score:
- `combined_score = base_score + TEMPORAL_MATCH_WEIGHT * temporal_match`

So the final pick is not only "how strong globally", but also "does this candidate's distortion shape match where EEG mismatch happens across time".

---

## 5) Quick tuning: softer vs harsher

If choices feel too harsh:
- increase `LOW_DISTANCE_DEADBAND`
- decrease `TRIPHONE_LEVEL_WEIGHT`
- decrease `TEMPORAL_MATCH_WEIGHT`
- optionally reduce `TEMPORAL_TOP_K`

If choices feel too soft:
- decrease `LOW_DISTANCE_DEADBAND`
- increase `TRIPHONE_LEVEL_WEIGHT`
- increase `TEMPORAL_MATCH_WEIGHT`
- optionally increase `TEMPORAL_TOP_K`

---

## 6) How to read logs quickly

When you see:
- `target_level` / `target_strength`: global desired distortion
- `score_details.temporal_match`: how well timing of distortion matched EEG mismatch
- `candidate_temporal_profile`: candidate distortion shape over time
- `phone_distance` and `level`: global distortion amount for that candidate

Fast check:
- If `temporal_match` is low for selected candidates, time-shape fit is weak.
- If selected candidates are too strong despite low mismatch, deadband/weights are too aggressive.

## 7) Edge TTS rendering

Install:

```bash
pip install edge-tts
```

You also need `ffmpeg` on `PATH` because the generator converts Edge's MP3 output
into the same WAV files used by the current scoring and library flow.

Australian female:

```bash
python deep_phone_candidate_stack_blabber_triphone.py \
  --tts-engine edge \
  --edge-voice en-AU-NatashaNeural \
  --word yes \
  --levels 8 \
  --render-audio \
  --skip-neural \
  --json-out yes_edge_au_female.json
```

Australian male:

```bash
python deep_phone_candidate_stack_blabber_triphone.py \
  --tts-engine edge \
  --edge-voice en-AU-WilliamNeural \
  --word yes \
  --levels 8 \
  --render-audio \
  --skip-neural \
  --json-out yes_edge_au_male.json
```

Both voices in one library run:

```bash
python deep_phone_candidate_stack_blabber_triphone.py \
  --tts-engine edge \
  --edge-voices en-AU-NatashaNeural,en-AU-WilliamNeural \
  --word yes \
  --levels 8 \
  --render-audio \
  --skip-neural \
  --json-out yes_edge_au_both.json
```
