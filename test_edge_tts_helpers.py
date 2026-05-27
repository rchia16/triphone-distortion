import sys
import types
import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.modules.setdefault("numpy", types.SimpleNamespace(asarray=lambda value, dtype=None: value, float32="float32"))

from deep_phone_candidate_stack_blabber_triphone import (
    Candidate,
    EDGE_TTS_VOICES,
    make_blabber_asset,
    resolve_edge_voice_mode,
    resolve_tts_voices,
)
from cmu_per_phoneme import (
    distance_key,
    granularity_values,
    per_phoneme_eeg_targets,
    pregenerate_word_assets,
    preview_rows,
    preview_targets_for_phone_count,
    resolve_cmudict_phones,
    select_dynamic_phoneme_candidate,
)
from my_test import _cmvn, _dtw_distance


class EdgeTTSHelperTests(unittest.TestCase):
    def make_args(self, **overrides):
        defaults = {
            "tts_engine": "edge",
            "voice_mode": "auto",
            "edge_voice": None,
            "edge_voices": None,
            "edge_voice_mode": "female",
            "kokoro_voices": "bf_emma,bm_lewis",
            "piper_speaker": None,
            "coqui_speaker": None,
        }
        defaults.update(overrides)
        return SimpleNamespace(**defaults)

    def test_resolve_tts_voices_uses_default_au_female(self):
        args = self.make_args()
        self.assertEqual(resolve_tts_voices(args), [EDGE_TTS_VOICES["female"]])

    def test_resolve_tts_voices_prefers_explicit_edge_voice_list(self):
        args = self.make_args(edge_voices="en-AU-NatashaNeural,en-AU-WilliamNeural")
        self.assertEqual(
            resolve_tts_voices(args),
            ["en-AU-NatashaNeural", "en-AU-WilliamNeural"],
        )

    def test_resolve_edge_voice_mode_returns_none_for_multi_voice_run(self):
        args = self.make_args(edge_voices="en-AU-NatashaNeural,en-AU-WilliamNeural")
        self.assertIsNone(resolve_edge_voice_mode(args))

    def test_make_blabber_asset_infers_voice_mode_from_edge_voice_name(self):
        candidate = Candidate(
            word="yes",
            level=0.0,
            grapheme="yes",
            phones=["Y", "EH", "S"],
            phone_distance=0.0,
            phone_similarity=1.0,
            neural_distance=None,
            neural_similarity=None,
            suitability_score=1.0,
            operations=["ground_truth:canonical"],
            voice="en-AU-WilliamNeural",
        )
        asset = make_blabber_asset(candidate, canonical_phones=["Y", "EH", "S"], asset_index=0)
        self.assertEqual(asset["voice"], "en-AU-WilliamNeural")
        self.assertEqual(asset["voice_mode"], "man")
        self.assertEqual(asset["gender"], "male")

    def test_cmvn_zero_centers_each_feature_dimension(self):
        features = np.array([[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]], dtype=np.float32)
        normalized = _cmvn(features)
        self.assertTrue(np.allclose(normalized.mean(axis=1), 0.0, atol=1e-6))

    def test_dtw_distance_is_zero_for_identical_features(self):
        features = np.array([[0.1, 0.2, 0.3], [1.0, 0.5, 0.0]], dtype=np.float32)
        self.assertAlmostEqual(_dtw_distance(features, features), 0.0, places=7)

    def test_cmudict_resolution_strips_stress_markers(self):
        fake_cmudict = SimpleNamespace(dict=lambda: {"test": [["T", "EH1", "S", "T"]]})
        with patch.dict(sys.modules, {"cmudict": fake_cmudict}):
            self.assertEqual(resolve_cmudict_phones("test"), ["T", "EH", "S", "T"])

    def test_per_phoneme_eeg_targets_match_phone_count(self):
        comparison = {"per_time_l2": [0.06, 0.18, 0.30], "times": [0, 1, 2]}
        targets = per_phoneme_eeg_targets(comparison, 5)
        self.assertEqual(len(targets), 5)

    def test_dynamic_phoneme_selection_rejects_mapping_over_cmvn_threshold(self):
        assets = [
            {"asset_id": "canon", "phones": ["S"], "audio_path": "canon.wav", "voice_mode": "woman"},
            {"asset_id": "z", "phones": ["Z"], "audio_path": "z.wav", "voice_mode": "woman"},
            {"asset_id": "sh", "phones": ["SH"], "audio_path": "sh.wav", "voice_mode": "woman"},
        ]

        def fake_cmvn(**kwargs):
            return {"S": 0.0, "Z": 0.2, "SH": 0.4}.get(kwargs["generated_phone"], 1.0)

        selected = select_dynamic_phoneme_candidate(
            word="see",
            canonical=["S"],
            comparison={"per_time_l2": [0.30], "times": [0]},
            assets=assets,
            voice_mode="woman",
            cmvn_phone_max_threshold=0.3,
            phone_cmvn_distance_fn=fake_cmvn,
        )
        self.assertTrue(selected["valid"])
        self.assertEqual(selected["generated_phones"], ["Z"])
        self.assertTrue(selected["invalid_phoneme_candidates"])

    def test_dynamic_phoneme_selection_prefers_largest_valid_distance(self):
        assets = [
            {"asset_id": "canon", "phones": ["S"], "audio_path": "canon.wav", "voice_mode": "woman"},
            {"asset_id": "z", "phones": ["Z"], "audio_path": "z.wav", "voice_mode": "woman"},
            {"asset_id": "sh", "phones": ["SH"], "audio_path": "sh.wav", "voice_mode": "woman"},
        ]

        def fake_cmvn(**kwargs):
            return {"S": 0.0, "Z": 0.2, "SH": 0.4}.get(kwargs["generated_phone"], 1.0)

        selected = select_dynamic_phoneme_candidate(
            word="see",
            canonical=["S"],
            comparison={"per_time_l2": [0.30], "times": [0]},
            assets=assets,
            voice_mode="woman",
            cmvn_phone_max_threshold=0.5,
            phone_cmvn_distance_fn=fake_cmvn,
        )
        self.assertTrue(selected["valid"])
        self.assertEqual(selected["generated_phones"], ["SH"])

    def test_dynamic_phoneme_selection_creates_synthetic_reference_without_assets(self):
        selected = select_dynamic_phoneme_candidate(
            word="yes",
            canonical=["Y", "EH", "S"],
            comparison={"per_time_l2": [0.30, 0.06, 0.06], "times": [0, 1, 2]},
            assets=[],
            voice_mode="woman",
            cmvn_phone_max_threshold=0.4,
        )
        self.assertTrue(selected["valid"])
        self.assertTrue(selected["reference_asset"]["synthetic_asset"])
        self.assertNotEqual(selected["generated_phones"][0], "Y")
        self.assertEqual(selected["generated_phones"][1:], ["EH", "S"])
        self.assertEqual(selected["score_details"]["distance_metric"], "articulatory_fallback")

    def test_preview_targets_use_half_average_for_two_phones(self):
        self.assertEqual(preview_targets_for_phone_count([0.1, 0.3, 0.6, 0.9], 2), [0.2, 0.75])

    def test_preview_targets_use_overlapping_pair_average_for_three_phones(self):
        self.assertTrue(np.allclose(preview_targets_for_phone_count([0.1, 0.3, 0.6, 0.9], 3), [0.2, 0.45, 0.75]))

    def test_preview_rows_covers_full_vocab(self):
        rows = preview_rows(distance_signal=[0.1, 0.3, 0.6, 0.9])
        self.assertEqual([row["word"] for row in rows], ["yes", "no", "bath", "pain", "help", "food", "go", "stop"])
        self.assertTrue(all(row["generated_phones"] for row in rows))

    def test_granularity_values_include_endpoints(self):
        self.assertEqual(granularity_values(2), [0.0, 1.0])
        values = granularity_values(10)
        self.assertEqual(len(values), 10)
        self.assertAlmostEqual(values[0], 0.0)
        self.assertAlmostEqual(values[-1], 1.0)

    def test_distance_key_formats_wav_name_components(self):
        self.assertEqual(distance_key((0.0, 0.5, 1.0)), "0.0_0.5_1.0")

    def test_pregenerate_word_assets_metadata_only_uses_separated_phonemes(self):
        args = SimpleNamespace(
            render_audio=False,
            metadata_only=True,
            overwrite=False,
            tts_engine="coqui",
            kokoro_lang_code="b",
            kokoro_speed=1.0,
            edge_voice=None,
            edge_rate="+0%",
            edge_pitch="+0Hz",
            piper_model="piper-models/en_GB-jenny_dioco-medium.onnx",
            piper_config="piper-models/en_GB-jenny_dioco-medium.onnx.json",
            piper_exe="piper",
            piper_speaker=None,
            piper_use_python_module=False,
            piper_length_scale=1.2,
            piper_noise_scale=None,
            piper_noise_w=None,
            piper_sentence_silence=None,
            coqui_model="tts_models/en/ljspeech/tacotron2-DDC",
            coqui_speaker=None,
            coqui_language=None,
            speaker_wav=None,
            max_duration_sec=3.0,
            max_duration_ratio=3.0,
            allow_overlong=False,
        )
        with self.subTest("three_phone_word"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                payload = pregenerate_word_assets(
                    word="yes",
                    voice_mode="woman",
                    output_root=Path(tmp),
                    distance_values=[0.0, 1.0],
                    render_audio=False,
                    overwrite=False,
                    args=args,
                )
                self.assertEqual(payload["asset_count"], 8)
                first = payload["assets"][0]
                self.assertTrue(first["audio_path"].endswith("0.0_0.0_0.0.wav"))
                self.assertEqual(first["render_text"], "Y EH S")
                self.assertNotIn("yehs", first["render_text"].lower())


if __name__ == "__main__":
    unittest.main()
