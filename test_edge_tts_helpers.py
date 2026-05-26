import sys
import types
import unittest
from types import SimpleNamespace

sys.modules.setdefault("numpy", types.SimpleNamespace(asarray=lambda value, dtype=None: value, float32="float32"))

from deep_phone_candidate_stack_blabber_triphone import (
    Candidate,
    EDGE_TTS_VOICES,
    make_blabber_asset,
    resolve_edge_voice_mode,
    resolve_tts_voices,
)


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


if __name__ == "__main__":
    unittest.main()
