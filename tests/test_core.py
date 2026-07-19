import tempfile
import unittest
from pathlib import Path

from subtitle_extractor.ensemble import (
    branch_medoid,
    conservative_finalize,
    cross_branch_agreement,
    exact_curated_override,
    rescale_words,
)
from subtitle_extractor.srt import Cue, parse_srt_text, read_srt, render_srt, write_srt
from subtitle_extractor.text import (
    frequent_ngrams,
    repetition_flags,
    serbian_cyrillic_to_latin,
    suspicious_repetition,
)


class SrtTests(unittest.TestCase):
    def test_parse_write_round_trip(self):
        cues = [
            Cue(0.0, 1.234, "Prvi red"),
            Cue(59.9996, 62.5, "Drugi\nred"),
        ]
        rendered = render_srt(cues)
        parsed = parse_srt_text(rendered)
        self.assertEqual(parsed[0], cues[0])
        self.assertEqual(parsed[1].start, 60.0)
        self.assertEqual(parsed[1].end, 62.5)
        self.assertEqual(parsed[1].text, "Drugi\nred")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.srt"
            write_srt(parsed, path)
            self.assertEqual(read_srt(path), parsed)

    def test_rejects_negative_duration_on_write(self):
        with self.assertRaises(ValueError):
            render_srt([Cue(2.0, 1.0, "bad")])


class SerbianTextTests(unittest.TestCase):
    def test_cyrillic_to_latin_digraphs(self):
        self.assertEqual(
            serbian_cyrillic_to_latin("Љуба Његош, ђак и Џони"),
            "Ljuba Njegoš, đak i Džoni",
        )

    def test_source_specific_repetition_flags(self):
        texts = [
            "Hvala što pratite kanal danas",
            "Hvala što pratite kanal opet",
            "Hvala što pratite kanal sada",
            "Hvala što pratite kanal ovde",
        ]
        frequent = frequent_ngrams(texts)
        self.assertIn("hvala što pratite", frequent)
        self.assertEqual(
            repetition_flags("Hvala što pratite kanal", frequent),
            ["hvala što pratite", "što pratite kanal"],
        )
        self.assertTrue(suspicious_repetition("women women women women women"))
        self.assertFalse(suspicious_repetition("Kako si putovao?"))


class EnsembleTests(unittest.TestCase):
    def test_branch_medoid_caps_correlated_original_decoders(self):
        sources = {
            "original_mlx_large": "pogrešno",
            "original_mlx_turbo": "pogrešno",
            "original_faster_whisper": "pogrešno",
            "slow70_mlx": "tačno",
            "slow50_mlx": "tačno",
            "resolve21_vi50_mlx": "tačno",
        }
        source, text = branch_medoid(sources)
        self.assertIn(source, {"slow70_mlx", "slow50_mlx", "resolve21_vi50_mlx"})
        self.assertEqual(text, "tačno")

    def test_cross_branch_agreement_needs_distinct_families(self):
        originals_only = {
            "original_mlx_large": "dobro sam",
            "original_mlx_turbo": "dobro sam",
            "original_faster_whisper": "dobro sam",
        }
        self.assertFalse(cross_branch_agreement(originals_only))
        with_slowed = {**originals_only, "slow70_mlx": "dobro sam"}
        self.assertTrue(cross_branch_agreement(with_slowed))

    def test_finalizer_rejects_source_flagged_repetition(self):
        windows = [{
            "window_id": 1,
            "start": 0.0,
            "end": 2.0,
            "sources": {
                "original_mlx_large": "Hvala što pratite kanal",
                "slow70_mlx": "Kako si danas",
                "resolve21_vi50_mlx": "Kako si danas",
            },
            "repeated_ngram_flags": {"original_mlx_large": ["hvala što pratite"]},
            "curated_override": None,
        }]
        results = [{
            "window_id": 1,
            "selected_text": "Hvala što pratite kanal",
            "status": "consensus",
            "confidence": 0.8,
            "reason": "bad resolver choice",
        }]
        selected, _ = conservative_finalize(windows, results)
        self.assertEqual(selected[0]["text"], "Kako si danas")
        self.assertEqual(selected[0]["mode"], "medoid_consensus")

    def test_timestamp_rescale(self):
        words = [{"start": 10.0, "end": 12.0, "word": "reč", "prob": 0.9}]
        self.assertEqual(
            rescale_words(words, 0.5),
            [{"start": 5.0, "end": 6.0, "word": "reč", "prob": 0.9}],
        )
        self.assertEqual(words[0]["start"], 10.0)

    def test_exact_curated_override_uses_window_id_only(self):
        corrections = [
            {"window_id": 7, "text": "А где иде?"},
            {"cue": 7, "suggested": "must not match"},
        ]
        self.assertEqual(exact_curated_override(7, corrections), "A gde ide?")
        self.assertIsNone(exact_curated_override(8, corrections))


if __name__ == "__main__":
    unittest.main()
