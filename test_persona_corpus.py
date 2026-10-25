"""Local synthetic-fixture tests; no external book access or model inference."""
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import zipfile

from persona_corpus import (big_five_records, book_candidates, book_documents, book_records, chunks, dialogue_labels,
                           ingest_books, jsonl_records, prepare, sha256, split_for, swap_speakers)


ROOT = Path(__file__).parent


class PersonaCorpusTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        (self.root / "data").mkdir()
        self.books = self.root / "books"
        self.books.mkdir()

    def epub(self, path):
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("META-INF/container.xml", '<container><rootfiles><rootfile full-path="OPS/book.opf"/></rootfiles></container>')
            archive.writestr("OPS/book.opf", '<package><manifest><item id="two" href="z.xhtml" media-type="application/xhtml+xml"/><item id="one" href="a.xhtml" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="two"/><itemref idref="one"/></spine></package>')
            archive.writestr("OPS/z.xhtml", '<html><style>HIDDEN</style><p>Skippy: A test line.</p><p>Joe: Another test line.</p><script>NEVER</script></html>')
            archive.writestr("OPS/a.xhtml", '<p>Second document &amp; text.</p>')

    def test_epub_spine_order_plain_text_and_member_relative_offsets(self):
        path = self.books / "test.epub"
        self.epub(path)
        docs = list(book_documents(path))
        self.assertEqual([name for name, _ in docs], ["OPS/z.xhtml", "OPS/a.xhtml"])
        self.assertEqual(docs[0][1], "Skippy: A test line.\nJoe: Another test line.")
        self.assertNotIn("HIDDEN", docs[0][1])
        rows = list(book_records(path))
        self.assertEqual(len(rows), 5)  # original, instruction, speaker swap; then original/instruction.
        for row in rows:
            text = dict(docs)[row["source"]["member"]]
            self.assertLessEqual(row["source"]["char_end"], len(text))
        self.assertEqual(len({row["split"] for row in rows if row["source"]["member"] == "OPS/z.xhtml"}), 1)

    def test_epub_size_bound_rejects_before_unbounded_expansion(self):
        path = self.books / "test.epub"
        self.epub(path)
        with self.assertRaises(ValueError):
            list(book_documents(path, max_document_bytes=10))

    def test_epub_percent_encoded_spine_href_resolves_archive_member(self):
        path = self.books / "encoded.epub"
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("META-INF/container.xml", '<container><rootfile full-path="OPS/book.opf"/></container>')
            archive.writestr("OPS/book.opf", '<package><manifest><item id="one" href="Text/Chapter%201.html" media-type="application/xhtml+xml"/></manifest><spine><itemref idref="one"/></spine></package>')
            archive.writestr("OPS/Text/Chapter 1.html", '<p>Synthetic chapter.</p>')
        self.assertEqual(list(book_documents(path)), [("OPS/Text/Chapter 1.html", "Synthetic chapter.")])

    def test_nonoverlapping_chunks_cover_the_original_characters(self):
        text = "a word sequence " * 80
        pieces = list(chunks(text, 128))
        self.assertEqual("".join(piece for _, _, piece in pieces), text)
        self.assertEqual(pieces[0][0], 0)
        self.assertEqual(pieces[-1][1], len(text))
        self.assertTrue(all(a[1] == b[0] for a, b in zip(pieces, pieces[1:])))

    def test_character_mentions_do_not_become_invented_speakers(self):
        text = 'Skippy was present. "Hello," he said.'
        labels = dialogue_labels(text)
        self.assertEqual(labels["character_mentions"], {"Skippy": 1})
        self.assertEqual(labels["dialogue_spans"], [])
        tagged = dialogue_labels('"Hello," Skippy said. "Why?" asked Joe.')
        self.assertEqual({span["speaker"] for span in tagged["dialogue_spans"]}, {"Skippy", "Joe"})

    def test_speaker_swap_changes_attribution_not_dialogue_or_narration(self):
        text = "Skippy: First words.\nJoe: Other words.\nA narrator mentions Skippy."
        changed, metadata = swap_speakers(text, dialogue_labels(text)["dialogue_spans"])
        self.assertEqual(changed, "Joe: First words.\nSkippy: Other words.\nA narrator mentions Skippy.")
        self.assertEqual(metadata["kind"], "speaker_attribution_swap")
        aliases = "Joe: One.\nBishop: Two."
        self.assertIsNone(swap_speakers(aliases, dialogue_labels(aliases)["dialogue_spans"]))

    def test_existing_splits_and_instruction_pairs_remain_bound(self):
        path = self.root / "prompts.jsonl"
        path.write_text(json.dumps({"prompt": "Example task", "split": "test", "category": "test"}) + "\n")
        rows = list(jsonl_records(path, "contrastive"))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["split"] for row in rows}, {"heldout"})
        self.assertEqual(len({row["pair_id"] for row in rows}), 1)
        self.assertEqual(rows[-1]["mutation"]["kind"], "instruction_negation")
        self.assertIn("Do not answer", rows[-1]["prompt"])
        self.assertFalse(rows[-1]["grade_expected_answer"])
        self.assertEqual(split_for("content"), split_for("content"))
        with self.assertRaises(ValueError):
            split_for("content", "unknown")

    def test_counterfactual_mutation_changes_only_explicit_named_speaker(self):
        text = '"Hello," Skippy said. A narrator mentions Skippy.'
        spans = dialogue_labels(text)["dialogue_spans"]
        self.assertIsNone(swap_speakers(text, spans))
        changed, mutation = swap_speakers(text, spans, counterfactual=True)
        self.assertEqual(changed, '"Hello," Joe Bishop said. A narrator mentions Skippy.')
        self.assertTrue(mutation["counterfactual"])
        self.assertEqual(mutation["kind"], "speaker_attribution_counterfactual")
        self.assertIsNone(swap_speakers("Skippy was nearby.", [], counterfactual=True))

    def test_ipip_item_keys_are_preserved_and_inventory_excluded_from_fitting(self):
        rows = list(big_five_records(ROOT / "data/big_five_ocean_test.json", ROOT / "data/persona_prompts.json"))
        items = json.loads((ROOT / "data/big_five_ocean_test.json").read_text())["questions"]
        inventory = [row for row in rows if row["purpose"] == "inventory_measurement"]
        self.assertEqual(len(rows), 314)
        self.assertEqual([row["labels"]["inventory_item"] for row in inventory], items)
        self.assertTrue(all(row["split"] == "heldout" and not row["adaptive_fit_allowed"] for row in inventory))
        self.assertEqual(sum(row["split"] == "discovery" for row in rows), 132)

    def prepare_small(self):
        (self.root / "data/scotus").mkdir()
        for name in ("big_five_ocean_test.json", "persona_prompts.json"):
            shutil.copyfile(ROOT / "data" / name, self.root / "data" / name)
        (self.root / "data/scotus/scotus_poke_prompts_v1.jsonl").write_text(json.dumps({"prompt": "Explain this test scenario.", "issue_area": "test"}) + "\n")
        output, manifest = self.root / "data/corpus.jsonl", self.root / "data/manifest.json"
        prepare(self.root, output, manifest)
        return output, manifest

    def test_hot_ingestion_appends_each_book_hash_once_and_preserves_existing_bytes(self):
        output, manifest = self.prepare_small()
        original = output.read_bytes()
        self.assertEqual(ingest_books(output, manifest, stable_seconds=0)["books_status"], "pending_redownload")
        book = self.books / "local.txt"
        book.write_text("Skippy: First synthetic line.\nJoe: Second synthetic line.")
        result = ingest_books(output, manifest, stable_seconds=0)
        self.assertEqual(result["appended_books"], 1)
        self.assertEqual(result["appended_records"], 3)
        self.assertTrue(output.read_bytes().startswith(original))
        identity = sha256(output)
        shutil.copyfile(book, self.books / "same-content.txt")
        self.assertEqual(ingest_books(output, manifest, stable_seconds=0)["appended_records"], 0)
        self.assertEqual(sha256(output), identity)
        metadata = json.loads(manifest.read_text())
        self.assertEqual(metadata["status"], "complete")
        self.assertEqual(metadata["output_sha256"], identity)
        self.assertEqual(metadata["generation"], 2)
        self.assertNotIn("First synthetic line", manifest.read_text())

    def test_invalid_book_is_pending_without_modifying_the_corpus(self):
        output, manifest = self.prepare_small()
        identity = sha256(output)
        (self.books / "partial.epub").write_text("not a zip file")
        result = ingest_books(output, manifest, stable_seconds=0)
        self.assertEqual(result["appended_records"], 0)
        self.assertEqual(len(result["pending_books"]), 1)
        self.assertEqual(sha256(output), identity)

    def test_downloads_scan_accepts_only_known_completed_books_and_one_child(self):
        downloads = self.root / "Downloads"
        downloads.mkdir()
        for name in ("Columbus Day.txt", "Valkyrie.torrent", "unrelated.txt", "model.zip"):
            (downloads / name).write_text("fixture")
        known = downloads / "Craig Alanson - Mavericks"
        known.mkdir()
        (known / "book.epub").write_text("fixture")
        nested = known / "nested"
        nested.mkdir()
        (nested / "book.epub").write_text("fixture")
        other = downloads / "unrelated"
        other.mkdir()
        (other / "book.epub").write_text("fixture")
        self.assertEqual(book_candidates(downloads), [downloads / "Columbus Day.txt", known / "book.epub"])


if __name__ == "__main__":
    unittest.main()
