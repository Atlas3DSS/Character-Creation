"""Source-span, split, bounded-queue and uncertainty tests; no model calls."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from persona_characters import ErrorBudget, Index, MODEL_SHA, QUOTE, SCHEMA, checked_span, original_chunks, pilot_order, planner_receipt, request, validate_batch, validate_response
from persona_corpus import record
from persona_manager import file_hash
from persona_vectors import canonical_digest


class CharacterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.text = 'Skippy said, “Hello.” Joe Bishop, also called Joe, replied, “Hi.”'
        self.row = {'id': 'chunk-a', 'book_sha256': 'b' * 64, 'split': 'discovery', 'text': self.text,
            'source': {'char_start': 100, 'member': 'chapter.xhtml'}, 'source_text_sha256': 's',
            'pilot_category': 'attributed', 'quotes': [{'id': i, 'start': m.start(), 'end': m.end(), 'text': m.group()}
                                                    for i, m in enumerate(QUOTE.finditer(self.text))]}
        def span(text):
            return {'text': text}
        self.span = span
        self.item = {'chunk_id': 'chunk-a', 'characters': [
            {'name': 'Skippy', 'evidence': span('Skippy said'), 'certainty': 'explicit'},
            {'name': 'Joe Bishop', 'evidence': span('Joe Bishop, also called Joe'), 'certainty': 'explicit'},
            {'name': 'Joe', 'evidence': span('called Joe'), 'certainty': 'explicit'}],
            'aliases': [{'character': 1, 'alias': 'Joe', 'evidence': span('Joe Bishop, also called Joe'), 'certainty': 'explicit'}]}
        self.response = {'schema': SCHEMA, 'corpus_sha256': 'c' * 64, 'chunks': [self.item]}

    def validate(self, response):
        return validate_response(response, [self.row], 'c' * 64)

    def test_exact_character_mentions_aliases_and_speech_remain_provisional(self):
        result = self.validate(self.response)['chunk-a']
        self.assertEqual(result['characters'][0]['document_start'], 100)
        self.assertEqual(result['characters'][0]['document_end'], 106)
        self.assertEqual(result['aliases'][0]['evidence']['text'], 'Joe Bishop, also called Joe')
        self.assertEqual(result['alias_policy'], 'candidate edges only; no automatic identity merge')
        self.assertIn('not exhaustive', result['coverage_claim'])

    def test_fabricated_or_shifted_evidence_is_rejected(self):
        for field, value in [('text', 'Skippy the Great'), ('start', 1), ('occurrence', -1), ('occurrence', True)]:
            changed = deepcopy(self.response)
            changed['chunks'][0]['characters'][0]['evidence'][field] = value
            with self.assertRaises(ValueError): self.validate(changed)
        changed = deepcopy(self.response)
        changed['chunks'][0]['aliases'][0]['evidence'] = self.span('Skippy')
        with self.assertRaises(ValueError): self.validate(changed)

    def test_unknown_speaker_is_explicitly_unresolved_without_invention(self):
        speeches = self.validate(self.response)['chunk-a']['speeches']
        self.assertEqual([speech['quote_id'] for speech in speeches], [0, 1])
        self.assertTrue(all(speech['speaker_mention'] is None and speech['certainty'] == 'uncertain'
                            and speech['unresolved_reason'] == 'not_parsed_by_character_inventory' for speech in speeches))
        changed = deepcopy(self.response)
        changed['chunks'][0]['speeches'] = [{'speaker_mention': 0}]
        with self.assertRaises(ValueError): self.validate(changed)

    def test_schema_bindings_and_quote_duplicates_fail_closed(self):
        for key, bad in [('schema', 'other'), ('corpus_sha256', 'other'), ('chunks', [])]:
            changed = deepcopy(self.response); changed[key] = bad
            with self.assertRaises(ValueError): self.validate(changed)
        changed = deepcopy(self.response); changed['shell'] = 'anything'
        with self.assertRaises(ValueError): self.validate(changed)
        changed = deepcopy(self.response); changed['chunks'][0]['characters'].append(self.item['characters'][0])
        with self.assertRaises(ValueError): self.validate(changed)

    def test_source_admission_excludes_mutations_and_heldout_and_checks_sha(self):
        source = {'path': '/private/book.epub', 'sha256': 'b' * 64, 'bytes': 100}
        locator = {'member': 'chapter.xhtml', 'char_start': 0, 'char_end': len(self.text)}
        kwargs = dict(corpus='skippy', category='book_character_dialogue', split='discovery', locator=locator,
                      prompt='Read this.\n\nPassage:\n' + self.text)
        original = record(source, self.text, **kwargs)
        mutation = record(source, self.text, **kwargs, mutation={'kind': 'instruction_negation'})
        heldout = record(source, self.text, **{**kwargs, 'split': 'heldout', 'locator': {**locator, 'member': 'heldout.xhtml'}})
        corpus = self.root / 'corpus.jsonl'; manifest = self.root / 'manifest.json'
        corpus.write_text('\n'.join(json.dumps(row) for row in (original, mutation, heldout)) + '\n')
        manifest.write_text(json.dumps({'status': 'complete', 'output_sha256': file_hash(corpus)}))
        digest, rows = original_chunks(corpus, manifest, 'discovery')
        self.assertEqual(len(rows), 1); self.assertEqual(rows[0]['text'], self.text)
        self.assertEqual(len(original_chunks(corpus, manifest, 'all')[1]), 2)
        original['prompt'] += 'Tampering'
        original['id'] = canonical_digest({key: value for key, value in original.items() if key != 'id'})
        corpus.write_text(json.dumps(original) + '\n')
        manifest.write_text(json.dumps({'status': 'complete', 'output_sha256': file_hash(corpus)}))
        with self.assertRaisesRegex(ValueError, 'source text'): original_chunks(corpus, manifest, 'discovery')

    def test_pilot_order_is_deterministic_and_spans_books_before_repeat(self):
        rows = [{**self.row, 'id': f'{book}-{i}', 'book_sha256': book,
                 'pilot_category': ('attributed', 'dialogue', 'narrative')[i % 3]}
                for book in ('a', 'b', 'c') for i in range(12)]
        ordered = pilot_order(rows)
        self.assertEqual(ordered, pilot_order(list(reversed(rows))))
        self.assertEqual(len({row['book_sha256'] for row in ordered[:3]}), 3)
        self.assertEqual(len(ordered), len(rows))
        self.assertEqual(len({row['id'] for row in ordered}), len(rows))

    def test_index_resume_and_corpus_binding_preserve_attempts(self):
        index = Index(self.root / 'index.sqlite')
        identity = {'schema': SCHEMA, 'corpus_sha256': 'c'}
        index.bind(identity, [self.row]); self.assertEqual(len(index.claim(4)), 1)
        index.recover(); self.assertEqual(index.status()['counts'], {'queued': 1})
        self.assertEqual(len(index.claim(4)), 1)
        index.recover(); self.assertEqual(index.status()['counts'], {'failed': 1})
        with self.assertRaises(ValueError): index.bind({**identity, 'corpus_sha256': 'd'}, [self.row])

    def test_discovery_index_can_expand_to_all_once_without_changing_prior_evidence(self):
        index = Index(self.root / 'index.sqlite')
        identity = {'schema': SCHEMA, 'corpus_sha256': 'c' * 64, 'split': 'discovery',
                    'planner': {'runtime_sha256': 'r' * 64}}
        heldout = {**self.row, 'id': 'heldout-chunk', 'split': 'heldout'}
        index.bind(identity, [self.row])
        claimed = index.claim(1)
        with index.connect() as db:
            attempt = db.execute('INSERT INTO attempts(started,request_sha256,chunks,script_sha256) VALUES(?,?,?,?)',
                                 (1, 'r', json.dumps([self.row['id']]), 's')).lastrowid
        index.finish(claimed, attempt, self.response, self.validate(self.response), {})
        with index.connect() as db:
            before_chunk = db.execute("SELECT * FROM chunks WHERE id='chunk-a'").fetchone()
            before_attempt = db.execute('SELECT * FROM attempts WHERE id=?', (attempt,)).fetchone()
        all_identity = {**identity, 'split': 'all'}
        for rejected in ({**all_identity, 'corpus_sha256': 'changed'},
                         {**all_identity, 'planner': {'runtime_sha256': 'changed'}}):
            with self.assertRaises(ValueError): index.bind(rejected, [heldout, self.row])
            self.assertEqual(index.status()['total_source_chunks'], 1)
        index.bind(all_identity, [heldout, self.row])
        index.bind(all_identity, [heldout, self.row])
        self.assertEqual(index.status()['counts'], {'complete': 1, 'queued': 1})
        with index.connect() as db:
            self.assertEqual(db.execute("SELECT * FROM chunks WHERE id='chunk-a'").fetchone(), before_chunk)
            self.assertEqual(db.execute('SELECT * FROM attempts WHERE id=?', (attempt,)).fetchone(), before_attempt)
            self.assertEqual(json.loads(db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()[0]), all_identity)
            self.assertEqual(db.execute("SELECT split,attempts,priority FROM chunks WHERE id='heldout-chunk'").fetchone(), ('heldout', 0, 1))
            audits = db.execute("SELECT data FROM audit WHERE event='scope_expanded'").fetchall()
            self.assertEqual(len(audits), 1)
            self.assertEqual(json.loads(audits[0][0]), {'old_split': 'discovery', 'new_split': 'all',
                'added_chunks': 1, 'corpus_sha256': 'c' * 64, 'annotation_only': True})
        for rejected in (identity, {**all_identity, 'split': 'heldout'}, {**all_identity, 'schema': 'changed'}):
            with self.assertRaises(ValueError): index.bind(rejected, [self.row])
        self.assertEqual([row['id'] for row in index.claim(4)], ['heldout-chunk'])

    def test_request_is_bounded_json_without_tools_or_arbitrary_paths(self):
        payload = request([self.row], 'c' * 64, 'persona-manager')
        self.assertEqual(payload['max_tokens'], 2048)
        self.assertEqual(payload['response_format'], {'type': 'json_object'})
        self.assertNotIn('tools', payload)
        brief = json.loads(payload['messages'][1]['content'])
        self.assertNotIn('source', brief['chunks'][0])
        self.assertNotIn('word_ledger', brief['chunks'][0])
        self.assertNotIn('quotes', brief['chunks'][0])

    def test_software_locates_unicode_exact_snippets_and_bounds_repetition(self):
        text = 'Éva said hi. Éva then saw Joe Bishop.'
        with self.assertRaisesRegex(ValueError, 'occurrence index'):
            checked_span({'text': 'Éva'}, text)
        located = checked_span({'text': 'Éva', 'occurrence': 1}, text)
        self.assertEqual(located['start'], text.rindex('Éva'))
        self.assertEqual(text[located['start']:located['end']], 'Éva')
        self.assertEqual(checked_span({'text': 'Joe Bishop'}, text)['start'], text.index('Joe Bishop'))
        for snippet in ({'text': 'Éva', 'occurrence': 2}, {'text': 'Éva', 'occurrence': True},
                        {'text': 'Eva'}, {'text': 'Éva', 'start': 0, 'end': 3}, {'text': ''}):
            with self.assertRaises(ValueError): checked_span(snippet, text)

    def test_planner_receipt_requires_active_verified_model_and_runtime(self):
        path = self.root / 'planner.jsonl'
        rows = [{'event': 'submitted', 'lease_id': 'owned', 'model_sha256': MODEL_SHA},
                {'event': 'started', 'lease_id': 'owned', 'runtime_sha256': 'a' * 64},
                {'event': 'ready', 'lease_id': 'owned', 'model_alias': 'persona-manager', 'context_tokens': 131072}]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.assertEqual(planner_receipt(path, 'owned', 'persona-manager')['context_tokens'], 131072)
        with self.assertRaises(ValueError): planner_receipt(path, 'other-owner', 'persona-manager')
        path.write_text(path.read_text() + json.dumps({'event': 'finished', 'lease_id': 'owned'}) + '\n')
        with self.assertRaises(ValueError): planner_receipt(path, 'owned', 'persona-manager')

    def test_retry_feedback_is_scoped_to_the_same_source_chunks(self):
        index = Index(self.root / 'index.sqlite')
        with index.connect() as db:
            db.execute('INSERT INTO attempts(started,request_sha256,chunks,response,error,script_sha256) VALUES(?,?,?,?,?,?)',
                       (1, 'r', json.dumps(['chunk-a']), '{}', 'duplicated quote', 's'))
        self.assertEqual(index.feedback([self.row]), ['duplicated quote'])
        self.assertEqual(index.feedback([{**self.row, 'id': 'other'}]), [])
        payload = request([self.row], 'c' * 64, 'persona-manager', index.feedback([self.row]))
        self.assertEqual(json.loads(payload['messages'][1]['content'])['previous_validation_errors'], ['duplicated quote'])

    def test_consecutive_errors_reset_after_valid_batch_but_total_is_retained(self):
        budget = ErrorBudget(total_limit=10000, consecutive_limit=3)
        for error in ('invalid source span', 'invalid source span', None, 'incomplete JSON', None):
            budget.record(error)
            self.assertIsNone(budget.stop_reason())
        self.assertEqual(budget.status()['total'], 3)
        self.assertEqual(budget.status()['consecutive'], 0)
        for _ in range(2):
            budget.record('systemic endpoint failure')
            self.assertIsNone(budget.stop_reason())
        budget.record('systemic endpoint failure')
        self.assertEqual(budget.stop_reason(), 'consecutive_error_budget')
        self.assertEqual(budget.status()['total'], 6)

    def test_total_error_budget_stops_isolated_failures_and_bounds_configuration(self):
        budget = ErrorBudget(total_limit=3, consecutive_limit=5)
        for _ in range(2):
            budget.record('rejected evidence'); budget.record(None)
            self.assertIsNone(budget.stop_reason())
        budget.record('rejected evidence')
        self.assertEqual(budget.stop_reason(), 'total_error_budget')
        self.assertEqual(budget.status()['consecutive'], 1)
        self.assertEqual(ErrorBudget(100000, 100000).status()['total_limit'], 100000)
        for total, consecutive in ((0, 5), (3, 0), (100001, 5), (3, 100001), (True, 5)):
            with self.assertRaises(ValueError): ErrorBudget(total, consecutive)

    def test_mixed_batch_saves_two_valid_chunks_and_retries_only_rejected_chunk(self):
        rows = [{**self.row, 'id': identity} for identity in ('chunk-a', 'chunk-b', 'chunk-c')]
        response = deepcopy(self.response)
        response['chunks'] = [{**deepcopy(self.item), 'chunk_id': row['id']} for row in rows]
        # The first two annotations in c are valid; its last invalid annotation
        # must prevent saving any records from c, without discarding a or b.
        response['chunks'][2]['characters'][2]['evidence'] = {'text': 'fabricated source evidence'}
        results, errors = validate_batch(response, rows, 'c' * 64)
        self.assertEqual(set(results), {'chunk-a', 'chunk-b'})
        self.assertEqual(set(errors), {'chunk-c'})
        index = Index(self.root / 'index.sqlite')
        index.bind({'corpus_sha256': 'c' * 64}, rows)

        def attempt(claimed):
            with index.connect() as db:
                return db.execute('INSERT INTO attempts(started,request_sha256,chunks,script_sha256) VALUES(?,?,?,?)',
                                  (1, 'r', json.dumps([row['id'] for row in claimed]), 's')).lastrowid

        claimed = index.claim(4)
        first = attempt(claimed)
        index.finish(claimed, first, response, results, errors)
        self.assertEqual(index.status()['counts'], {'complete': 2, 'queued': 1})
        self.assertEqual(index.status()['bad_chunk_attempts'], 1)
        with index.connect() as db:
            stored = db.execute('SELECT response,chunk_errors FROM attempts WHERE id=?', (first,)).fetchone()
            self.assertEqual(json.loads(stored[0]), response)
            self.assertEqual(json.loads(stored[1]), errors)
            self.assertIsNone(db.execute("SELECT result FROM chunks WHERE id='chunk-c'").fetchone()[0])
        self.assertEqual(index.feedback([rows[0]]), [])
        self.assertEqual(index.feedback([rows[2]]), [errors['chunk-c']])
        retry = index.claim(4)
        self.assertEqual([row['id'] for row in retry], ['chunk-c'])
        index.finish(retry, attempt(retry), response, {}, errors)
        self.assertEqual(index.status()['counts'], {'complete': 2, 'failed': 1})
        self.assertEqual(index.status()['bad_chunk_attempts'], 2)
        self.assertEqual(index.claim(4), [])

    def test_batch_global_schema_and_exact_chunk_ids_fail_closed(self):
        valid = deepcopy(self.response)
        invalids = []
        for key, bad in [('schema', 'other'), ('corpus_sha256', 'other'), ('chunks', [])]:
            value = deepcopy(valid); value[key] = bad; invalids.append(value)
        value = deepcopy(valid); value['shell'] = 'untrusted command'; invalids.append(value)
        value = deepcopy(valid); value['chunks'][0]['chunk_id'] = 'unknown'; invalids.append(value)
        value = deepcopy(valid); value['chunks'].append(deepcopy(value['chunks'][0])); invalids.append(value)
        value = deepcopy(valid); value['chunks'][0]['characters'][0]['evidence']['start'] = 0; invalids.append(value)
        value = deepcopy(valid); value['chunks'][0]['characters'][0]['command'] = 'untrusted'; invalids.append(value)
        for value in invalids:
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_batch(value, [self.row], 'c' * 64)
        invalid = deepcopy(valid)
        invalid['chunks'][0]['characters'][0]['evidence']['text'] = 'fabricated'
        results, errors = validate_batch(invalid, [self.row], 'c' * 64)
        self.assertEqual(results, {})
        self.assertEqual(set(errors), {'chunk-a'})

    def test_mixed_batch_resets_systemic_streak_and_audits_bad_chunks_separately(self):
        budget = ErrorBudget(10000, 3)
        budget.record_batch(valid_chunks=0, bad_chunks=4)
        budget.record_batch(valid_chunks=0, bad_chunks=4)
        budget.record_batch(valid_chunks=2, bad_chunks=1)
        self.assertEqual(budget.status()['consecutive'], 0)
        self.assertEqual(budget.status()['total'], 3)
        self.assertEqual(budget.status()['bad_chunk_attempts'], 9)
        for _ in range(3):
            budget.record_batch(valid_chunks=0, bad_chunks=4)
        self.assertEqual(budget.stop_reason(), 'consecutive_error_budget')


if __name__ == '__main__':
    unittest.main()
