"""Grading-policy regressions, independent of provider calls."""
import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace
import unittest

from inspect_ai.scorer import Target
from scoring import OUTPUT_FIELDS, FACT_FIELDS, grade, label_fields, legacy_reference, parse_fields, validate_reference, split_species, prepare_reference, normalize


def empty_answer():
    return {field: [] if field in FACT_FIELDS else '' for field in OUTPUT_FIELDS}


def unknown_reference():
    return {field: {'status': 'unknown'} for field in OUTPUT_FIELDS}


class GradingRulesTests(unittest.TestCase):
    def test_fact_matches_ignore_split_or_merged_list_boundaries(self):
        reference = unknown_reference()
        reference['Location'] = {'status': 'present', 'facts': [
            {'value': 'Görlitz'}, {'value': 'Weinlache'}]}
        for prediction in [['Görlitz: Weinlache.'], ['Görlitz Weinlache'], ['Weinlache', 'Görlitz']]:
            values, _ = grade({'Location': prediction}, False, reference)
            self.assertEqual(values['Location'], 1)
            self.assertEqual(values['Location_precision'], 1)
            self.assertEqual(values['Location_recall'], 1)
        reference['Location']['facts'] = [{'value': 'Görlitz: Weinlache'}]
        values, _ = grade({'Location': ['Görlitz', 'Weinlache']}, False, reference)
        self.assertEqual(values['Location'], 1)

    def test_merged_notes_keep_extra_and_duplicate_claims(self):
        reference = unknown_reference()
        reference['Notes'] = {'status': 'present', 'facts': [{'value': 'flowering'}, {'value': 'altitude 200 m'}]}
        values, detail = grade({'Notes': ['flowering; altitude 200 m; flowering; near river']}, False, reference)
        self.assertEqual(values['Notes_recall'], 1)
        self.assertEqual(values['Notes_precision'], .5)
        self.assertEqual(values['unsupported_additions'], 2)
        for prediction in [['not flowering', 'altitude 300 m'], ['flowering not', 'altitude -200 m']]:
            values, _ = grade({'Notes': prediction}, False, reference)
            self.assertEqual(values['Notes_recall'], 0)

    def test_matching_does_not_reuse_overlapping_tokens_and_chooses_best_cover(self):
        reference = unknown_reference()
        reference['Location'] = {'status': 'present', 'facts': [
            {'value': 'New York', 'alternatives': ['NY']}, {'value': 'York'}]}
        values, _ = grade({'Location': ['New York']}, False, reference)
        self.assertEqual(values['Location_recall'], .5)
        values, _ = grade({'Location': ['NY York']}, False, reference)
        self.assertEqual(values['Location'], 1)

    def test_species_parser_handles_authorities_ranks_and_hybrids(self):
        cases = {
            'Dryopteris filix-mas (L.) Schott': ('Dryopteris filix-mas', '(L.) Schott'),
            'Centaurea jacea L. subsp. jacea': ('Centaurea jacea subsp. jacea', 'L.'),
            'Festuca rubra L. ssp. Caespitosa Hack.': ('Festuca rubra ssp. Caespitosa', 'L. Hack.'),
            'Salix × rubens Schrank': ('Salix x rubens', 'Schrank'),
            'Salix repens x purpurea Wim': ('Salix repens x purpurea', 'Wim'),
            'Salix repens x Salix purpurea Wim': ('Salix repens x Salix purpurea', 'Wim'),
            'Viola odorata x hirta': ('Viola odorata x hirta', ''),
            'Andropogon Ischaemum L.': ('Andropogon Ischaemum', 'L.'),
            'Solanum nigrum L. ssp. nigrum var. atriplicifolium (DESP.) G. MEY. f. atriplicifolium':
                ('Solanum nigrum ssp. nigrum var. atriplicifolium f. atriplicifolium', 'L. (DESP.) G. MEY.'),
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(split_species(text), expected)
        for text in ['Ajuga reptans L. var.', 'Unknown', 'Salix alba L. x fragilis L.']:
            self.assertEqual(split_species(text), (None, None))

    def test_rank_and_author_formatting_normalization_is_narrow(self):
        self.assertEqual(normalize('Species name', 'Centaurea jacea ssp. jacea'),
                         normalize('Species name', 'Centaurea jacea subsp. jacea'))
        self.assertEqual(normalize('Species author', '(L.) P. B.'), normalize('Species author', '(L) P.B'))
        self.assertNotEqual(normalize('Species author', 'Wim'), normalize('Species author', 'Wimm'))
        self.assertNotEqual(normalize('Species author', '(L.) Schott'), normalize('Species author', 'L. Schott'))

    def test_saved_catalogue_targets_upgrade_but_reviewed_targets_do_not(self):
        reference = unknown_reference()
        reference['Species name'] = {'status': 'present', 'source': 'catalogue', 'value': 'Dryopteris filix-mas (L.) Schott'}
        reference['Species author'] = {'status': 'unknown', 'source': 'catalogue'}
        upgraded = prepare_reference({'fields': reference})
        self.assertEqual(upgraded['Species name']['value'], 'Dryopteris filix-mas')
        self.assertEqual(upgraded['Species author']['value'], '(L.) Schott')
        self.assertEqual(reference['Species name']['value'], 'Dryopteris filix-mas (L.) Schott')
        reference['Species author'] = {'status': 'absent'}
        self.assertEqual(prepare_reference({'fields': reference}), reference)

    def test_incomplete_catalogue_notes_do_not_label_extra_text_as_unsupported(self):
        reference = unknown_reference()
        reference['Notes'] = {'status': 'present', 'source': 'catalogue', 'facts': [{'value': 'Neu für die Lausitz'}]}
        reference = prepare_reference({'fields': reference})
        values, details = grade({'Notes': ['Neu für die Lausitz', 'Acc. Nr. 93']}, False, reference)
        self.assertEqual(values['Notes_recall'], 1)
        self.assertTrue(math.isnan(values['Notes']))
        self.assertTrue(math.isnan(values['Notes_precision']))
        self.assertEqual(values['unsupported_additions'], 0)
        self.assertEqual(details['Notes']['unverified'], ['acc nr 93'])
        reference['Notes'].update(source='golden', complete=True)
        values, _ = grade({'Notes': ['Neu für die Lausitz', 'Acc. Nr. 93']}, False, reference)
        self.assertEqual(values['Notes'], 0)
        self.assertEqual(values['unsupported_additions'], 1)

    def test_fact_normalization_preserves_relationships_and_numeric_signs(self):
        self.assertNotEqual(normalize('Location', 'north of A, south of B'),
                            normalize('Location', 'south of A, north of B'))
        self.assertNotEqual(normalize('Notes', 'temperature -5'), normalize('Notes', 'temperature 5'))
        self.assertNotEqual(normalize('Notes', 'altitude 20.5 m'), normalize('Notes', 'altitude 205 m'))

    def test_species_identity_is_independent_of_author(self):
        reference = unknown_reference()
        reference['Species name'] = {'status': 'present', 'value': 'Salix repens × purpurea'}
        reference['Species author'] = {'status': 'present', 'value': 'Wim'}
        answer = dict(empty_answer(), **{'Species name': 'salix repens x purpurea', 'Species author': 'Wimm'})
        scores, _ = grade(answer, True, reference)
        self.assertEqual(scores['Species name'], 1)
        self.assertEqual(scores['Species author'], 0)
        answer['Species name'] = 'Salix repens x cinerea'
        self.assertEqual(grade(answer, True, reference)[0]['Species name'], 0)

    def test_partial_dates_get_component_credit_but_not_full_credit(self):
        reference = unknown_reference()
        reference['Collection date'] = {'status': 'present', 'value': '3. Mai 1855'}
        scores, _ = grade({'Collection date': '1855'}, False, reference)
        self.assertEqual([scores[k] for k in ['Collection date', 'date_year', 'date_month', 'date_day']], [0, 1, 0, 0])
        reference['Collection date']['value'] = '1855'
        scores, _ = grade({'Collection date': '1855-05-03'}, False, reference)
        self.assertEqual(scores['Collection date'], 0)
        self.assertTrue(math.isnan(scores['date_month']))
        reference['Collection date']['value'] = 'Mai 1855'
        self.assertEqual(grade({'Collection date': '1855-5'}, False, reference)[0]['Collection date'], 1)

    def test_only_explicit_collector_aliases_are_accepted(self):
        reference = unknown_reference()
        reference["Collector's name"] = {'status': 'present', 'value': 'H. Riese'}
        answer = {"Collector's name": 'Hermann Riese'}
        self.assertEqual(grade(answer, False, reference)[0]["Collector's name"], 0)
        reference["Collector's name"]['alternatives'] = ['Hermann Riese']
        self.assertEqual(grade(answer, False, reference)[0]["Collector's name"], 1)

    def test_absent_unknown_and_unreadable_have_distinct_meanings(self):
        reference = unknown_reference()
        reference['Country'] = {'status': 'absent'}
        reference['Region'] = {'status': 'unreadable'}
        scores, detail = grade(empty_answer(), True, reference)
        self.assertEqual(scores['Country'], 1)
        self.assertTrue(math.isnan(scores['Region']))
        self.assertTrue(math.isnan(scores['specimen_exact']))
        scores, detail = grade({'Country': 'Deutschland', 'Region': 'Lausitz'}, False, reference)
        self.assertEqual(scores['Country'], 0)
        self.assertEqual(scores['unsupported_additions'], 1)
        self.assertEqual(detail['Country']['unsupported'], ['Deutschland'])
        self.assertNotIn('unsupported', detail['Region'])
        self.assertEqual(grade({}, False, reference)[0]['Country'], 0)  # Missing key isn't an empty answer.

    def test_location_components_and_unsupported_additions(self):
        reference = unknown_reference()
        reference['Location'] = {'status': 'present', 'facts': [
            {'value': 'Spremberg'}, {'value': 'Dorf Roitz', 'alternatives': ['Roitz']},
            {'value': 'unter Kiefern'}]}
        scores, detail = grade({'Location': ['ROITZ.', 'Spremberg', 'near river']}, False, reference)
        self.assertEqual(scores['Location'], 0)
        self.assertAlmostEqual(scores['Location_recall'], 2 / 3)
        self.assertAlmostEqual(scores['Location_precision'], 2 / 3)
        self.assertEqual(detail['Location']['omitted'], ['unter Kiefern'])
        self.assertEqual(detail['Location']['unsupported'], ['near river'])
        self.assertEqual(scores['omitted_facts'], 1)
        self.assertEqual(scores['unsupported_additions'], 1)
        answer = {'Location': ['unter Kiefern.', 'Roitz', 'Spremberg']}
        self.assertEqual(grade(answer, False, reference)[0]['Location'], 1)
        answer['Location'].append('Roitz')
        scores, _ = grade(answer, False, reference)
        self.assertEqual(scores['Location'], 0)  # Duplicate facts cannot earn extra credit.
        self.assertEqual(scores['Location_recall'], 1)

    def test_fact_negation_and_numbers_are_not_discarded(self):
        reference = unknown_reference()
        reference['Notes'] = {'status': 'present', 'facts': [{'value': 'not flowering'}, {'value': 'altitude 200 m'}]}
        scores, _ = grade({'Notes': ['flowering', 'altitude 300 m']}, False, reference)
        self.assertEqual(scores['Notes_recall'], 0)
        self.assertEqual(scores['unsupported_additions'], 2)

    def test_invalid_field_does_not_erase_other_credit(self):
        answer = dict(empty_answer(), **{'Species name': 'Salix repens', 'Notes': None, 'extra': 5})
        usable, valid = parse_fields(json.dumps(answer))
        reference = unknown_reference()
        reference['Species name'] = {'status': 'present', 'value': 'Salix repens'}
        reference['Notes'] = {'status': 'absent'}
        scores, _ = grade(usable, valid, reference)
        self.assertEqual(scores['valid_json'], 0)
        self.assertEqual(scores['Species name'], 1)
        self.assertEqual(scores['Notes'], 0)
        duplicate = '{"Country": "A", "Country": "B", "State": "Brandenburg"}'
        usable, valid = parse_fields(duplicate)
        self.assertFalse(valid)
        self.assertNotIn('Country', usable)
        self.assertEqual(usable['State'], 'Brandenburg')
        usable, _ = parse_fields('{"Country":"A", "Notes":{"Country":1,"Country":2}}')
        self.assertEqual(usable['Country'], 'A')

    def test_specimen_exact_requires_complete_reference_but_not_perfect_format(self):
        reference = {field: {'status': 'absent'} for field in OUTPUT_FIELDS}
        answer = empty_answer()
        scores, _ = grade(answer, False, reference)
        self.assertEqual(scores['specimen_exact'], 1)
        self.assertEqual(scores['valid_json'], 0)
        answer['Country'] = 'Deutschland'
        self.assertEqual(grade(answer, True, reference)[0]['specimen_exact'], 0)

    def test_bad_golden_annotations_fail(self):
        for item in [{'status': 'maybe'}, {'status': 'present'},
                     {'status': 'absent', 'value': 'text'},
                     {'status': 'present', 'value': 'text', 'alternatives': 'alias'}]:
            with self.subTest(item=item), self.assertRaises(ValueError):
                validate_reference({**unknown_reference(), 'Region': item})
        overlap = {'status': 'present', 'facts': [{'value': 'Roitz'}, {'value': 'Dorf Roitz', 'alternatives': ['Roitz']}]}
        with self.assertRaisesRegex(ValueError, 'Overlapping'):
            validate_reference({**unknown_reference(), 'Location': overlap})

    def test_scorer_metadata_explains_errors(self):
        reference = unknown_reference()
        reference['Country'] = {'status': 'absent'}
        target = Target(json.dumps({'schema_version': 2, 'fields': reference}))
        state = SimpleNamespace(output=SimpleNamespace(completion=json.dumps({'Country': 'Germany'})))
        result = asyncio.run(label_fields()(state, target))
        self.assertEqual(result.metadata['scoring_version'], 5)
        self.assertEqual(result.metadata['fields']['Country']['unsupported'], ['Germany'])
        self.assertIn('Notes', result.metadata['unscored_fields'])

    def test_documented_golden_example_is_valid(self):
        data = json.loads((Path(__file__).resolve().parents[1] / 'goldens.example.json').read_text())
        self.assertEqual(data['schema_version'], 2)
        for fields in data['samples'].values():
            validate_reference(fields)


class LocationExtrasTests(unittest.TestCase):
    def score_location(self, prediction):
        ref = unknown_reference()
        ref['Location'] = {'status': 'present', 'facts': [{'value': 'Bautzen'}, {'value': 'Johnsdorf'}]}
        return grade({'Location': prediction}, True, ref)[0]['Location']

    def test_benign_extras(self):
        self.assertEqual(self.score_location(['MTB 48 52/33 Bautzen', 'Johnsdorf', 'Feld', 'Geschiebelehm, Sand', '147 m ü. NN']), 1)

    def test_wrong_or_missing_places_still_fail(self):
        for prediction in [['Bautzen', 'Jehnsdorf', 'Sand'], ['Bautzen', 'Sand'], ['Bautzen', 'Johnsdorf', 'Dresden'], ['nicht Bautzen', 'Johnsdorf']]:
            self.assertEqual(self.score_location(prediction), 0)

    def test_context_inside_clause(self):
        ref = unknown_reference()
        ref['Location'] = {'status': 'present', 'facts': [{'value': 'bei einer ehemaligen Glassandgrube'}]}
        self.assertEqual(grade({'Location': ['Ödland bei einer ehemaligen Glassandgrube']}, True, ref)[0]['Location'], 1)

    def test_required_habitat_not_omitted(self):
        ref = unknown_reference()
        ref['Location'] = {'status': 'present', 'facts': [{'value': 'Bautzen'}, {'value': 'Wald'}]}
        self.assertEqual(grade({'Location': ['Bautzen']}, True, ref)[0]['Location'], 0)


if __name__ == '__main__':
    unittest.main()
