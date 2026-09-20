"""Offline dataset, scoring, provider configuration, and Inspect integration tests."""
import asyncio
import base64
import csv
import io
import json
import math
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from contextlib import chdir
from unittest.mock import AsyncMock, patch

from PIL import Image
from inspect_ai import eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import ModelOutput, ModelUsage, get_model
from inspect_ai.scorer import Target

from herbarium import herbarium, image_input, load_dataset
from scoring import FIELDS, OUTPUT_FIELDS, FACT_FIELDS, label_fields, normalize, parse_answer, legacy_reference


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.folder = self.root / 'handwritten'
        self.folder.mkdir()
        self.listing = self.root / 'handwritten.txt'
        self.names = ['secret_taxon.jpg', 'second.png']
        self.listing.write_text('\n'.join(self.names), encoding='utf-8')
        for name in self.names:
            Image.new('RGB', (80, 40)).save(self.folder / name)
        self.reference = dict(zip(FIELDS, ['Salix repens x purpurea Wim', '3. Mai 1855',
                                          'Hermann Riese', 'Deutschland: Brandenburg',
                                          'Spremberg / Dorf Roitz', '', '']))
        self.rows = [dict(Bildname=name, **{column: self.reference[field] for field, column in FIELDS.items()})
                     for name in reversed(self.names)]
        self.write_csv()

    def write_csv(self):
        with (self.folder / 'label_data.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Bildname', *FIELDS.values()])
            writer.writeheader()
            writer.writerows(self.rows)

    def dataset(self):
        return load_dataset(str(self.root), str(self.listing), 32)

    def test_join_order_and_no_answer_or_filename_in_prompt(self):
        dataset = self.dataset()
        self.assertEqual([sample.id for sample in dataset], self.names)
        sample = dataset[0]
        self.assertEqual(json.loads(sample.target)["fields"], legacy_reference(self.rows[0]))
        message = sample.input[0]
        self.assertEqual(len(message.content), 2)
        self.assertNotIn('secret_taxon', message.text)
        self.assertNotIn('Hermann Riese', message.text)
        image_url = message.content[1].image
        with Image.open(io.BytesIO(base64.b64decode(image_url.split(',')[1]))) as img:
            self.assertEqual(img.size, (32, 16))
            self.assertFalse(img.getexif())

    def test_rgba_and_no_upscale(self):
        path = self.folder / 'alpha.png'
        Image.new('RGBA', (10, 5)).save(path)
        url = image_input(path, 32)
        with Image.open(io.BytesIO(base64.b64decode(url.split(',')[1]))) as img:
            self.assertEqual(img.size, (10, 5))
            self.assertEqual(img.mode, 'RGB')

    def test_golden_overrides_are_embedded_without_changing_the_prompt(self):
        golden = self.root / 'goldens.json'
        golden.write_text(json.dumps({'schema_version': 2, 'samples': {
            self.names[0]: {'Country': {'status': 'absent'},
                            "Collector's name": {'status': 'present', 'value': 'Riese'}}}}))
        original = self.dataset()
        dataset = load_dataset(str(self.root), str(self.listing), 32, str(golden))
        fields = json.loads(dataset[0].target)['fields']
        self.assertEqual(fields['Country'], {'status': 'absent'})
        self.assertEqual(fields["Collector's name"]['value'], 'Riese')
        self.assertEqual(fields['State']['source'], 'catalogue')
        self.assertEqual(dataset[0].input[0].content, original[0].input[0].content)
        self.assertEqual(dataset[1].target, original[1].target)
        self.assertEqual(dataset[0].metadata['golden_fields'], ['Country', "Collector's name"])

    def test_bad_golden_fails_before_image_processing(self):
        golden = self.root / 'goldens.json'
        for samples in [{self.names[0]: {'Country': {'status': 'absent', 'value': 'Germany'}}},
                        {'misspelled-filename.jpg': {}},
                        {self.names[0]: {'County': {'status': 'absent'}}}]:
            golden.write_text(json.dumps({'schema_version': 2, 'samples': samples}))
            with patch('herbarium.image_input', side_effect=AssertionError('Should validate first')), \
                 self.assertRaises(ValueError):
                load_dataset(str(self.root), str(self.listing), 32, str(golden))

    def test_invalid_dataset_fails_before_generation(self):
        self.listing.write_text('absent.jpg')
        with self.assertRaisesRegex(ValueError, 'No ground-truth'):
            self.dataset()
        self.listing.write_text(self.names[0] + '\n' + self.names[0])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.dataset()
        self.listing.write_text(self.names[0])
        (self.folder / self.names[0]).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing image'):
            self.dataset()

    def test_duplicate_truth_and_missing_column(self):
        self.rows.append(self.rows[0])
        self.write_csv()
        with self.assertRaisesRegex(ValueError, 'Duplicate ground-truth'):
            self.dataset()
        (self.folder / 'label_data.csv').write_text('Bildname\nsecret_taxon.jpg\n')
        with self.assertRaisesRegex(ValueError, 'missing columns'):
            self.dataset()

    def test_empty_list_and_bad_size(self):
        self.listing.write_text('')
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.dataset()
        with self.assertRaisesRegex(ValueError, 'positive'):
            load_dataset(str(self.root), str(self.listing), 0)

    def test_inspect_eval_logs_scores_and_usage_without_network(self):
        correct = {field: ([fact['value'] for fact in item.get('facts', [])] if field in FACT_FIELDS
                            else item.get('value', ''))
                   for field, item in legacy_reference(self.rows[0]).items()}
        correct['Collection date'] = '1855-05-03'
        wrong = dict(correct, **{'Collection date': '1865-09-26', "Collector's name": 'Riese'})
        outputs = [ModelOutput.from_content('mockllm/model', json.dumps(row)) for row in [correct, wrong]]
        for output in outputs:
            output.usage = ModelUsage(input_tokens=3341, output_tokens=586, total_tokens=3927)
        with patch.dict(os.environ, {'INSPECT_TRACE_FILE': str(self.root / 'trace.log')}), \
             patch('inspect_ai._util.appdirs.user_data_path', return_value=self.root / 'inspect-data'), \
             patch('inspect_ai._util.appdirs.user_cache_path', return_value=self.root / 'inspect-cache'), \
             patch.object(socket.socket, 'connect', side_effect=AssertionError('Network disabled in tests')):
            logs = eval(
                herbarium(str(self.root), str(self.listing), 32),
                model=get_model('mockllm/model', custom_outputs=outputs),
                max_connections=1, max_samples=1, display='none', ctl_server=False,
                log_dir=str(self.root / 'logs'),
            )
        log = read_eval_log(logs[0].location)
        self.assertEqual(log.status, 'success', str(log.error))
        self.assertEqual(len(log.samples), 2)
        metrics = {score.name: score for score in log.results.scores}
        self.assertEqual(metrics['Collection date'].metrics['mean'].value, .5)
        self.assertEqual(metrics["Collector's name"].metrics['mean'].value, .5)
        self.assertEqual(metrics['Region'].scored_samples, 0)
        self.assertEqual(metrics['Region'].unscored_samples, 2)
        self.assertEqual(metrics['valid_json'].metrics['mean'].value, 1)
        usage = next(iter(log.stats.model_usage.values()))
        self.assertEqual(usage.input_tokens, 6682)
        self.assertEqual(usage.output_tokens, 1172)
        self.assertEqual(len(log.samples[0].messages), 2)

    def test_openrouter_returned_costs_survive_eval_log(self):
        import httpx2
        charges = iter([0.0123, 0.0])

        def respond(request):
            return httpx2.Response(200, json={
                'id': 'chat_test', 'object': 'chat.completion', 'created': 0,
                'model': 'google/gemini-2.5-pro',
                'choices': [{'index': 0, 'finish_reason': 'stop',
                             'message': {'role': 'assistant', 'content': json.dumps(self.reference)}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 20,
                          'total_tokens': 120, 'cost': 0, 'is_byok': True,
                          'cost_details': {'upstream_inference_cost': next(charges)}},
            })

        client = httpx2.AsyncClient(transport=httpx2.MockTransport(respond))
        try:
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'router-test'}), \
                 patch('inspect_ai._util.appdirs.user_data_path', return_value=self.root / 'inspect-data'), \
                 patch('inspect_ai._util.appdirs.user_cache_path', return_value=self.root / 'inspect-cache'), \
                 patch.object(socket.socket, 'connect', side_effect=AssertionError('Network disabled')):
                logs = eval(
                    herbarium(str(self.root), str(self.listing), 32),
                    model=get_model('openrouter-cost/google/gemini-2.5-pro', http_client=client, memoize=False),
                    max_connections=1, max_samples=1, display='none', ctl_server=False,
                    log_dir=str(self.root / 'cost-logs'),
                )
            log = read_eval_log(logs[0].location)
            self.assertEqual(log.status, 'success', str(log.error))
            self.assertEqual(next(iter(log.stats.model_usage.values())).total_cost, 0.0123)
            self.assertCountEqual([sample.output.usage.total_cost for sample in log.samples], [0.0123, 0.0])
            for sample in log.samples:
                billing = sample.output.metadata['openrouter_billing']
                self.assertEqual(billing['openrouter_cost'], 0)
                self.assertEqual(billing['upstream_inference_cost'], sample.output.usage.total_cost)
            self.assertCountEqual([next(iter(sample.model_usage.values())).total_cost
                                   for sample in log.samples], [0.0123, 0.0])
        finally:
            asyncio.run(client.aclose())

    def test_cli_task_discovery_and_limit(self):
        from click.testing import CliRunner
        from inspect_ai._cli.main import inspect
        from inspect_ai.log import list_eval_logs
        from inspect_ai.model._providers.mockllm import MockLLM
        task_file = Path(__file__).resolve().parents[1] / 'herbarium.py'
        log_dir = str(self.root / 'cli-logs')
        # MockLLM otherwise downloads a tokenizer to estimate its synthetic usage.
        with chdir(task_file.parent), \
             patch('inspect_ai._util.appdirs.user_data_path', return_value=self.root / 'inspect-data'), \
             patch('inspect_ai._util.appdirs.user_cache_path', return_value=self.root / 'inspect-cache'), \
             patch.object(MockLLM, 'count_tokens', AsyncMock(return_value=50)), \
             patch.object(socket.socket, 'connect', side_effect=AssertionError('Network disabled')):
            result = CliRunner().invoke(inspect, [
                'eval', task_file.name, '-T', f'dataset_path={self.root}',
                '-T', f'image_list={self.listing}', '--model', 'mockllm/model',
                '--limit', '1', '--display', 'none', '--ctl-server', 'false',
                '--log-dir', log_dir,
            ])
            self.assertEqual(result.exit_code, 0, f'{result.output}\n{result.exception!r}')
            logs = list_eval_logs(log_dir)
            self.assertEqual(len(logs), 1)
            log = read_eval_log(logs[0].name)
            self.assertEqual(log.status, 'success', str(log.error))
            self.assertEqual(len(log.samples), 1)
            self.assertEqual(log.samples[0].id, self.names[0])
            self.assertEqual(log.samples[0].scores['label_fields'].value['valid_json'], 0)


class ScoringTests(unittest.TestCase):
    def test_normalization_and_full_value_comparison(self):
        self.assertEqual(normalize('Species name', ' Salix  repens × purpurea Wim '),
                         normalize('Species name', 'salix repens x purpurea wim'))
        for source, expected in [('3. Mai 1855', '1855-05-03'), ('August 1862', '1862-08'),
                                 ('1936', '1936'), ('1855-5-3', '1855-05-03')]:
            self.assertEqual(normalize('Collection date', source), expected)
        self.assertNotEqual(normalize('Collection date', '1855-05-03'), normalize('Collection date', '1855-09-26'))
        self.assertNotEqual(normalize("Collector's name", 'Riese'), normalize("Collector's name", 'Hermann Riese'))
        self.assertNotEqual(normalize('Species name', 'Salix repens x purpurea'), normalize('Species name', 'Salix repens x cinerea'))

    def test_schema_rejects_malformed_missing_extra_and_nonstring_fields(self):
        correct = {field: [] if field in FACT_FIELDS else '' for field in OUTPUT_FIELDS}
        self.assertEqual(parse_answer(json.dumps(correct)), correct)
        for text in ['not JSON', '[]', '{}', '```json\n{}\n```',
                     json.dumps({**correct, 'extra': ''}),
                     json.dumps({**correct, 'Notes': None}),
                     json.dumps(correct)[:-1] + ', "Notes": "duplicate"}']:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_answer(text)

    def test_invalid_response_fails_known_fields_blank_references_excluded(self):
        reference = {field: '' for field in FIELDS}
        reference['Species name'] = 'Salix repens'
        state = SimpleNamespace(output=SimpleNamespace(completion='bad JSON'))
        result = asyncio.run(label_fields()(state, Target(json.dumps(reference))))
        self.assertEqual(result.value['valid_json'], 0)
        self.assertEqual(result.value['Species name'], 0)
        self.assertTrue(math.isnan(result.value['Region']))


class ProviderTests(unittest.TestCase):
    def test_inspect_providers_keep_credentials_separate(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'test-openai', 'OPENROUTER_API_KEY': 'test-router'}, clear=True):
            direct = get_model('openai/gpt-5.6-luna', responses_api=True)
            router = get_model('openrouter/google/gemini-2.5-pro')
            self.assertEqual(direct.api.api_key, 'test-openai')
            self.assertEqual(router.api.api_key, 'test-router')


class ProviderWireTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_and_invalid_costs_remain_unknown(self):
        from inspect_ai.model import GenerateConfig, ModelCall
        from inspect_ai.model._providers.openrouter import OpenRouterAPI
        with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'router-test'}):
            model = get_model('openrouter-cost/google/gemini-2.5-pro')
        for raw in [{}, {'cost': None}, {'cost': -1}, {'cost': True},
                    {'cost': '0.12'}, {'cost': float('nan')}, {'cost': float('inf')}]:
            output = ModelOutput.from_content('test', '{}')
            output.usage = ModelUsage(input_tokens=1, output_tokens=1, total_tokens=2)
            call = ModelCall(request={}, response={'usage': raw})
            with patch.object(OpenRouterAPI, 'generate', AsyncMock(return_value=(output, call))), \
                 self.assertLogs('openrouter_cost', level='WARNING'):
                result, _ = await model.api.generate([], [], 'none', GenerateConfig())
            self.assertIsNone(result.usage.total_cost)

    async def test_openai_and_openrouter_image_requests_offline(self):
        import httpx2
        from inspect_ai.model import ChatMessageUser, ContentImage, ContentText
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append((request, payload))
            if request.url.path.endswith('/responses'):
                response = {
                    'id': 'resp_test', 'object': 'response', 'created_at': 0,
                    'model': 'gpt-5.6-luna', 'status': 'completed',
                    'output': [{'type': 'message', 'id': 'msg_test', 'role': 'assistant',
                                'status': 'completed', 'content': [{'type': 'output_text',
                                'text': '{}', 'annotations': []}]}],
                    'usage': {'input_tokens': 3341, 'output_tokens': 586, 'total_tokens': 3927},
                }
            else:
                response = {
                    'id': 'chat_test', 'object': 'chat.completion', 'created': 0,
                    'model': 'google/gemini-2.5-pro',
                    'choices': [{'index': 0, 'finish_reason': 'stop',
                                 'message': {'role': 'assistant', 'content': '{}'}}],
                    'usage': {'prompt_tokens': 3341, 'completion_tokens': 586, 'total_tokens': 3927, 'cost': 0.0123},
                }
            return httpx2.Response(200, json=response)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'secret.png'
            Image.new('RGB', (10, 10)).save(path)
            message = ChatMessageUser(content=[ContentText(text='Read label'),
                                               ContentImage(image=image_input(path, 32))])
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'direct-test', 'OPENROUTER_API_KEY': 'router-test'}, clear=True), \
                 patch.object(socket.socket, 'connect', side_effect=AssertionError('Network disabled')):
                for name, key, endpoint in [
                    ('openai/gpt-5.6-luna', 'direct-test', 'https://api.openai.com/v1/responses'),
                    ('openrouter-cost/google/gemini-2.5-pro', 'router-test', 'https://openrouter.ai/api/v1/chat/completions'),
                ]:
                    async with httpx2.AsyncClient(transport=httpx2.MockTransport(respond)) as client:
                        opts = {'responses_api': True} if name.startswith('openai/') else {}
                        model = get_model(name, http_client=client, memoize=False, **opts)
                        result = await model.generate([message])
                        self.assertEqual(result.completion, '{}')
                        self.assertEqual(result.usage.input_tokens, 3341)
                        self.assertEqual(result.usage.output_tokens, 586)
                        if name.startswith('openrouter-cost/'):
                            self.assertEqual(result.usage.total_cost, 0.0123)
                        request, payload = requests[-1]
                        self.assertEqual(str(request.url), endpoint)
                        self.assertEqual(request.headers['authorization'], 'Bearer ' + key)
                        self.assertIn('data:image/jpeg;base64,', request.content.decode())
                        self.assertNotIn('secret.png', request.content.decode())
                        self.assertEqual(payload['model'], name.split('/', 1)[1])


if __name__ == '__main__':
    unittest.main()
