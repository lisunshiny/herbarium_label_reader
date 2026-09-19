"""Offline OpenRouter routing, wire-format, and web configuration tests."""
import base64
import io
import os
import tempfile
import unittest
from unittest.mock import patch

import httpx
from PIL import Image
from omegaconf import OmegaConf
from llms.openrouter import OpenRouterModel
from utils.extract_utils import ExtractionPipeline


class TestOpenRouter(unittest.TestCase):
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-router", "OPENAI_API_KEY": "wrong", "OPENAI_BASE_URL": "https://wrong.invalid"})
    def test_wire_request_and_csv(self):
        requests = []

        def respond(request):
            import json
            requests.append((request, json.loads(request.content)))
            return httpx.Response(200, json={
                "id": "offline", "object": "chat.completion", "created": 0,
                "model": "google/gemini-2.5-pro",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "Species name: Rosa canina\nNotes: test"}}],
            })

        cfg = OmegaConf.create({"llm": {"model_name": "openrouter:google/gemini-2.5-pro", "prompt": "Read label", "gen_opts": {"extra_body": {"provider": {"allow_fallbacks": False}}}}, "img_max_size": 2048, "batch_size": 1})
        pipeline = ExtractionPipeline(cfg)
        pipeline.llm.client.close()
        from openai import OpenAI
        pipeline.llm.client = OpenAI(api_key="test-router", base_url="https://openrouter.ai/api/v1", http_client=httpx.Client(transport=httpx.MockTransport(respond)))
        with tempfile.TemporaryDirectory() as directory, patch("os.getcwd", return_value=directory):
            results, _ = pipeline(Image.new("RGBA", (20, 10)), image_names=["label.jpg"])
            pipeline.save_results(results, os.path.join(directory, "result.csv"))
            self.assertEqual(results[0]["source_image"], "label.jpg")
            self.assertEqual(results[0]["Species name"], "Rosa canina")
        request, payload = requests[0]
        self.assertEqual(str(request.url), "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(request.headers["authorization"], "Bearer test-router")
        self.assertEqual(payload["model"], "google/gemini-2.5-pro")
        self.assertNotIn("temperature", payload)
        self.assertFalse(payload["provider"]["allow_fallbacks"])
        content = payload["messages"][0]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "Read label"})
        url = content[1]["image_url"]["url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        self.assertEqual(Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).size, (20, 10))
        pipeline.llm.client.close()

    @patch("llms.openrouter.OpenAI")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-router", "OPENAI_BASE_URL": "https://wrong.invalid"})
    def test_constructor_and_options(self, client):
        model = OpenRouterModel("openrouter:google/gemini-2.5-pro:free", temperature=0, generate_options={"max_tokens": 300, "extra_body": {"provider": {"allow_fallbacks": False}}})
        client.assert_called_once_with(base_url="https://openrouter.ai/api/v1", api_key="test-router")
        model._get_response(model._prepare_prompt(["Read", Image.new("RGB", (1, 1)), "Next", Image.new("RGB", (1, 1))]))
        args = client.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(args["model"], "google/gemini-2.5-pro:free")
        self.assertEqual(args["temperature"], 0)
        self.assertEqual(args["max_tokens"], 300)
        self.assertFalse(args["extra_body"]["provider"]["allow_fallbacks"])
        self.assertEqual(len(args["messages"][0]["content"]), 4)

    @patch.dict(os.environ, {"OPENAI_API_KEY": "not-router"}, clear=True)
    def test_missing_key(self):
        with self.assertRaisesRegex(ValueError, "OPENROUTER_API_KEY"):
            OpenRouterModel("openrouter:google/gemini-2.5-pro")

    def test_invalid_model(self):
        for name in ["google/gemini-2.5-pro", "openrouter:", "openrouter:gemini", "openrouter:/gemini", "openrouter:google/"]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                OpenRouterModel(name)

    @patch("llms.openrouter.OpenAI")
    @patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-router"})
    def test_empty_response_and_stream(self, client):
        model = OpenRouterModel("openrouter:google/gemini-2.5-pro")
        client.return_value.chat.completions.create.return_value.choices = []
        with self.assertRaisesRegex(ValueError, "no text"):
            model._get_response({"messages": []})
        model.generate_options = {"stream": True}
        with self.assertRaisesRegex(ValueError, "stream=false"):
            model._get_response({"messages": []})

    @patch("webapp.process_request.ExtractionPipeline")
    def test_web_preserves_options(self, pipeline):
        from webapp.process_request import create_pipeline
        config = OmegaConf.load("config.yaml")
        config.llm.gen_opts = {"max_tokens": 900}
        config.llm.init_opts = {"timeout": 25}
        create_pipeline("Read", False, .25, .3, "Find", "openrouter:google/gemini-2.5-pro", 2048, 0, config)
        actual = pipeline.call_args.args[0]
        self.assertEqual(actual.llm.model_name, "openrouter:google/gemini-2.5-pro")
        self.assertEqual(actual.llm.gen_opts.max_tokens, 900)
        self.assertEqual(actual.llm.init_opts.timeout, 25)
        self.assertFalse(actual.preprocessors.grounding_dino.enabled)
        self.assertEqual(config.llm.model_name, "gemini-2.5-pro")
