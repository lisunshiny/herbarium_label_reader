"""Offline direct OpenAI tests using the real SDK with an in-memory transport."""
import base64
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pandas as pd
from PIL import Image
from openai import OpenAI
from omegaconf import OmegaConf

from llms.openai import OpenAIModel
from llms.vllm import VLLMModel

ROOT = Path(__file__).resolve().parents[1]


class TestDirectOpenAI(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.dict(os.environ, {
            "PATH": os.environ.get("PATH", ""),
            "OPENAI_API_KEY": "test-direct", "OPENROUTER_API_KEY": "test-router",
            "OPENAI_BASE_URL": "https://wrong.invalid", "GRADIO_ANALYTICS_ENABLED": "False",
        }, clear=True))
        self.directory = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.requests = []

        def respond(request):
            self.requests.append((request, json.loads(request.content)))
            return httpx.Response(200, json={
                "id": "resp_offline", "object": "response", "created_at": 0,
                "model": "gpt-4.1", "status": "completed",
                "output": [{"id": "msg_offline", "type": "message", "role": "assistant",
                            "status": "completed", "content": [{"type": "output_text",
                            "text": "Species name: Rosa canina\nNotes: test", "annotations": []}]}],
            })

        def client(**kwargs):
            kwargs["http_client"] = httpx.Client(transport=httpx.MockTransport(respond))
            instance = OpenAI(**kwargs)
            self.stack.callback(instance.close)
            return instance

        self.stack.enter_context(patch("llms.openai.OpenAI", side_effect=client))
        self.stack.enter_context(patch("os.getcwd", return_value=str(self.directory)))
        self.cfg = OmegaConf.load(ROOT / "config.yaml")
        self.cfg.preprocessors.grounding_dino.enabled = False
        self.cfg.llm.model_name = "gpt-4.1"
        self.cfg.batch_size = 1
        self.cfg.n_images = 1

    def test_wire_images_options_and_credentials(self):
        model = OpenAIModel("gpt-4.1", temperature=0,
                            template_options=OmegaConf.create({"text": {"format": {"type": "text"}}}),
                            generate_options=OmegaConf.create({"max_output_tokens": 300, "temperature": .2}))
        self.assertIn("Rosa canina", model.prompt(["Read", Image.new("RGBA", (20, 10)),
                                                  "Next", Image.new("P", (2, 3))]))
        request, payload = self.requests[0]
        self.assertEqual(str(request.url), "https://api.openai.com/v1/responses")
        self.assertEqual(request.headers["authorization"], "Bearer test-direct")
        self.assertEqual(payload["model"], "gpt-4.1")
        self.assertEqual(payload["temperature"], .2)
        self.assertEqual(payload["max_output_tokens"], 300)
        self.assertEqual(payload["text"], {"format": {"type": "text"}})
        content = payload["input"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["input_text", "input_image", "input_text", "input_image"])
        url = content[1]["image_url"]
        self.assertTrue(url.startswith("data:image/jpeg;base64,"))
        with Image.open(io.BytesIO(base64.b64decode(url.split(",")[1]))) as image:
            self.assertEqual(image.size, (20, 10))
            self.assertEqual(image.mode, "RGB")

    def test_missing_or_blank_key_never_uses_openrouter(self):
        for value in [None, "", "   "]:
            if value is None:
                os.environ.pop("OPENAI_API_KEY", None)
            else:
                os.environ["OPENAI_API_KEY"] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                OpenAIModel("gpt-4.1")
        self.assertEqual(self.requests, [])

    def test_null_temperature_and_invalid_output_modes(self):
        model = OpenAIModel("gpt-4.1")
        model.prompt(["Read"])
        self.assertNotIn("temperature", self.requests[0][1])
        for option in ["stream", "background"]:
            model.generate_options = {option: True}
            with self.assertRaisesRegex(ValueError, option + "=false"):
                model.prompt(["Read"])
        model.generate_options = {}
        with patch.object(model.client.responses, "create", return_value=SimpleNamespace(output_text="")):
            with self.assertRaisesRegex(ValueError, "no text content"):
                model.prompt(["Read"])

    def test_vllm_endpoint_and_explicit_key_preserved(self):
        model = VLLMModel("vllm:local/model", remote_server="http://localhost:8001",
                          init_options={"api_key": "test-local"})
        self.assertEqual(str(model.client.base_url), "http://localhost:8001/v1/")
        self.assertEqual(model.client.api_key, "test-local")

    def test_cli_project_dotenv_and_csv(self):
        import extract_data
        (self.directory / ".env").write_text("OPENAI_API_KEY=test-dotenv\nOPENROUTER_API_KEY=other\n")
        image_dir = self.directory / "handwritten"
        image_dir.mkdir()
        Image.new("RGB", (10, 10)).save(image_dir / "label.png")
        image_list = self.directory / "handwritten.txt"
        image_list.write_text("label.png\n")
        self.cfg.dataset_path = str(self.directory)
        self.cfg.image_list = str(image_list)
        for exported in [False, True]:
            if exported:
                os.environ["OPENAI_API_KEY"] = "test-exported"
            else:
                os.environ.pop("OPENAI_API_KEY", None)
            output_dir = self.directory / str(exported)
            runtime = SimpleNamespace(runtime=SimpleNamespace(output_dir=str(output_dir)))
            with patch.object(extract_data, "__file__", str(self.directory / "extract_data.py")), \
                 patch("hydra.core.hydra_config.HydraConfig.get", return_value=runtime):
                extract_data.main.__wrapped__(self.cfg)
            result = pd.read_csv(output_dir / "extracted_data.csv")
            self.assertEqual(result.iloc[0]["source_image"], "label.png")
            self.assertEqual(result.iloc[0]["Species name"], "Rosa canina")
            expected = "test-exported" if exported else "test-dotenv"
            self.assertEqual(self.requests[-1][0].headers["authorization"], "Bearer " + expected)

    def test_web_launch_dotenv_and_both_tabs(self):
        import app
        (self.directory / ".env").write_text("OPENAI_API_KEY=test-web\n")
        (self.directory / "webapp_supported_models.txt").write_text("gpt-4.1\ngpt-4.1-mini\n")
        os.environ.pop("OPENAI_API_KEY")
        # Capture real entry-point callbacks; never launch a server or share tunnel.
        with patch.object(app, "__file__", str(self.directory / "app.py")), \
             patch.object(app.gr, "Interface") as interface, \
             patch.object(app.gr, "TabbedInterface"):
            app.main.__wrapped__(self.cfg)
        self.assertEqual(os.environ["OPENAI_API_KEY"], "test-web")
        single, batch = [call.kwargs["fn"] for call in interface.call_args_list]
        args = ("Read label", False, .25, .3, "Find label", "gpt-4.1", 2048, 0)
        result, images = single(Image.new("RGBA", (10, 10)), *args)
        self.assertEqual(result["Species name"], "Rosa canina")
        self.assertNotIn("source_image", result)
        self.assertEqual(len(images), 1)
        # Gradio file inputs are strings with a .name attribute.
        from gradio.utils import NamedString
        path = self.directory / "web.png"
        Image.new("RGB", (10, 10)).save(path)
        previous = Path(os.fsdecode(os.getcwdb()))
        try:
            os.chdir(self.directory)
            results, _, output = batch([NamedString(str(path))], 1, "csv", *args)
            self.assertEqual(results[0]["source_image"], "web.png")
            self.assertEqual(pd.read_csv(output).iloc[0]["Species name"], "Rosa canina")
        finally:
            os.chdir(previous)
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all(req.headers["authorization"] == "Bearer test-web" for req, _ in self.requests))


if __name__ == "__main__":
    unittest.main()
