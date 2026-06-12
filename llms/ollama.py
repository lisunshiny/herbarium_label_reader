import base64
import io
import json
import shutil
import subprocess
import urllib.error
import urllib.parse
import urllib.request
from PIL import Image

from .base import LLMBase


def encode_image(image: Image.Image, format="jpeg") -> str:
    with io.BytesIO() as output:
        image.save(output, format=format)
        return base64.b64encode(output.getvalue()).decode('utf-8')


def wrap_prompt(prompt: list) -> list:
    image_data = []
    text_data = ""

    for d in prompt:
        if isinstance(d, Image.Image):
            image_data.append(encode_image(d))
        else:
            text_data += str(d) + "\n"

    return [
        {
            "role": "user",
            "content": text_data,
            "images": image_data,
        }
    ]


class OllamaModel(LLMBase):
    CLIENT_ERROR = urllib.error.HTTPError
    SERVER_ERROR = urllib.error.HTTPError
    DEFAULT_SERVER_URL = "http://127.0.0.1:11434"

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.raw_model_name = model_name

        if not self.remote_server:
            self.remote_server = self.DEFAULT_SERVER_URL

        self.model_id, self.server_url, self.mode = self._parse_model_name(model_name)

    def _parse_model_name(self, model_name: str):
        if model_name.startswith("ollama-local:"):
            model_id = model_name.split("ollama-local:", 1)[1]
            if not model_id:
                raise ValueError("Ollama local model name must follow the prefix 'ollama-local:'")
            return model_id, None, "local"

        if model_name.startswith("ollama:"):
            model_id = model_name.split("ollama:", 1)[1]
            if not model_id:
                raise ValueError("Ollama model name must follow the prefix 'ollama:'")
            return model_id, self.remote_server, "remote"

        raise ValueError(
            "Unsupported Ollama model name. Use 'ollama-local:<model>', " + \
            "or 'ollama:<model>'"
        )

    def _prepare_prompt(self, prompt: list) -> dict:
        return {"messages": wrap_prompt(prompt)}

    def _get_response(self, prepared_prompt: dict) -> str:
        if self.mode == "local":
            return self._get_local_response(prepared_prompt)
        return self._get_remote_response(prepared_prompt)

    def _get_local_response(self, prepared_prompt: dict) -> str:
        if shutil.which("ollama") is None:
            raise FileNotFoundError(
                "Ollama CLI was not found in PATH. Install Ollama or use a remote Ollama server model."
            )

        prompt_text = self._prompt_to_text(prepared_prompt)
        command = ["ollama", "run", self.model_id, "--prompt", prompt_text]
        if self.temperature is not None:
            command.extend(["--temperature", str(self.temperature)])

        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )

        if completed.returncode != 0:
            raise RuntimeError(
                f"Ollama local CLI failed with status {completed.returncode}: {completed.stderr.strip()}"
            )

        return completed.stdout.strip()

    def _get_remote_response(self, prepared_prompt: dict) -> str:
        if not self.server_url:
            raise ValueError("Ollama server URL is not configured for remote Ollama requests.")

        endpoint = urllib.parse.urljoin(self.server_url, "/api/chat")
        payload = {
            "model": self.model_id,
            "messages": prepared_prompt["messages"],
            "options": {},
            "stream": False,
        }
        if self.temperature is not None:
            payload["options"]["temperature"] = self.temperature

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as error:
            raise error

        parsed = json.loads(body)
        try:
            return parsed["message"]["content"]
        except (KeyError, IndexError) as error:
            raise RuntimeError(f"Unexpected Ollama response format: {error}\nResponse body: {body}")

    def _prompt_to_text(self, prepared_prompt: dict) -> str:
        return "\n\n".join(message.get("content", "") for message in prepared_prompt.get("messages", []))

    def get_api_error_status_code(self, error: Exception) -> int:
        return getattr(error, "code", None) if isinstance(error, urllib.error.HTTPError) else None
