import base64
import io
from typing import List

from PIL import Image
from openai import OpenAI

from .openai import OpenAIModel


def encode_image(image: Image.Image, format: str = "jpeg") -> str:
    with io.BytesIO() as output:
        image.save(output, format=format)
        return f"data:image/{format};base64,{base64.b64encode(output.getvalue()).decode('utf-8')}"


def wrap_image(image: Image.Image) -> dict:
    return {
        "type": "image_url",
        "image_url": {
            "url": encode_image(image),
        },
    }


def wrap_text(text: str) -> dict:
    return {
        "type": "text",
        "text": str(text),
    }


def wrap_prompt(prompt: List) -> list:
    return [{
        "role": "user",
        "content": [
            wrap_image(d) if isinstance(d, Image.Image) else wrap_text(d)
            for d in prompt
        ],
    }]


class VLLMModel(OpenAIModel):
    def __init__(self, model_name: str, *args, **kwargs):
        self.raw_model_name = model_name
        model_id = self._parse_model_name(model_name)
        base_url = kwargs.pop("base_url", self._get_remote_base_url(kwargs.get("remote_server")))
        kwargs["base_url"] = base_url
        super().__init__(model_id, *args, **kwargs)

    def _parse_model_name(self, model_name: str) -> str:
        if not model_name.startswith("vllm:"):
            raise ValueError("vLLM model name must follow the prefix 'vllm:'")
        model_id = model_name.split("vllm:", 1)[1]
        if not model_id:
            raise ValueError("vLLM model name must follow the prefix 'vllm:'")
        return model_id

    def _get_base_url(self) -> str:
        return self._get_remote_base_url(self.remote_server)

    def _get_remote_base_url(self, remote_server: str | None) -> str:
        server = (remote_server or "http://localhost:8000").strip().rstrip("/")
        if server.endswith("/v1"):
            return server
        return f"{server}/v1"

    def _prepare_prompt(self, prompt: list) -> dict:
        return {"messages": wrap_prompt(prompt)}

    def _get_response(self, prepared_prompt: dict) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            **prepared_prompt,
            **self.options,
        )
        return response.choices[0].message.content
