import io
import base64
import os
from PIL import Image
from openai import OpenAI, APIStatusError, InternalServerError
from omegaconf import OmegaConf

from .base import LLMBase


def encode_image(image: Image.Image, format="jpeg") -> str:
    with io.BytesIO() as output:
        image.convert("RGB").save(output, format=format)
        return f"data:image/{format};base64,{base64.b64encode(output.getvalue()).decode('utf-8')}"


def wrap_image(image: Image.Image) -> dict:
    return {
        "type": "input_image",
        "image_url": encode_image(image),
    }


def wrap_text(text: str) -> dict:
    return {
        "type": "input_text",
        "text": text,
    }


def wrap_prompt(prompt: list) -> list:
    return [
        {"role": "user", "content": prompt},
    ]


class OpenAIModel(LLMBase):
    CLIENT_ERROR = APIStatusError
    SERVER_ERROR = InternalServerError

    def __init__(self, *args, **kwargs):
        base_url = kwargs.pop("base_url", None)
        super().__init__(*args, **kwargs)
        for name in ("init_options", "template_options", "generate_options"):
            value = getattr(self, name)
            if OmegaConf.is_config(value):
                setattr(self, name, OmegaConf.to_container(value, resolve=True))
        options = dict(self.init_options)
        # An explicit endpoint is still supported for subclasses such as vLLM.
        base_url = base_url or options.pop("base_url", None) or "https://api.openai.com/v1"
        options.pop("base_url", None)
        api_key = options.pop("api_key", None) or os.environ.get("OPENAI_API_KEY")
        if not api_key or not api_key.strip():
            raise ValueError("Set OPENAI_API_KEY in your environment or project .env file")
        self.client = OpenAI(base_url=base_url, api_key=api_key, **options)

    def _prepare_prompt(self, prompt: list) -> dict:
        return {
            "input": wrap_prompt([
                wrap_image(d) if isinstance(d, Image.Image) else wrap_text(d)
                for d in prompt
            ])
        }

    def _get_response(self, prepared_prompt: dict) -> str:
        options = {**self.template_options, **self.generate_options}
        if self.temperature is not None:
            options.setdefault("temperature", self.temperature)
        if options.get("stream"):
            raise ValueError("OpenAI extraction requires stream=false")
        if options.get("background"):
            raise ValueError("OpenAI extraction requires background=false")
        response = self.client.responses.create(
            model=self.model_name,
            **prepared_prompt,
            **options,
        )
        if not response.output_text or not response.output_text.strip():
            raise ValueError("OpenAI returned no text content; check model vision support and token limits")
        return response.output_text

    def get_api_error_status_code(self, error: Exception) -> int:
        return error.status_code
