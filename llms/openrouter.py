"""OpenRouter vision requests through its OpenAI-compatible chat API."""
import os

from PIL import Image
from omegaconf import OmegaConf
from openai import OpenAI, APIStatusError, InternalServerError

from .base import LLMBase
from .openai import encode_image


class OpenRouterModel(LLMBase):
    CLIENT_ERROR = APIStatusError
    SERVER_ERROR = InternalServerError

    def __init__(self, model_name, **kwargs):
        prefix = "openrouter:"
        if not model_name.startswith(prefix):
            raise ValueError("Use openrouter:<provider>/<model> (e.g. openrouter:google/gemini-2.5-pro)")
        model_id = model_name[len(prefix):]
        if "/" not in model_id or not all(model_id.split("/", 1)):
            raise ValueError("OpenRouter requires a provider-qualified model ID: <provider>/<model>")
        super().__init__(model_id, **kwargs)
        for name in ("init_options", "template_options", "generate_options"):
            value = getattr(self, name)
            if OmegaConf.is_config(value):
                setattr(self, name, OmegaConf.to_container(value, resolve=True))
        options = dict(self.init_options)
        api_key = options.pop("api_key", None) or os.environ.get("OPENROUTER_API_KEY")
        if not api_key or not api_key.strip():
            raise ValueError("Set OPENROUTER_API_KEY in your environment or project .env file")
        if "base_url" in options:
            raise ValueError("OpenRouter uses https://openrouter.ai/api/v1; remove llm.init_opts.base_url")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=api_key, **options
        )

    def _prepare_prompt(self, prompt):
        content = [
            {"type": "image_url", "image_url": {"url": encode_image(part.convert("RGB"))}}
            if isinstance(part, Image.Image) else {"type": "text", "text": str(part)}
            for part in prompt
        ]
        return {"messages": [{"role": "user", "content": content}]}

    def _get_response(self, prepared_prompt):
        options = {**self.template_options, **self.generate_options}
        if self.temperature is not None:
            options.setdefault("temperature", self.temperature)
        if options.get("stream"):
            raise ValueError("OpenRouter extraction requires stream=false")
        response = self.client.chat.completions.create(
            model=self.model_name, **prepared_prompt, **options
        )
        if not response.choices or not response.choices[0].message.content:
            raise ValueError("OpenRouter returned no text content; check model vision support and token limits")
        return response.choices[0].message.content

    def get_api_error_status_code(self, error):
        return error.status_code
