import base64
import io
from ollama import Client, ResponseError
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
            print("Image size:", d.size)
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
    CLIENT_ERROR = ResponseError
    SERVER_ERROR = ResponseError

    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        self.raw_model_name = model_name

        self.model_id, self.mode = self._parse_model_name(model_name)

        self.client = Client(host=self.remote_server)

    def _parse_model_name(self, model_name: str):
        model_id = model_name.split("ollama:", 1)[1]
        if not model_id:
            raise ValueError("Ollama model name must follow the prefix 'ollama:'")
        return model_id, "remote"

    def _prepare_prompt(self, prompt: list) -> dict:
        return {"messages": wrap_prompt(prompt)}

    def _get_response(self, prepared_prompt: dict) -> str:
        if not self.remote_server:
            raise ValueError("Ollama server URL is not configured for remote Ollama requests.")

        options = {
            "num_ctx": 32768,
        }

        if self.temperature is not None:
            options["temperature"] = self.temperature

        response = self.client.chat(
            model=self.model_id,
            messages=prepared_prompt["messages"],
            options=options,
            stream=False,
        )

        return response.message.content

    def _prompt_to_text(self, prepared_prompt: dict) -> str:
        return "\n\n".join(message.get("content", "") for message in prepared_prompt.get("messages", []))

    def get_api_error_status_code(self, error: Exception) -> int:
        return error.status_code
