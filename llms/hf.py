import io
import base64
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
from warnings import warn

from .base import LLMBase

REPO2GGUF = {
    "google/gemma-4-31B-it-qat-q4_0-gguf": "gemma-4-31B_q4_0-it.gguf",
    "google/gemma-4-26B-A4B-it-qat-q4_0-gguf": "gemma-4-26B_q4_0-it.gguf",
    "google/gemma-4-12B-it-qat-q4_0-gguf": "gemma-4-12b-it-qat-q4_0.gguf",
    "google/gemma-4-E4B-it-qat-q4_0-gguf": "gemma-4-E4B_q4_0-it.gguf",
    "google/gemma-4-E2B-it-qat-q4_0-gguf": "gemma-4-E2B_q4_0-it.gguf",
}


def encode_image(image: Image.Image, format="jpeg") -> str:
    with io.BytesIO() as output:
        image.save(output, format=format)
        return f"data:image/{format};base64,{base64.b64encode(output.getvalue()).decode('utf-8')}"


def wrap_image(image: Image.Image) -> dict:
    return {
        "type": "image",
        "image": encode_image(image),
    }


def wrap_text(text: str) -> dict:
    return {
        "type": "text",
        "text": str(text),
    }


def wrap_prompt(prompt: list) -> list:
    return {
            "prompt": [
            {
                "role": "user",
                "content": [
                    wrap_image(d) if isinstance(d, Image.Image) else wrap_text(d)
                    for d in prompt
                ],
            }
        ],
        "images": [img for img in prompt if isinstance(img, Image.Image)],
    }


class HuggingFaceModel(LLMBase):
    # Transformers runs locally; use Exception as a catch-all for client/server errors
    CLIENT_ERROR = Exception
    SERVER_ERROR = Exception

    def __init__(self, model_name: str, *args, **kwargs):
        # model_name is expected with prefix 'hf:'
        self.raw_model_name = model_name
        model_id = self._parse_model_name(model_name)
        super().__init__(model_id, *args, **kwargs)

        additional_params = {}

        gguf_file = REPO2GGUF.get(model_id)
        quantization_parameters = {}

        if gguf_file:
            additional_params["gguf_file"] = gguf_file
        else:
            if model_id.lower().endswith("gguf"):
                warn(f"The model repository '{model_id}' looks like a GGUF file, but is currently not supported. Add a GGUF file mapping to the REPO2GGUF dictionary in llms/hf.py for this model or select a different model.")
            else:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
                quantization_parameters = {"quantization_config": bnb_config}

        self.processor = AutoProcessor.from_pretrained(
            model_id,
            **additional_params,
            **self.init_options,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            dtype=torch.bfloat16,
            **additional_params,
            **quantization_parameters,
        )

    def _parse_model_name(self, model_name: str) -> str:
        if not model_name.startswith("hf:"):
            raise ValueError("HuggingFace model name must follow the prefix 'hf:'")
        model_id = model_name.split("hf:", 1)[1]
        if not model_id:
            raise ValueError("HuggingFace model name must follow the prefix 'hf:'")
        return model_id

    def _prepare_prompt(self, prompt: list) -> dict:
        return wrap_prompt(prompt)

    def _get_response(self, prepared_prompt: dict) -> str:
        prompt = prepared_prompt["prompt"]
        images = prepared_prompt["images"]

        # opts = {**self.options}

        # max_soft_tokens = opts.pop("max_soft_tokens", 1120)

        # self.processor.max_soft_tokens = max_soft_tokens

        text = self.processor.apply_chat_template(
            prompt,
            tokenize=False,
            add_generation_prompt=True,
            **self.template_options,
        )
        print("Text:", text)
        inputs = self.processor(images=images, text=text, return_tensors="pt").to(self.model.device)

        inputs = {
            k: v.to(device=self.model.device, dtype=torch.bfloat16) if torch.is_floating_point(v) else v.to(self.model.device)
            for k, v in inputs.items()
        }

        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(**inputs, max_new_tokens=32768, do_sample=True, temperature=self.temperature, **self.generate_options)
        response = self.processor.decode(outputs[0][input_len:], skip_special_tokens=False)
        response = self.processor.parse_response(response, prefix=text)["content"]

        return response.strip()

    def get_api_error_status_code(self, error: Exception) -> int:
        # No HTTP status codes for local transformers; return 0
        return 0
