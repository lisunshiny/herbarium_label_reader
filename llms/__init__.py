try:
    from .gemini import GeminiModel
except ImportError:  # pragma: no cover - optional dependency
    class GeminiModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("google-genai is required to use GeminiModel")

try:
    from .openai import OpenAIModel
except ImportError:  # pragma: no cover - optional dependency
    class OpenAIModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("openai is required to use OpenAIModel")

try:
    from .groq import GroqModel
except ImportError:  # pragma: no cover - optional dependency
    class GroqModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("groq is required to use GroqModel")

try:
    from .ollama import OllamaModel
except ImportError:  # pragma: no cover - optional dependency
    class OllamaModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("ollama is required to use OllamaModel")

try:
    from .hf import HuggingFaceModel
except ImportError:  # pragma: no cover - optional dependency
    class HuggingFaceModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("transformers is required to use HuggingFaceModel")

try:
    from .vllm import VLLMModel
except ImportError:  # pragma: no cover - optional dependency
    class VLLMModel:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            raise ImportError("openai is required to use VLLMModel")
