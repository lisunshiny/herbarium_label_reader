try:
    from .grounding_dino import GroundingDinoPreprocessor
except ImportError:  # Optional for API-only extraction with detection disabled.
    class GroundingDinoPreprocessor:
        def __init__(self, *args, **kwargs):
            raise ImportError("Install torch and transformers to enable Grounding DINO")
