"""Preserve OpenRouter's returned charge in Inspect's native usage logs.

Remove this adapter when Inspect's OpenRouter provider copies usage.cost itself.
The provider import is internal, so Inspect is pinned in pyproject.toml.
"""
import logging
import math

from inspect_ai.model import ModelOutput, modelapi
from inspect_ai.model._providers.openrouter import OpenRouterAPI


@modelapi(name="openrouter-cost")
class OpenRouterCostAPI(OpenRouterAPI):
    async def generate(self, input, tools, tool_choice, config):
        result = await super().generate(input, tools, tool_choice, config)
        if isinstance(result, tuple):
            output, call = result
            if isinstance(output, ModelOutput) and output.usage is not None:
                response = call.response if call is not None else None
                usage = response.get("usage") if isinstance(response, dict) else None
                cost = usage.get("cost") if isinstance(usage, dict) else None
                if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
                    output.usage.total_cost = float(cost)
                else:
                    logging.getLogger(__name__).warning(
                        "OpenRouter did not return a valid usage.cost; recorded cost totals may be incomplete."
                    )
        return result
