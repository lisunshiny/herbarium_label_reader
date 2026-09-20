"""Preserve OpenRouter's returned charge in Inspect's native usage logs.

Remove this adapter when Inspect's OpenRouter provider copies usage.cost itself.
The provider import is internal, so Inspect is pinned in pyproject.toml.
"""
import logging
import math

from inspect_ai.model import ModelOutput, modelapi
from inspect_ai.model._providers.openrouter import OpenRouterAPI


def cost_breakdown(usage):
    """Keep router billing separate; BYOK incurs a separate upstream charge."""
    def amount(value):
        return float(value) if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None

    usage = usage if isinstance(usage, dict) else {}
    details = usage.get("cost_details")
    details = details if isinstance(details, dict) else {}
    router = amount(usage.get("cost"))
    upstream = amount(details.get("upstream_inference_cost"))
    byok = usage.get("is_byok") is True
    total = (router + upstream if router is not None and upstream is not None else None) if byok else router
    return {"version": 2, "is_byok": byok, "openrouter_cost": router,
            "upstream_inference_cost": upstream, "total_cost": total}


def apply_cost(output, response):
    usage = response.get("usage") if isinstance(response, dict) else None
    breakdown = cost_breakdown(usage)
    output.metadata = {**(output.metadata or {}), "openrouter_billing": breakdown}
    if output.usage is not None:
        output.usage.total_cost = breakdown["total_cost"]
    if breakdown["total_cost"] is None:
        logging.getLogger(__name__).warning(
            "OpenRouter billing is incomplete; total cost is unknown. See openrouter_billing metadata."
        )
    return breakdown


@modelapi(name="openrouter-cost")
class OpenRouterCostAPI(OpenRouterAPI):
    async def generate(self, input, tools, tool_choice, config):
        result = await super().generate(input, tools, tool_choice, config)
        if isinstance(result, tuple):
            output, call = result
            if isinstance(output, ModelOutput):
                apply_cost(output, call.response if call is not None else None)
        return result
