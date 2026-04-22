import json
import logging
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

from app.config import ENV_PROJECT

client = AsyncOpenAI(api_key=ENV_PROJECT.OPENAI_API_KEY)


async def recommend_best_carrier(carriers_json: dict, context: str) -> dict:
    """
    Analyzes multiple carrier JSON objects and returns a written recommendation
    in a human-readable insurance comparison format (like 'Based on our analysis...').

    Args:
        carriers_json (dict): {
            "carriers": [
                {
                    "carrier_name": "Aviva",
                    "lob_details": {...}
                },
                {
                    "carrier_name": "Intact",
                    "lob_details": {...}
                },
                ...
            ]
        }

    Returns:
        dict: {
            "recommended_carrier": str,
            "recommendation_text": str
        }
    """
    input_token = 0
    output_token = 0
    cached_token = 0

    if not carriers_json or "carriers" not in carriers_json:
        raise ValueError("Input must contain key 'carriers' with carrier JSON objects.")

    # Prepare summarized input for LLM
    serialized_carriers = json.dumps(carriers_json, indent=4)

    prompt = f"""
You are an insurance analyst comparing multiple Canadian carriers.

Each carrier JSON includes multiple LOBs (CGL, Property, Auto, Umbrella, etc.)
with coverages, deductibles, and premiums.

Task:
1. Review all carrier data.
2. Identify which carrier provides the most comprehensive and cost-effective offering.
3. Use the actual carrier/insurer names from the data (do NOT use labels like Option A or Option B).
4. Write a recommendation section in this format:

Recommendation  
Based on our analysis, <carrier_name> is recommended because:  
• Broader coverage across [specific LOBs].  
• Includes [special feature] automatically.  
• Carrier is [admitted / financially strong].  
• [Other relevant comparative advantage].  

Also mention comparison context (e.g., “compared to <other_carrier_name>” if relevant).

Return your result strictly in JSON:
{{
  "recommended_carrier": "<carrier_name which is insurer name from the carriers_json>",
  "recommendation_text": "<paragraph and bullet-style recommendation as shown>"
}}

Carrier Data:
{serialized_carriers}
"""

    try:
        response = await client.chat.completions.create(
            model=ENV_PROJECT.GPT_VERIFICATION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional insurance underwriting analyst.",
                },
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        logger.error(f"Carrier recommendation LLM call failed: {e}")
        return {
            "result": {
                "recommended_carrier": "N/A",
                "recommendation_text": "Carrier recommendation is currently unavailable. Please review carrier data manually.",
                "status": "error",
            },
            "token": {ENV_PROJECT.GPT_VERIFICATION_MODEL: {"input_token": 0, "output_token": 0, "cached_token": 0}}
        }

    content = response.choices[0].message.content
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
            it = getattr(usage, "input_tokens", None)
            ot = getattr(usage, "output_tokens", None)
            input_token += int((it if it is not None else (pt or 0)) or 0)
            output_token += int((ot if ot is not None else (ct or 0)) or 0)
            cached_token += int(
                getattr(usage, "cache_creation_input_tokens", 0)
                or getattr(usage, "cached_tokens", 0)
                or 0
            )
    except Exception:
        pass
    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "recommended_carrier": None,
            "recommendation_text": content.strip(),
            "note": "Model output was not valid JSON; returning raw text.",
        }

    # Build verification token map for return
    model_key = ENV_PROJECT.GPT_VERIFICATION_MODEL

    token_map = {
        model_key: {
            "input_token": input_token,
            "output_token": output_token,
            "cached_token": cached_token,
        }
    }

    return {"result": result, "token": token_map}
