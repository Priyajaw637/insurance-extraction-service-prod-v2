import asyncio

from app.insurance_policy_processor.policy_orchestrator.models.state_models import (
    InsuranceDocumentState,
)
from app.insurance_policy_processor.processing_cost_calculator.token_pricing import (
    cost_mapping,
)
from app.logging_config import get_logger


logger = get_logger(__name__)


async def get_insurance_cost(total_token_usage, agent):
    input_without_cache_cost = 0
    input_with_cache_cost = 0
    cached_cost = 0
    output_cost = 0
    thinking_cost = 0

    for token_usage in total_token_usage:
        cost = cost_mapping.get(token_usage.get("model"))

        input_tokens = token_usage.get("input_tokens")
        cached_tokens = token_usage.get("cached_tokens")
        output_tokens = token_usage.get("output_tokens")
        thinking_tokens = token_usage.get("thinking_tokens")

        input_if_also_cache = input_tokens - cached_tokens

        # Final Cost
        input_without_cache_cost += (input_tokens / 1000000) * (cost.get("prompt"))
        input_with_cache_cost += (input_if_also_cache / 1000000) * (cost.get("prompt"))
        output_cost += (output_tokens / 1000000) * (cost.get("completion"))
        thinking_cost += (thinking_tokens / 1000000) * (cost.get("completion"))
        cached_cost += (cached_tokens / 1000000) * (cost.get("cached"))

    total_with_cache = input_with_cache_cost + cached_cost + output_cost + thinking_cost
    total_without_cache = input_without_cache_cost + output_cost + thinking_cost

    return {
        "agent": agent,
        "input_if_cache_found": input_with_cache_cost,
        "input_if_no_cache_found": input_without_cache_cost,
        "cached": cached_cost,
        "output": output_cost,
        "thinking_cost": thinking_cost,
        "total_with_cache": total_with_cache,
        "total_without_cache": total_without_cache,
    }


async def get_insurance_tokens(state: InsuranceDocumentState):
    model_tokens = {}

    all_token_sources = [
        state.coverage_mapping_cost,
        state.coverage_detail_mapping_cost,
        state.extraction_cost,
    ]

    for token_source in all_token_sources:
        for token_usage in token_source:
            model = token_usage.get("model")
            if model not in model_tokens:
                model_tokens[model] = {
                    "input_token": 0,
                    "output_token": 0,
                    "thinking_token": 0,
                }

            model_tokens[model]["input_token"] += token_usage.get("input_tokens", 0)
            model_tokens[model]["output_token"] += token_usage.get("output_tokens", 0)
            model_tokens[model]["thinking_token"] += token_usage.get(
                "thinking_tokens", 0
            )

    return model_tokens


async def calculate_insurance_cost(state: InsuranceDocumentState):
    token_usage = await get_insurance_tokens(state)
    state.token_consumption = token_usage
    return state

    # cost tasks including token usage calculation
    tasks = [
        get_insurance_cost(
            total_token_usage=state.coverage_mapping_cost,
            agent="Coverage Mapper",
        ),
        get_insurance_cost(
            total_token_usage=state.coverage_detail_mapping_cost,
            agent="Coverage Detail Mapper",
        ),
        get_insurance_cost(
            total_token_usage=state.extraction_cost,
            agent="Policy Data Extractor",
        ),
        get_insurance_tokens(state),
    ]

    total_without_cache = 0
    *all_costs, token_usage = await asyncio.gather(*tasks)

    # print("\n\n=====================TOTAL COST USAGE===============================\n")
    for cost in all_costs:
        # print("Agent: ", cost.get("agent"))
        # print(
        #     f"    Input Cost: $ {cost.get("input_if_no_cache_found"):.3f}",
        # )
        # print(f"    Output Cost: $ {cost.get("output"):.3f}")
        # print(f"    Thinking Cost: $ {cost.get("thinking_cost"):.3f}")
        # print(f"    Total Cost: $ {cost.get("total_without_cache"):.3f}")
        logger.info(
            f"Cost for ({cost.get('agent')}): $ {cost.get('total_without_cache'):.3f}"
        )
        total_without_cache += cost.get("total_without_cache")

    # print("========================================================================\n")
    # print(f"TOTAL COST (without cache): $ {total_without_cache:.3f}")

    # print("========================================================================\n")

    logger.info(f"Total Cost: $ {total_without_cache:.3f}")

    state.token_consumption = token_usage

    # print("Token Usage by Model:", token_usage)
    return state
