import logging
import os
import asyncio
import json
import re
import time
import aiofiles
import aiohttp
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union, Set, Tuple
import uuid
from app.config import ENV_PROJECT
from rapidfuzz import fuzz
from app.insurance_policy_processor.policy_orchestrator.enums import GeminiModels
from app.insurance_policy_processor.policy_orchestrator.modules.gemini import (
    gemini_client,
)

logger = logging.getLogger(__name__)



class Anomaly(BaseModel):
    anomalyFieldName: Optional[str] = None
    policy_value: Any = None
    binder_value: Any = None
    quote_value: Any = None
    status: Optional[str] = None
    field_name: Optional[str] = None
    policy_pages: Optional[List[int]] = None
    binder_pages: Optional[List[int]] = None
    quote_pages: Optional[List[int]] = None


async def check_semantic_similarity_batch(
    comparisons: List[Tuple[str, str]],
    verification_provider: str = "openai",
    verification_model: Optional[str] = None,
    token_tracker: Optional[Dict[str, int]] = None
) -> Dict[Tuple[str, str], bool]:
    """
    Batch check semantic similarity for multiple identifier pairs.
    Returns a dictionary mapping (identifier1, identifier2) tuples to boolean results.
    
    Args:
        comparisons: List of (identifier1, identifier2) tuples to compare
        verification_provider: Provider to use ("openai" or "gemini")
        verification_model: Model to use (optional)
        token_tracker: Dictionary to track token usage
    """
    if not comparisons:
        return {}
    
    results = {}
    
    try:
        # Build batch prompt
        prompt_parts = [
            "Determine whether each pair of identifiers refers to the same underlying entity, location, object, or concept.",
            "Respond with only 'YES' or 'NO' for each pair, one per line.\n",
            "Treat two identifiers as the same if they are equivalent in meaning, even when extracted differently or expressed in different formats. This includes cases where:",
            "- One value is a shortened, simplified, or partial version of the other.",
            "- One contains extra descriptive details such as addresses, floors, prefixes, or labels.",
            "- They refer to the same building, location, or unit even if one version includes a full address or expanded description.",
            "- They describe the same claim type, vehicle, or entity with different wording, numbering, or formatting.",
            "- They differ only in prefixes, suffixes, ordering, titles, or abbreviations.\n",
            "Only answer based on semantic equivalence, not formatting differences.",
        ]

        
        for idx, (id1, id2) in enumerate(comparisons, 1):
            prompt_parts.append(f"Pair {idx}:")
            prompt_parts.append(f"  Identifier 1: {id1}")
            prompt_parts.append(f"  Identifier 2: {id2}")
            prompt_parts.append(f"  Answer (YES/NO):")
        
        prompt = "\n".join(prompt_parts)
        
        if (verification_provider or "gemini").lower() == "openai":
            try:
                client = AsyncOpenAI(api_key=ENV_PROJECT.OPENAI_API_KEY)
                response = await client.chat.completions.create(
                    model=(verification_model or ENV_PROJECT.GPT_VERIFICATION_MODEL or "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1,
                )
                answer = response.choices[0].message.content.strip()
                
                # Parse batch results - look for YES/NO answers
                lines = [line.strip() for line in answer.split('\n') if line.strip()]
                # Try to find answers in format "Pair X: YES/NO" or just "YES/NO"
                for idx, (id1, id2) in enumerate(comparisons):
                    result = False
                    # Look for this pair's answer
                    pair_num = idx + 1
                    for line in lines:
                        line_upper = line.upper()
                        # Check if this line contains the pair number or is a standalone YES/NO
                        if f"PAIR {pair_num}" in line_upper or (idx == 0 and "PAIR 1" not in answer.upper() and line_upper.startswith("YES")):
                            result = "YES" in line_upper
                            break
                        # Also check for standalone YES/NO answers in order
                        if idx < len(lines) and line_upper in ["YES", "NO"]:
                            result = line_upper == "YES"
                            break
                    results[(id1, id2)] = result

                # Track tokens
                if token_tracker is not None:
                    try:
                        usage = getattr(response, 'usage', None)
                        if usage:
                            pt = getattr(usage, 'prompt_tokens', None)
                            ct = getattr(usage, 'completion_tokens', None)
                            it = getattr(usage, 'input_tokens', None)
                            ot = getattr(usage, 'output_tokens', None)
                            token_tracker['input_token'] = token_tracker.get('input_token', 0) + int((it if it is not None else (pt or 0)) or 0)
                            token_tracker['output_token'] = token_tracker.get('output_token', 0) + int((ot if ot is not None else (ct or 0)) or 0)
                            token_tracker['cached_token'] = token_tracker.get('cached_token', 0) + int(getattr(usage, 'cache_creation_input_tokens', 0) or getattr(usage, 'cached_tokens', 0) or 0)
                    except Exception as e:
                        logger.warning(f"Failed to track OpenAI tokens: {e}")
                
                return results
            except Exception as e:
                logger.warning(f"OpenAI batch semantic check failed: {e}")
                # Fallback to individual calls
                for id1, id2 in comparisons:
                    results[(id1, id2)] = False
                return results
        else:
            try:
                resp = await gemini_client.aio.models.generate_content(
                    model=(verification_model or GeminiModels.FLASH.value),
                    contents=[prompt]
                )
                answer = (resp.text or '').strip()
                
                # Parse batch results - look for YES/NO answers
                lines = [line.strip() for line in answer.split('\n') if line.strip()]
                # Try to find answers in format "Pair X: YES/NO" or just "YES/NO"
                for idx, (id1, id2) in enumerate(comparisons):
                    result = False
                    # Look for this pair's answer
                    pair_num = idx + 1
                    for line in lines:
                        line_upper = line.upper()
                        # Check if this line contains the pair number or is a standalone YES/NO
                        if f"PAIR {pair_num}" in line_upper or (idx == 0 and "PAIR 1" not in answer.upper() and line_upper.startswith("YES")):
                            result = "YES" in line_upper
                            break
                        # Also check for standalone YES/NO answers in order
                        if idx < len(lines) and line_upper in ["YES", "NO"]:
                            result = line_upper == "YES"
                            break
                    results[(id1, id2)] = result

                # Track tokens
                if token_tracker is not None:
                    try:
                        usage = getattr(resp, 'usage_metadata', None)
                        if usage:
                            token_tracker['input_token'] = token_tracker.get('input_token', 0) + int(getattr(usage, 'prompt_token_count', 0) or getattr(usage, 'input_tokens', 0) or 0)
                            token_tracker['output_token'] = token_tracker.get('output_token', 0) + int(getattr(usage, 'candidates_token_count', 0) or getattr(usage, 'output_tokens', 0) or 0)
                            token_tracker['cached_token'] = token_tracker.get('cached_token', 0) + int(getattr(usage, 'cached_content_token_count', 0) or 0)
                    except Exception as e:
                        logger.warning(f"Failed to track Gemini tokens: {e}")
                
                return results
            except Exception as e:
                logger.warning(f"Gemini batch semantic check failed: {e}")
                # Fallback to individual calls
                for id1, id2 in comparisons:
                    results[(id1, id2)] = False
                return results
    except Exception as e:
        logger.warning(f"Batch semantic similarity check failed: {e}")
        # Fallback: return False for all
        for id1, id2 in comparisons:
            results[(id1, id2)] = False
        return results

async def check_semantic_similarity_global(
    identifier1: str, 
    identifier2: str,
    verification_provider: str = "openai",
    verification_model: Optional[str] = None,
    token_tracker: Optional[Dict[str, int]] = None
) -> bool:
    """
    Check if two identifiers are semantically similar using GPT/Gemini.
    Returns True if they refer to the same entity/location, False otherwise.
    
    Args:
        identifier1: First identifier to compare
        identifier2: Second identifier to compare
        verification_provider: Provider to use ("openai" or "gemini")
        verification_model: Model to use (optional)
        token_tracker: Dictionary to track token usage (optional, should have keys: input_token, output_token, cached_token)
    """
    try:
        prompt = (
            f"Are these two identifiers referring to the same entity, location, or concept? "
            f"Answer with only 'YES' or 'NO'.\n\n"
            f"Identifier 1: {identifier1}\n"
            f"Identifier 2: {identifier2}\n\n"
            f"Consider these as equivalent if they:\n"
            f"- Refer to the same claim type (e.g., 'Cyber Extortion' vs '1.D Cyber Extortion')\n"
            f"- Refer to the same location (e.g., 'Loc 1' vs 'Location 1' vs 'At Loc 1')\n"
            f"- Refer to the same vehicle (with or without VIN numbers)\n"
            f"- Are the same term with different formatting or prefixes\n\n"
            f"Answer:"
        )
        
        if (verification_provider or "gemini").lower() == "openai":
            try:
                client = AsyncOpenAI(api_key=ENV_PROJECT.OPENAI_API_KEY)
                response = await client.chat.completions.create(
                    model=(verification_model or ENV_PROJECT.GPT_VERIFICATION_MODEL or "gpt-4o-mini"),
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1,
                    # max_tokens=10
                )
                answer = response.choices[0].message.content.strip().upper()
                result = answer.startswith("YES")

                # Track tokens if tracker provided
                if token_tracker is not None:
                    try:
                        usage = getattr(response, 'usage', None)
                        if usage:
                            pt = getattr(usage, 'prompt_tokens', None)
                            ct = getattr(usage, 'completion_tokens', None)
                            it = getattr(usage, 'input_tokens', None)
                            ot = getattr(usage, 'output_tokens', None)
                            token_tracker['input_token'] = token_tracker.get('input_token', 0) + int((it if it is not None else (pt or 0)) or 0)
                            token_tracker['output_token'] = token_tracker.get('output_token', 0) + int((ot if ot is not None else (ct or 0)) or 0)
                            token_tracker['cached_token'] = token_tracker.get('cached_token', 0) + int(getattr(usage, 'cache_creation_input_tokens', 0) or getattr(usage, 'cached_tokens', 0) or 0)
                    except Exception as e:
                        logger.warning(f"Failed to track OpenAI tokens: {e}")
                
                return result
            except Exception as e:
                logger.warning(f"OpenAI semantic check failed: {e}")
                return False
        else:
            try:
                resp = await gemini_client.aio.models.generate_content(
                    model=(verification_model or GeminiModels.FLASH.value),
                    contents=[prompt]
                )
                answer = (resp.text or '').strip().upper()
                result = answer.startswith("YES")

                # Track tokens if tracker provided
                if token_tracker is not None:
                    try:
                        usage = getattr(resp, 'usage_metadata', None)
                        if usage:
                            token_tracker['input_token'] = token_tracker.get('input_token', 0) + int(getattr(usage, 'prompt_token_count', 0) or getattr(usage, 'input_tokens', 0) or 0)
                            token_tracker['output_token'] = token_tracker.get('output_token', 0) + int(getattr(usage, 'candidates_token_count', 0) or getattr(usage, 'output_tokens', 0) or 0)
                            token_tracker['cached_token'] = token_tracker.get('cached_token', 0) + int(getattr(usage, 'cached_content_token_count', 0) or 0)
                    except Exception as e:
                        logger.warning(f"Failed to track Gemini tokens: {e}")
                
                return result
            except Exception as e:
                logger.warning(f"Gemini semantic check failed: {e}")
                return False
    except Exception as e:
        logger.warning(f"Semantic similarity check failed: {e}")
        return False

async def commercial_compare_jsons(
    policy: Dict[str, Any] = None,
    binder: Dict[str, Any] = None,
    quote: Dict[str, Any] = None,
    similarity_threshold: float = 70.0,
    verification_provider: str = "openai",
    verification_model: Optional[str] = None,
):
    """
    Enhanced deep comparison that ensures every field at every nesting level
    is compared using identifier-based matching for lists of dictionaries.
    """
    start_time = time.time()

    logger.info("Starting enhanced deep JSON comparison")
    
    matched_count = 0
    mismatched_count = 0
    anomalies = []
    
    # Verification token accounting
    verification_input_tokens = 0
    verification_output_tokens = 0
    verification_cached_tokens = 0
    
    # Token tracker for semantic similarity checks
    semantic_token_tracker = {
        'input_token': 0,
        'output_token': 0,
        'cached_token': 0
    }
    
    # Track semantic matching statistics
    semantic_match_stats = {
        'total_calls': 0,
        'total_comparisons': 0,
        'matches_found': 0,
        'fields_checked': []  # Track which fields trigger semantic checks
    }
    
    SKIP_FIELDS = {
        "token.input_token", 
        "token.output_token", 
        "token.cached_token",
        "lead_repo.agency_phone_number"
    }

    _identifier_cache = {}
    _none_item_counter = {}

    def remove_trailing_underscore(value: str) -> str:
        return value[:-1] if value.endswith("_") else value

    def safe_string(value, default=""):
        """Safely convert any value to string."""
        if value is None:
            return default
        
        if isinstance(value, str):
            return value.strip()
        
        return str(value)

    def normalize_model(model):
        """Normalize vehicle model by removing trim levels and extra characters."""
        import re
        model_str = safe_string(model).strip().upper()
        
        terms_to_remove = [
            r'\s*4\s*DOOR\s*(WAGON|SEDAN|SUV|HATCHBACK|COUPE)?',
            r'\s*2\s*DOOR\s*(WAGON|SEDAN|SUV|HATCHBACK|COUPE)?',
            r'\s*4DR\s*(WAGON|SEDAN|SUV|HATCHBACK|COUPE)?',
            r'\s*2DR\s*(WAGON|SEDAN|SUV|HATCHBACK|COUPE)?',
            r'\s*4D\s*(WAGON|SEDAN|SUV|HATCHBACK|COUPE)?',
            r'\s*2D\s*(WAGON|SEDAN|SUV|HATCHBACK|COUPE)?',
            r'\s+(WAGON|SEDAN|SUV|HATCHBACK|COUPE)$',
            r'\s+4$',
            r'\s+2$'
        ]
        
        for term in terms_to_remove:
            model_str = re.sub(term, '', model_str, flags=re.IGNORECASE)
        
        return model_str.strip()
    
    def normalize(value: Any) -> str:
        """Normalize values for comparison"""
        if not isinstance(value, str) or not value.strip():
            return ""
        
        # Normalize location identifiers by removing common prefixes
        normalized = value.strip().lower()
        
        # Remove common location prefixes that might be inconsistently added
        location_prefixes = [
            r'^at\s+',           # "At Loc 1" -> "Loc 1"
            r'^location\s+',     # "Location 1" -> "1"
            r'^loc\s+',          # "Loc 1" -> "1" (but we want to keep this for matching)
            r'^site\s+',         # "Site 1" -> "1"
            r'^project\s+',      # "Project 1" -> "1"
        ]
        
        # For location identifiers, remove "At" prefix but keep "Loc" as it's part of the identifier
        if re.match(r'^at\s+loc\s+', normalized):
            normalized = re.sub(r'^at\s+', '', normalized)
        
        # Normalize addresses: remove extra spaces, normalize punctuation
        # Replace multiple spaces with single space, then replace space with underscore
        normalized = re.sub(r'\s+', ' ', normalized)
        # Normalize common address variations
        normalized = normalized.replace(',', '')  # Remove commas
        normalized = normalized.replace('.', '')  # Remove periods
        normalized = normalized.replace('  ', ' ')  # Remove double spaces
        
        return normalized.replace(" ", "_")

    def clean_vehicle_name(name: str) -> str:
        """Remove digits and extra spaces from vehicle name"""
        name = re.sub(r'\d+', '', name)
        name = re.sub(r'\s+', ' ', name)
        return normalize(name)

    def get_name_key(name_dict: Union[Dict, str]) -> str:
        """Extract name key from name dictionary or string"""
        if isinstance(name_dict, str):
            return normalize(name_dict)
        full_name_parts = [
            normalize(name_dict.get("first_name") or name_dict.get("name") or ""),
            normalize(name_dict.get("middle_name") or ""),
            normalize(name_dict.get("last_name") or "")
            ]
        return "_".join(filter(None, full_name_parts)) or "generic"

    def get_identifier_value(field_path: str, item: Union[Dict, str]) -> Optional[str]:
        """
        Extract the identifier value from an item based on field path.
        Returns the identifier string or None if not applicable.
        This is used for fuzzy matching similar identifiers.
        
        For list-based fields like cyber_liability.deductibles, this extracts
        the identifier (e.g., claim_identifier) from each object in the list
        to match objects across policy/binder/quote.
        
        Based on the schema, handles all identifier types:
        - Property/Location: property_address, location_address, rental_unit_address, unit_number, building_identifier, region_identifier, premises_identifier
        - Vehicle: vehicle_identification_number_vin, vehicle_name
        - Project/Location: project_or_location_identifier, insuring_agreement_identifier, covered_item_identifier
        - Claim: claim_identifier
        - Scheduled items: scheduled_item_id
        - Coverage: coverage_identifier
        - Driver: driver_name, drivers_licence_number
        - Dependency: dependency_identifier
        """
        if not isinstance(item, dict):
            return None
        
        identifier = None
        
        # Order matters: Check most specific patterns first, then general ones
        
        # 0. Location deductibles - MUST be checked FIRST before generic "deductibles" checks
        if field_path.endswith("location_deductibles"):
            identifier = (
                item.get("location_address") or
                item.get("property_address") or
                None
            )
            if not identifier:
                logger.warning(f"[get_identifier_value] location_deductibles: no location_address or property_address found in item keys: {list(item.keys())}")
        
        # ============================================================
        # NEW: location_classification_table and nested structures
        # ============================================================
        
        # 1a. location_classification_table (top level) - uses location_identifier
        elif field_path.endswith("location_classification_table"):
            identifier = item.get("location_identifier") or item.get("location_name") or None
        
        # 1b. classification_details (nested inside location_classification_table) - uses class_code
        elif "location_classification_table" in field_path and field_path.endswith("classification_details"):
            identifier = item.get("class_code") or item.get("classification_description") or None
        
        # 1c. premium_breakdown (nested inside classification_details) - uses basis
        elif "classification_details" in field_path and field_path.endswith("premium_breakdown"):
            identifier = item.get("basis") or item.get("premium_basis") or None
        
        # 1. Claim-based identifiers (cyber_liability, e_and_o, fiduciary_liability, d_and_o) - Most specific
        elif field_path.endswith((
            "e_and_o.additional_coverages_extensions.available_extensions",
            "fiduciary_liability.additional_coverages_extensions",
            "fiduciary_liability.coverage_overview.claim_coverages",
            "fiduciary_liability.limits_and_sublimits.claim_limits",
            "fiduciary_liability.deductibles",
            "e_and_o.deductibles", "e_and_o.limits_and_sublimits.claim_limits",
            "e_and_o.coverage_overview.claim_specific_coverages",
            "cyber_liability.coverage_overview.claim_coverages",
            "cyber_liability.deductibles",
            "cyber_liability.limits_and_sublimits.claim_limits",
            "d_and_o.coverage_overview.coverage_details",
            "d_and_o.limits_sublimits.policy_limits",
            "d_and_o.deductibles.policy_deductibles",
            "d_and_o.additional_coverages_extensions.coverage_extensions"
        )):
            identifier = item.get("claim_identifier") or item.get("coverage_identifier") or None
        
        # 2. Vehicle-based identifiers
        # IMPORTANT: Check for vehicle-specific paths first, exclude location_deductibles
        elif (field_path.endswith((
            "vehicle_details", "vehicles", "vehicle_coverages", "vehicle_limits",
            "vehicle_extensions", "vehicle_endorsements",
            "commercial_auto.vehicle_coverages",
            "commercial_auto.vehicle_limits"
        )) or ("personal_auto" in field_path.lower() and "vehicle" in field_path.lower())
        or (field_path.endswith("deductibles") and ("vehicle" in field_path.lower() or "auto" in field_path.lower()))
        or field_path.endswith(("commercial_auto.deductibles", "personal_auto.deductibles"))):
            identifier = (
                item.get("vin") or
                item.get("vehicle_identification_number_vin") or
                item.get("vehicle_name") or None
            )
        
        # 3. Driver-based identifiers
        elif field_path.endswith(("driver_details", "co_insured", "drivers")):
            # Try driver_name first, then drivers_licence_number
            name_dict = item.get("name", {}) if "name" in item else item
            if isinstance(name_dict, dict):
                identifier = get_name_key(name_dict)
            if not identifier:
                identifier = item.get("drivers_licence_number") or None
        
        # 4. Insured names
        elif field_path.endswith(("insured_names", "named_additional_insureds", "additional_insureds")):
            identifier = get_name_key(item)
        
        # 5. Scheduled items (homeowners)
        elif field_path.endswith(("scheduled_item_limits", "scheduled_limits", "scheduled_item_coverages")):
            identifier = item.get("scheduled_item_id") or item.get("item_identifier") or None
        
        # 6. Business Interruption - premises and dependencies
        elif "business_interruption" in field_path.lower():
            identifier = (
                item.get("premises_identifier") or
                item.get("dependency_identifier") or
                item.get("utility_type") or  # For utility_services_coverages
                None
            )
        
        # 7. Unit-based identifiers (condominium, tenant_renters)
        elif field_path.endswith((
            "unit_coverages", "unit_limits", "unit_deductibles", "unit_endorsements",
            "rental_value_coverages", "dollar_deductibles", "rental_units", "condo_units"
        )):
            identifier = (
                item.get("unit_number") or
                item.get("rental_unit_address") or
                item.get("property_address") or
                item.get("condo_unit_address") or
                None
            )
        
        # 8. Equipment Breakdown - covered item identifiers
        elif "equipment_breakdown" in field_path.lower():
            identifier = (
                item.get("covered_item_identifier") or
                item.get("equipment_identifier") or
                item.get("item_identifier") or
                None
            )
        
        # 8b. Inland Marine - comprehensive identifier handling
        elif "inland_marine" in field_path.lower() or field_path.endswith((
            "item_deductibles", "transit_deductibles", "occurrence_deductibles",
            "shipment_coverages", "transit_limits", "conveyance_coverages",
            "off_premises_location_coverages", "scheduled_item_coverages",
            "installation_project_coverages", "exhibition_event_coverages",
            "sublimited_coverages", "occurrence_based_coverages",
            "aggregate_limits", "item_limits", "occurrence_limits",
            "transit_limits", "off_premises_limits", "fine_arts_sublimits",
            "installation_project_limits", "general_sublimits",
            "time_limited_coverages", "sublimited_extensions",
            "general_additional_coverages", "automatic_item_category_coverages",
            "transit_extensions"
        )):
            identifier = (
                item.get("item_identifier") or
                item.get("shipment_identifier") or
                item.get("conveyance_identifier") or
                item.get("covered_item_identifier") or
                item.get("location_identifier") or
                item.get("project_identifier") or
                item.get("event_identifier") or
                item.get("sublimit_identifier") or
                item.get("occurrence_identifier") or
                item.get("item_or_collection_identifier") or
                item.get("limit_type_identifier") or
                item.get("item_category_identifier") or
                item.get("coverage_identifier") or
                None
            )
        
        # 9. Cargo Insurance - conveyance identifiers
        elif "cargo_insurance" in field_path.lower():
            identifier = (
                item.get("conveyance_identifier") or
                item.get("shipment_identifier") or
                item.get("coverage_identifier") or
                None
            )
        
        # 10. Builders Risk - project identifiers
        elif "builders_risk" in field_path.lower():
            identifier = (
                item.get("project_identifier") or
                item.get("project_or_location_identifier") or
                item.get("building_identifier") or
                None
            )
        
        # 11. Crime - insuring agreement identifiers
        elif "crime" in field_path.lower():
            identifier = (
                item.get("insuring_agreement_identifier") or
                item.get("coverage_identifier") or
                None
            )
        
        # 13. Umbrella - underlying policy identifiers
        elif "umbrella" in field_path.lower():
            identifier = (
                item.get("underlying_policy_number") or
                item.get("coverage_identifier") or
                item.get("coverage_type") or  # For repeating_coverages
                None
            )
        
        # 14. Property/Location-based identifiers (homeowners, tenant_renters, condominium, property)
        # Note: location_deductibles is already handled above (check #0)
        elif field_path.endswith((
            "location_coverages", "location_limits",
            "location_specific_deductibles", "location_specific_endorsements",
            "location_specific_coverages", "location_specific_limits",
            "region_coverages", "region_limits", "properties", "condo_buildings"
        )) or ("property" in field_path.lower() and ("location" in field_path.lower() or "address" in field_path.lower())):
            identifier = (
                item.get("property_address") or
                item.get("location_address") or
                item.get("rental_unit_address") or
                item.get("unit_number") or
                item.get("building_identifier") or
                item.get("region_identifier") or
                item.get("premises_identifier") or
                item.get("dependency_identifier") or
                item.get("location_details") or  # For business_interruption scheduled_premises
                None
            )
        
        # 15. General Liability - project/location identifiers
        elif "general_liability" in field_path.lower() or field_path.endswith((
            "project_location_coverages", "project_location_limits",
            "project_location_deductibles", "project_location_additional_coverages"
        )):
            identifier = (
                item.get("project_or_location_identifier") or
                item.get("underlying_policy_number") or
                item.get("coverage_identifier") or
                None
            )
        
        # 16. General fallback for limits_and_sublimits, deductibles, coverage_overview
        elif field_path.endswith((
            "limits_and_sublimits", "deductibles", "coverage_overview",
            "limits_sublimits", "additional_coverages_extensions"
        )):
            # Try common identifier fields in order of likelihood
            identifier = (
                item.get("project_or_location_identifier") or
                item.get("underlying_policy_number") or
                item.get("claim_identifier") or
                item.get("coverage_identifier") or
                item.get("insuring_agreement_identifier") or
                item.get("covered_item_identifier") or
                item.get("item_identifier") or
                item.get("premises_identifier") or
                item.get("dependency_identifier") or
                item.get("shipment_identifier") or
                item.get("conveyance_identifier") or
                item.get("project_identifier") or
                item.get("location_identifier") or
                item.get("occurrence_identifier") or
                item.get("sublimit_identifier") or
                None
            )
        
        # 17. Forms and endorsements - use forms_or_endorsements_number
        elif field_path.endswith("forms_and_endorsements"):
            identifier = item.get("forms_or_endorsements_number") or None
        
        # ============================================================
        # Schema-specific list-of-dict fields (using first field as identifier)
        # ============================================================
        
        # 18. Commercial Property - premises_info_table
        elif field_path.endswith("premises_info_table"):
            identifier = item.get("premise_identifier") or None
        
        # 19. Commercial Property - mortgage_details
        elif field_path.endswith("mortgage_details"):
            identifier = item.get("mortgagee_address") or item.get("mortgagee_name") or None
        
        # 20. Workers Compensation - locations
        elif field_path.endswith("workers_compensation.locations") or (
            "workers_compensation" in field_path.lower() and field_path.endswith("locations")
        ):
            identifier = item.get("location_identifier") or item.get("location_address") or None
        
        # 21. Workers Compensation - hazards
        elif field_path.endswith("hazards"):
            identifier = item.get("location_identifier") or None

        # 21a. Workers Compensation - hazard_details
        elif field_path.endswith("hazard_details"):
            identifier = item.get("class_code") or item.get("description_of_duties_performed") or None
        
        # 22. Workers Compensation - individuals_inclusion_exclusion
        elif field_path.endswith("individuals_inclusion_exclusion"):
            identifier = item.get("location_identifier") or item.get("owner_name") or None
        
        # 23. Homeowners - co_applicant_info
        elif field_path.endswith("co_applicant_info"):
            identifier = item.get("co_applicant_name") or None
        
        # 24. Inland Marine - scheduled_equipment_info
        elif field_path.endswith("scheduled_equipment_info"):
            identifier = item.get("equipment_name") or item.get("serial_number") or None
        
        # 25. Inland Marine - unscheduled_equipment_info
        elif field_path.endswith("unscheduled_equipment_info"):
            identifier = item.get("equipment_name") or None
        
        # 26. Inland Marine - limits_deductibles_table
        elif field_path.endswith("limits_deductibles_table"):
            identifier = item.get("equipment_name") or item.get("item_location_identifier") or None
        
        # 27. Commercial Auto - vehicles
        elif field_path.endswith("commercial_auto.vehicles") or (
            "commercial_auto" in field_path.lower() and field_path.endswith("vehicles")
        ):
            identifier = item.get("vin") or item.get("vehicle_identification_number_vin") or None
        
        # 28. Commercial Auto - drivers
        elif field_path.endswith("commercial_auto.drivers") or (
            "commercial_auto" in field_path.lower() and field_path.endswith("drivers")
        ):
            identifier = item.get("driver_name") or item.get("license_number") or None
        
        # 29. Excess Liability - underlying_insurance_schedule
        elif field_path.endswith("underlying_insurance_schedule"):
            identifier = item.get("policy_number") or item.get("coverage_type") or None

        # 30. Schedules - insured_locations
        elif "schedules" in field_path and field_path.endswith("insured_locations"):
            identifier = item.get("location_name") or None

        # 31. Schedules - locations
        elif "schedules" in field_path and field_path.endswith("locations"):
            identifier = item.get("Location_identifier") or item.get("Location_Address") or None

        # 32. Schedules - drivers
        elif "schedules" in field_path and field_path.endswith("drivers"):
            identifier = item.get("driver_name") or item.get("driver_license_number") or None
        
        # ============================================================
        # FALLBACK: Use first field value as identifier
        # ============================================================
        if identifier is None and isinstance(item, dict) and item:
            # Get the first key in the dictionary
            first_key = next(iter(item.keys()), None)
            if first_key:
                first_value = item.get(first_key)
                if first_value is not None and first_value != "" and not isinstance(first_value, (list, dict)):
                    identifier = str(first_value).strip()

        if identifier and isinstance(identifier, str) and identifier.strip():
            return identifier.strip()

        return None

    def get_unique_key(field_path: str, item: Union[Dict, str], parent_item: Optional[Dict] = None) -> str:
        """
        Generate consistent UUID for array items based on identifier fields.
        Enhanced to handle all policy types and nested structures.
        """
        # Handle non-dict items
        if not isinstance(item, dict):
            if isinstance(item, str):
                if field_path.endswith(("mortgagees_lienholders", "insured_names")) and parent_item:
                    address = normalize(
                        parent_item.get("property_address") or 
                        parent_item.get("condo_unit_address") or 
                        parent_item.get("unit_number") or ""
                    )
                    key = address or normalize(item)
                else:
                    key = normalize(item)
            else:
                logger.error(f"Invalid item type for field_path {field_path}: {type(item)}")
                key = "generic"
            
            cache_key = f"{field_path}:{key}" if key else f"{field_path}:generic"
            if cache_key not in _identifier_cache:
                _identifier_cache[cache_key] = str(uuid.uuid4())
            return _identifier_cache[cache_key]

        try:
            key = None

            # ============================================================
            # PRIORITY 1: location_classification_table and nested structures
            # These MUST be checked FIRST before any other patterns
            # ============================================================
            
            # location_classification_table - use location_identifier
            if field_path.endswith("location_classification_table"):
                identifier = item.get("location_identifier") or item.get("location_name") or ""
                key = normalize(identifier) if identifier else None

            # classification_details - use class_code (MUST check before generic patterns)
            elif "location_classification_table" in field_path and "classification_details" in field_path and "premium_breakdown" not in field_path:
                identifier = item.get("class_code") or item.get("classification_description") or ""
                key = normalize(identifier) if identifier else None

            # premium_breakdown - use basis (MUST check before generic patterns)
            elif "classification_details" in field_path and "premium_breakdown" in field_path:
                identifier = item.get("basis") or ""
                key = normalize(identifier) if identifier else None

            # hazards - use location_identifier (Priority 1)
            elif field_path.endswith("hazards"):
                identifier = item.get("location_identifier") or ""
                key = normalize(identifier) if identifier else None

            # hazard_details - use class_code or description
            elif "hazards" in field_path and "hazard_details" in field_path:
                identifier = item.get("class_code") or item.get("description_of_duties_performed") or ""
                key = normalize(identifier) if identifier else None

            # schedules.insured_locations
            elif "schedules" in field_path and field_path.endswith("insured_locations"):
                identifier = item.get("location_name") or ""
                key = normalize(identifier) if identifier else None

            # schedules.locations
            elif "schedules" in field_path and field_path.endswith("locations"):
                identifier = item.get("Location_identifier") or item.get("Location_Address") or ""
                key = normalize(identifier) if identifier else None

            # schedules.drivers
            elif "schedules" in field_path and field_path.endswith("drivers"):
                identifier = item.get("driver_name") or item.get("driver_license_number") or ""
                key = normalize(identifier) if identifier else None
            
            # ============================================================
            # PRIORITY 2: Other specific field patterns
            # ============================================================
            
            elif field_path.endswith((
                "project_location_coverages", "project_location_limits",
                "project_location_deductibles", "project_location_additional_coverages",
                "limits_and_sublimits", "coverage_overview",
                "crime.additional_coverages_extensions",
                "equipment_breakdown.limits_sublimits", "equipment_breakdown.deductibles",
                "equipment_breakdown.additional_coverages_extensions"
            )):
                identifier = (
                    item.get("underlying_policy_number") or 
                    item.get("project_or_location_identifier") or 
                    item.get("insuring_agreement_identifier") or 
                    item.get("covered_item_identifier") or ""
                )
                key = normalize(identifier) if identifier else None

            elif field_path.endswith(("driver_details", "co_insured", "drivers")):
                name_dict = item.get("driver_name", {}) if "driver_name" in item else item
                name_key = get_name_key(name_dict)
                if name_key == "generic":
                    _none_item_counter[field_path] = _none_item_counter.get(field_path, 0) + 1
                    key = f"none_driver_{_none_item_counter[field_path]}"
                else:
                    key = name_key

            elif field_path.endswith(("insured_names", "named_additional_insureds", "additional_insureds")):
                name_key = get_name_key(item)
                if name_key == "generic":
                    _none_item_counter[field_path] = _none_item_counter.get(field_path, 0) + 1
                    key = f"none_insured_{_none_item_counter[field_path]}"
                else:
                    key = name_key
            
            # NOTE: Removed "deductibles" from this check to avoid conflicts with classification_details
            elif field_path.endswith((
                "vehicle_details", "vehicles", "vehicle_coverages", "vehicle_limits",
                "vehicle_extensions", "vehicle_endorsements",
                "commercial_auto.deductibles"
            )):
                vin_value = (
                    item.get("vin") or
                    item.get("vehicle_identification_number_vin") or ""
                )
                if vin_value:
                    key = normalize(vin_value)
                else:
                    year = item.get("year") or ""
                    make = item.get("make") or ""
                    model = item.get("model") or ""

                    if year and make and model:
                        composite_key = f"{year}_{make}_{model}"
                        key = normalize_model(normalize(composite_key))
                    else:
                        vehicle_name = item.get("vehicle_name") or ""
                        if vehicle_name:
                            key = normalize(vehicle_name)
                        else:
                            _none_item_counter[field_path] = _none_item_counter.get(field_path, 0) + 1
                            key = f"none_vehicle_{_none_item_counter[field_path]}"

            elif field_path.endswith((
                "e_and_o.additional_coverages_extensions.available_extensions",
                "fiduciary_liability.additional_coverages_extensions",
                "fiduciary_liability.coverage_overview.claim_coverages",
                "fiduciary_liability.limits_and_sublimits.claim_limits",
                "fiduciary_liability.deductibles",
                "e_and_o.deductibles", "e_and_o.limits_and_sublimits.claim_limits",
                "e_and_o.coverage_overview.claim_specific_coverages",
                "cyber_liability.coverage_overview.claim_coverages",
                "cyber_liability.deductibles",
                "cyber_liability.limits_and_sublimits.claim_limits"
            )):
                identifier = item.get("claim_identifier") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith((
                "d_and_o.coverage_overview.coverage_details",
                "d_and_o.limits_sublimits.policy_limits",
                "d_and_o.deductibles.policy_deductibles",
                "d_and_o.additional_coverages_extensions.coverage_extensions",
                "cyber_liability.additional_coverages_extensions.policy_extensions"
            )):
                identifier = (
                    item.get("coverage_identifier") or 
                    item.get("limit_identifier") or 
                    item.get("deductible_identifier") or 
                    item.get("extension_identifier") or ""
                )
                key = normalize(identifier) if identifier else None

            elif field_path.endswith((
                "inland_marine.coverage_overview.shipment_coverages",
                "inland_marine.limits_sublimits.transit_limits",
                "inland_marine.deductibles.transit_deductibles",
                "inland_marine.additional_coverages_extensions.transit_extensions"
            )):
                identifier = item.get("shipment_identifier") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith((
                "inland_marine.coverage_overview.scheduled_item_coverages",
                "inland_marine.limits_sublimits.item_limits",
                "inland_marine.limits_sublimits.fine_arts_sublimits",
                "inland_marine.deductibles.item_deductibles"
            )):
                identifier = (
                    item.get("item_identifier") or 
                    item.get("item_or_collection_identifier") or ""
                )
                key = normalize(identifier) if identifier else None
            
            elif field_path.endswith((
                "inland_marine.coverage_overview.installation_project_coverages",
                "inland_marine.limits_sublimits.installation_project_limits",
                "builders_risk.limits_and_sublimits",
                "builders_risk.deductibles",
                "builders_risk.additional_coverages_extensions"
            )):
                identifier = item.get("project_identifier") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("inland_marine.coverage_overview.exhibition_event_coverages"):
                identifier = item.get("event_identifier") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith((
                "inland_marine.coverage_overview.sublimited_coverages",
                "inland_marine.limits_sublimits.general_sublimits",
                "inland_marine.additional_coverages_extensions.sublimited_extensions"
            )):
                identifier = item.get("sublimit_identifier") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith((
                "inland_marine.coverage_overview.occurrence_based_coverages",
                "inland_marine.limits_sublimits.occurrence_limits",
                "inland_marine.deductibles.occurrence_deductibles"
            )):
                identifier = item.get("occurrence_identifier") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("inland_marine.limits_sublimits.aggregate_limits"):
                identifier = item.get("limit_type_identifier") or ""
                key = normalize(identifier) if identifier else None
            
            elif field_path.endswith((
                "inland_marine.limits_sublimits.off_premises_limits",
                "inland_marine.coverage_overview.off_premises_location_coverages"
            )):
                identifier = item.get("location_identifier") or ""
                key = normalize(identifier) if identifier else None
            
            elif field_path.endswith((
                "inland_marine.additional_coverages_extensions.time_limited_coverages",
                "inland_marine.additional_coverages_extensions.general_additional_coverages"
            )):
                identifier = item.get("coverage_identifier") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("inland_marine.additional_coverages_extensions.automatic_item_category_coverages"):
                identifier = item.get("item_category_identifier") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith((
                "cargo_insurance.additional_coverages_extensions",
                "builders_risk.additional_coverages_extensions",
                "epli.coverage_overview.claim_coverages"
            )):
                identifier = next((item.get(k) for k in item if item.get(k) not in (None, 'Not-Present', '')), "") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("builders_risk.coverage_overview.conveyance_coverages"):
                identifier = item.get("transit_types_covered") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("builders_risk.coverage_overview.shipment_coverages"):
                identifier = item.get("property_insured") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("builders_risk.coverage_overview.storage_coverages"):
                identifier = item.get("storage_coverage") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("builders_risk.coverage_overview.special_coverages_and_endorsements"):
                identifier = item.get("refrigeration_breakdown_coverage") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("cargo_insurance.coverage_overview.conveyance_coverages"):
                identifier = item.get("conveyance_identification_number") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("cargo_insurance.coverage_overview.shipment_coverages"):
                identifier = item.get("shipment_identification_number") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("cargo_insurance.coverage_overview.storage_location_coverages"):
                identifier = item.get("storage_location_identifier") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("cargo_insurance.coverage_overview.special_endorsement_coverages"):
                identifier = item.get("refrigeration_breakdown_coverage") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("cargo_insurance.limits_and_sublimits.shipment_limits"):
                identifier = item.get("shipment_identification_number") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("cargo_insurance.limits_and_sublimits.conveyance_limits"):
                identifier = item.get("conveyance_identification_number") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("cargo_insurance.limits_and_sublimits.policy_level_sublimits"):
                identifier = item.get("aggregate_limit") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith(("cargo_insurance.deductibles", "builders_risk.deductibles")):
                identifier = item.get("per_shipment_deductible") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("mortgage_details"):
                vehicle_name = item.get("mortgage_vehicle_name") or ""
                if vehicle_name:
                    key = f"vehicle:{clean_vehicle_name(vehicle_name)}"
                else:
                    company = normalize(item.get('mortgage_company') or '')
                    if company:
                        key = f"company:{company}"
                    else:
                        _none_item_counter[field_path] = _none_item_counter.get(field_path, 0) + 1
                        key = f"none_mortgage_{_none_item_counter[field_path]}"

            elif field_path.endswith("claim_details"):
                date = normalize(item.get("date_of_loss") or "")
                if date:
                    key = date
                else:
                    _none_item_counter[field_path] = _none_item_counter.get(field_path, 0) + 1
                    key = f"none_claim_{_none_item_counter[field_path]}"

            elif field_path.endswith("dependent_properties_coverages"):
                key = normalize(item.get("dependency_identifier") or "") or None

            elif field_path.endswith((
                "location_coverages", "location_limits", "location_deductibles",
                "property_extensions", "property_endorsements",
                "location_specific_coverages", "location_specific_limits",
                "location_specific_deductibles", "location_specific_endorsements"
            )):
                address_value = (
                    item.get("property_address") or 
                    item.get("rental_unit_address") or 
                    item.get("location_address") or 
                    item.get("location_identifier") or ""
                )
                if address_value:
                    key = normalize(address_value)
                else:
                    _none_item_counter[field_path] = _none_item_counter.get(field_path, 0) + 1
                    key = f"none_address_{_none_item_counter[field_path]}"

            elif field_path.endswith(("region_coverages", "region_limits", "region_endorsements")):
                key = normalize(item.get("region_identifier") or "") or None
            
            elif field_path.endswith((
                "scheduled_item_limits", "scheduled_item_endorsements",
                "scheduled_personal_property", "scheduled_personal_property_coverages"
            )):
                key = normalize(
                    item.get("scheduled_item_id") or 
                    item.get("scheduled_item_identifier") or ""
                ) or None
            
            elif field_path.endswith((
                "properties", "rental_units", "condo_units", 
                "condo_buildings", "mortgagees_lienholders"
            )):
                property_value = (
                    item.get("property_address") or 
                    item.get("rental_unit_address") or 
                    item.get("condo_unit_address") or 
                    item.get("building_identifier") or 
                    item.get("unit_number") or ""
                )
                if property_value:
                    key = normalize(property_value)
                else:
                    condo_corporation_name = item.get("condo_corporation_name") or ""
                    if condo_corporation_name:
                        key = normalize(condo_corporation_name)
                    else:
                        _none_item_counter[field_path] = _none_item_counter.get(field_path, 0) + 1
                        key = f"none_property_{_none_item_counter[field_path]}"

            elif field_path.endswith("mortgagees_lienholders"):
                if isinstance(item, dict):
                    key = normalize(
                        item.get("unit_number") or 
                        item.get("mortgagee_info") or ""
                    ) or None
                elif parent_item:
                    key = normalize(
                        parent_item.get("property_address") or 
                        parent_item.get("condo_unit_address") or ""
                    ) or normalize(item) or None
                else:
                    key = None

            elif field_path.endswith(("unit_coverages", "unit_limits", "unit_deductibles", "unit_endorsements")):
                key = normalize(item.get("unit_number") or "") or None

            elif field_path.endswith("underlying_policies"):
                key = normalize(item.get("underlying_policy_number") or "") or None

            elif field_path.endswith(("buildings", "building_limits")):
                key = normalize(item.get("building_identifier") or "") or None

            elif field_path.endswith("locations"):
                key = normalize(item.get("location_address") or "") or None

            elif field_path.endswith((
                "scheduled_premises", "business_income_limits", "extra_expense_limits",
                "ordinary_payroll_limits", "civil_authority_limits",
                "service_interruption_limits", "ingress_egress_limits",
                "rental_value_limits", "leasehold_interest_coverages",
                "professional_fees_sublimits", "extra_expense_coverages",
                "rental_value_coverages", "dollar_deductibles"
            )):
                key = normalize(item.get("premises_identifier") or "") or None
            
            elif field_path.endswith((
                "contingent_business_interruption_coverages",
                "contingent_business_interruption_limits"
            )):
                key = normalize(item.get("dependency_identifier") or "") or None

            elif field_path.endswith("utility_services_coverages"):
                key = normalize(item.get("utility_type") or "") or None

            elif field_path.endswith(("specific_claim_deductibles", "claim_deductibles")):
                identifier = item.get("claim_type") or item.get("deductible_type") or ""
                key = normalize(identifier) if identifier else None

            elif field_path.endswith("epli.additional_coverages_extensions.additional_coverages"):
                key = normalize(item.get("coverage_type") or "") or None

            elif field_path.endswith("high_risk_activity_limits"):
                key = normalize(item.get("activity_type") or "") or None
            
            elif field_path.endswith((
                "excess_liability_coverages", "excess_um_uim_coverages",
                "umbrella.limits_and_sublimits", "umbrella.deductibles",
                "umbrella.additional_coverages_extensions"
            )):
                key = normalize(item.get("underlying_policy_number") or "") or None

            elif field_path.endswith("special_limits"):
                key = normalize(item.get("item_category") or "") or None

            elif field_path.endswith("attached_forms_and_endorsements"):
                key = normalize(item.get("form_endorsement_type") or "") or None

            elif field_path.endswith(("scheduled_limits", "agreement_deductibles")):
                key = normalize(item.get("agreement_identifier") or "") or None

            elif field_path.endswith("separate_peril_limits"):
                peril = normalize(
                    item.get("separate_limit_for_certain_perils", "").split(" for ")[-1] 
                    if item.get("separate_limit_for_certain_perils") else ""
                ) or ""
                key = peril if peril else None

            elif field_path.endswith((
                "valuable_papers_and_records", "accounts_receivable",
                "newly_acquired_constructed_property", "outdoor_signs_limit",
                "transit_coverage_limit", "off_premises_property_limit",
                "Earthquake", "Flood", "fire_department_service_charge",
                "pollutant_clean_up_and_removal", "increased_cost_of_construction",
                "utility_services_direct_damage", "utility_services_time_element"
            )):
                key = normalize(item) if isinstance(item, str) else None

            elif field_path.endswith("umbrella.coverage_overview.repeating_coverages"):
                key = normalize(item.get("underlying_policy_number") or "") or None
            
            # ============================================================
            # Handle generic "deductibles" that didn't match specific patterns above
            # ============================================================
            elif field_path.endswith("deductibles"):
                # Try common identifier fields
                identifier = (
                    item.get("underlying_policy_number") or 
                    item.get("project_or_location_identifier") or 
                    item.get("insuring_agreement_identifier") or 
                    item.get("covered_item_identifier") or 
                    item.get("claim_identifier") or
                    item.get("deductible_type") or
                    ""
                )
                key = normalize(identifier) if identifier else None

            # ============================================================
            # FALLBACK: Use first non-empty, non-list, non-dict field
            # ============================================================
            if key is None:
                if item:
                    # Skip list and dict values when finding first key
                    for k, v in item.items():
                        if v and not isinstance(v, (list, dict)) and str(v).strip():
                            key = normalize(str(v))
                            break
                    if not key:
                        key = None
            
            if not key or key == "generic":
                non_empty_values = [
                    normalize(str(item.get(k))) 
                    for k in sorted(item.keys()) 
                    if item.get(k) and not isinstance(item.get(k), (list, dict)) and normalize(str(item.get(k)))
                ]
                if non_empty_values:
                    key = "_".join(non_empty_values)
                else:
                    _none_item_counter[field_path] = _none_item_counter.get(field_path, 0) + 1
                    key = f"generic_{_none_item_counter[field_path]}"

            # Final check for generic key
            if not key or key == "generic":
                _none_item_counter[field_path] = _none_item_counter.get(field_path, 0) + 1
                key = f"generic_{_none_item_counter[field_path]}"
            
            # Generate/cache UUID
            cache_key = f"{field_path}:{remove_trailing_underscore(key)}" if key else f"{field_path}:generic"
            if cache_key not in _identifier_cache:
                _identifier_cache[cache_key] = str(uuid.uuid4())

            return _identifier_cache[cache_key]

        except Exception as e:
            logger.error(f"Error in get_unique_key for {field_path}: {e}")
            _none_item_counter[field_path] = _none_item_counter.get(field_path, 0) + 1
            cache_key = f"{field_path}:generic_{_none_item_counter[field_path]}"
            if cache_key not in _identifier_cache:
                _identifier_cache[cache_key] = str(uuid.uuid4())
            return _identifier_cache[cache_key]
            
    def is_complex_object(obj: Any, path: str) -> bool:
        """Check if object should be excluded from direct comparison (container fields)."""
        if not isinstance(obj, dict):
            return False
        
        # Check if this object contains lists of dictionaries (container pattern)
        has_list_of_dicts = any(
            isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict) 
            for v in obj.values()
        )
        
        # Check if this object contains nested dictionaries (container pattern)
        has_nested_dicts = any(
            isinstance(v, dict) and len(v) > 0 
            for v in obj.values()
        )
        
        # Check if this object contains lists (potential container)
        has_lists = any(isinstance(v, list) and len(v) > 0 for v in obj.values())
        
        # If it has lists of dicts or nested dicts, it's definitely a container
        if has_list_of_dicts or has_nested_dicts:
            return True
        
        # If it has many keys and contains lists, likely a container
        complex_threshold = 3
        is_complex = len(obj.keys()) > complex_threshold and has_lists
        
        # Known container patterns
        container_patterns = [
            "coverage_overview", "limits_and_sublimits", "deductibles", 
            "additional_coverages_extensions", "policy_information",
            "separate_limit_for_certain_perils", "claim_coverages",
            "policy_limits", "policy_deductibles", "coverage_extensions",
            "policy_extensions", "claim_specific_coverages", "claim_limits",
            "claim_deductibles", "available_extensions", "claim_coverages",
            "shipment_coverages", "transit_limits", "transit_deductibles",
            "transit_extensions", "scheduled_item_coverages", "item_limits",
            "fine_arts_sublimits", "item_deductibles", "installation_project_coverages",
            "installation_project_limits", "exhibition_event_coverages",
            "event_identifier", "sublimited_coverages", "general_sublimits",
            "sublimited_extensions", "occurrence_based_coverages", "occurrence_limits",
            "occurrence_deductibles", "aggregate_limits", "off_premises_limits",
            "off_premises_location_coverages", "time_limited_coverages",
            "general_additional_coverages", "automatic_item_category_coverages",
            "repeating_coverages", "conveyance_coverages", "shipment_coverages",
            "storage_location_coverages", "special_endorsement_coverages",
            "shipment_limits", "conveyance_limits", "policy_level_sublimits",
            "contingent_business_interruption_coverages", "contingent_business_interruption_limits",
            "utility_services_coverages", "specific_claim_deductibles", "claim_deductibles",
            "additional_coverages", "high_risk_activity_limits", "special_limits",
            "attached_forms_and_endorsements", "scheduled_limits", "agreement_deductibles",
            "separate_peril_limits"
        ]
        
        for pattern in container_patterns:
            if path.endswith(pattern):
                return True
        
        return is_complex
            
    def get_all_subfields(obj: Any, path: str, subfields: List[str], processed_paths: Set[str] = None, parent_item: dict = None) -> None:
        """
        Recursively collect ALL subfield paths from dictionary or list.
        Enhanced to ensure no field is missed.
        """
        if processed_paths is None:
            processed_paths = set()
        
        if path in processed_paths:
            return
        processed_paths.add(path)

        # Unwrap wrapper objects shaped as {"value": ..., "pages": [...]}
        if isinstance(obj, dict) and set(obj.keys()).issubset({"value", "pages"}):
            obj = obj.get("value")

        if isinstance(obj, dict):
            # Add current path if it's not a complex parent
            if path and not is_complex_object(obj, path):
                subfields.append(path)
                
            # Recurse into all dict keys
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                # Unwrap per-key wrapper without adding '.value' in the path
                if isinstance(value, dict) and set(value.keys()).issubset({"value", "pages"}):
                    inner_val = value.get("value")
                    if isinstance(inner_val, (dict, list)):
                        get_all_subfields(inner_val, new_path, subfields, processed_paths, obj)
                    else:
                        subfields.append(new_path)
                elif isinstance(value, (dict, list)):
                    get_all_subfields(value, new_path, subfields, processed_paths, obj)
                else:
                    # Leaf value
                    subfields.append(new_path)
        
        elif isinstance(obj, list):
            if not obj:  # Empty list
                if path:
                    subfields.append(path)
            else:
                for item in obj:
                    # Unwrap wrapped list items
                    if isinstance(item, dict) and set(item.keys()).issubset({"value", "pages"}):
                        inner_item = item.get("value")
                    else:
                        inner_item = item
                    # Skip generating keys for nested lists in mortgagees_lienholders if they are strings
                    if path.endswith("mortgagees_lienholders") and isinstance(inner_item, str):
                        continue
                    # Get unique key for this item
                    key = get_unique_key(path, inner_item, parent_item)
                    get_all_subfields(
                        inner_item, 
                        f"{path}[{key}]", 
                        subfields, 
                        processed_paths, 
                        inner_item if isinstance(inner_item, dict) else parent_item
                    )
        
        else:
            # Leaf value
            if path:
                subfields.append(path)

    def get_nested_value(d: Dict, path: List[str]) -> Any:
        """Access nested dictionary value using a list of keys"""
        current = d
        for key in path:
            try:
                current = current[key]
            except (KeyError, TypeError):
                return None
        return current

    def get_descriptive_field_name(field_path: str, *_) -> str:
        """Convert field path to descriptive name"""
        # Replace [content] with .content
        field_path = re.sub(r'\[([^\]]+)\]', r'.\1', field_path)

        # Split path
        segments = [seg for seg in field_path.split('.') if seg]

        formatted_segments = []
        for segment in segments:
            # If segment contains special characters, keep as-is
            if re.search(r'\s|\(|\)|:|-|\d', segment) or len(segment) > 6:
                formatted_segments.append(segment.strip())
            else:
                # Format snake_case or camelCase to words
                words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', segment.replace('_', ' '))
                formatted = ' '.join(word.capitalize() for word in words)
                formatted_segments.append(formatted)

        return ' ~ '.join(formatted_segments)

    def compare_comma_separated_values(val1: str, val2: str) -> bool:
        """Compare comma-separated values - only match if all parts match"""
        if not val1 or not val2:
            return val1 == val2
        
        # Split by comma and clean each part
        parts1 = [part.strip() for part in val1.split(',') if part.strip()]
        parts2 = [part.strip() for part in val2.split(',') if part.strip()]
        
        # Must have same number of parts and all parts must match
        if len(parts1) != len(parts2):
            return False
        
        # Check if all parts match (case-insensitive)
        return all(part1.lower() == part2.lower() for part1, part2 in zip(parts1, parts2))
    
    def is_numeric(value: Any) -> bool:
        """Check if a value is numeric (int or float)"""
        if isinstance(value, (int, float)):
            return True
        if isinstance(value, str):
            try:
                float(value)
                return True
            except ValueError:
                return False
        return False
    
    def extract_numeric_tokens(value: Any) -> List[float]:
        """Extract numeric tokens from any value as floats"""
        if value is None:
            return []
        text = str(value)
        # Remove commas to handle numbers like 25,000
        text = text.replace(',', '')
        tokens = re.findall(r'-?\d+(?:\.\d+)?', text)
        numbers: List[float] = []
        for tok in tokens:
            try:
                numbers.append(float(tok))
            except Exception:
                continue
        return numbers

    async def match_items_by_key(p_list: List, b_list: List, q_list: List, field_path: str, p_pages_ctx: List[int], b_pages_ctx: List[int], q_pages_ctx: List[int]):
        """Enhanced matching with 3-tier strategy: exact -> fuzzy -> semantic"""
        
        # Special handling for forms_and_endorsements
        if field_path.endswith("forms_and_endorsements"):
            def build_named_map(lst: List[Dict[str, Any]], pages_ctx: List[int]) -> tuple:
                named_map: Dict[str, List[str]] = {}
                named_pages: Dict[str, List[int]] = {}  # Track pages for each number
                unnamed: List[str] = []
                unnamed_pages: List[List[int]] = []  # Track pages for unnamed items
                for item in lst or []:
                    item_pages = pages_ctx or []
                    item_value = item
                    # Extract pages if wrapped
                    if isinstance(item, dict) and set(item.keys()).issubset({"value", "pages"}):
                        item_pages = item.get("pages") or pages_ctx or []
                        item_value = item.get("value", {})
                    
                    number = item_value.get("forms_or_endorsements_number", "") or "" if isinstance(item_value, dict) else ""
                    name = item_value.get("forms_or_endorsements_name", "") or "" if isinstance(item_value, dict) else ""
                    if number:
                        if number not in named_map:
                            named_map[number] = []
                            named_pages[number] = []
                        named_map[number].append(name)
                        # Aggregate pages for this number
                        named_pages[number].extend(item_pages)
                    else:
                        unnamed.append(name)
                        unnamed_pages.append(item_pages)
                return named_map, unnamed, named_pages, unnamed_pages

            p_named, p_unnamed, p_named_pages, p_unnamed_pages = build_named_map(p_list, p_pages_ctx)
            b_named, b_unnamed, b_named_pages, b_unnamed_pages = build_named_map(b_list, b_pages_ctx)
            q_named, q_unnamed, q_named_pages, q_unnamed_pages = build_named_map(q_list, q_pages_ctx)

            all_named_keys = set(p_named.keys()) | set(b_named.keys()) | set(q_named.keys())
            for key in sorted(all_named_keys):
                p_names = p_named.get(key)
                if p_names is not None:
                    p_val = ', '.join([n for n in p_names if n]) or ''
                    p_item_pages = sorted(list(set(p_named_pages.get(key, []))))  # Remove duplicates and sort
                else:
                    p_val = None
                    p_item_pages = []
                b_names = b_named.get(key)
                if b_names is not None:
                    b_val = ', '.join([n for n in b_names if n]) or ''
                    b_item_pages = sorted(list(set(b_named_pages.get(key, []))))  # Remove duplicates and sort
                else:
                    b_val = None
                    b_item_pages = []
                q_names = q_named.get(key)
                if q_names is not None:
                    q_val = ', '.join([n for n in q_names if n]) or ''
                    q_item_pages = sorted(list(set(q_named_pages.get(key, []))))  # Remove duplicates and sort
                else:
                    q_val = None
                    q_item_pages = []
                sub_path = f"{field_path}[{key}]"
                check_values(p_val, b_val, q_val, sub_path, p_item_pages, b_item_pages, q_item_pages)

            unnamed_sources = {
                "policy": p_unnamed,
                "binder": b_unnamed,
                "quote": q_unnamed
            }
            rep_to_key: Dict[str, str] = {}
            source_unnamed_maps = {"policy": {}, "binder": {}, "quote": {}}
            for src, names in unnamed_sources.items():
                for name in names:
                    norm = normalize(name)
                    if not norm:
                        key = str(uuid.uuid4())
                    else:
                        best_rep = None
                        best_sim = 0
                        for existing_rep in list(rep_to_key.keys()):
                            sim = fuzz.ratio(norm, existing_rep)
                            if sim > best_sim and sim >= similarity_threshold:
                                best_sim = sim
                                best_rep = existing_rep
                        
                        # Only call semantic if fuzzy matching failed (best_rep is None)
                        # if best_rep is None and rep_to_key:
                        #     # Batch process all comparisons for this name
                        #     comparisons = [(name, existing_rep) for existing_rep in list(rep_to_key.keys())]
                        #     if comparisons:
                        #         if field_path not in semantic_match_stats['fields_checked']:
                        #             semantic_match_stats['fields_checked'].append(field_path)
                        #         logger.info(f"[Field: {field_path}] Fuzzy match failed for forms_and_endorsements name '{name}', trying semantic check with {len(comparisons)} comparisons...")
                        #         batch_results = await check_semantic_similarity_batch(
                        #             comparisons, verification_provider, verification_model, semantic_token_tracker
                        #         )
                        #         for existing_rep in list(rep_to_key.keys()):
                        #             if batch_results.get((name, existing_rep), False):
                        #                 semantic_match_stats['matches_found'] += 1
                        #                 logger.info(f"[Field: {field_path}] Semantic match found: '{name}' matches '{existing_rep}'")
                        #                 best_rep = existing_rep
                        #                 break
                        
                        if best_rep is not None:
                            key = rep_to_key[best_rep]
                        else:
                            key = str(uuid.uuid4())
                            rep_to_key[norm] = key
                    source_unnamed_maps[src][key] = name

            all_unnamed_keys = set(source_unnamed_maps["policy"].keys()) | set(source_unnamed_maps["binder"].keys()) | set(source_unnamed_maps["quote"].keys())
            # Build pages mapping for unnamed items - match pages to keys based on name
            unnamed_pages_map = {"policy": {}, "binder": {}, "quote": {}}
            for src_idx, (src, names) in enumerate(unnamed_sources.items()):
                pages_list = [p_unnamed_pages, b_unnamed_pages, q_unnamed_pages][src_idx]
                for name_idx, name in enumerate(names):
                    if name_idx < len(pages_list):
                        # Find the key for this name in source_unnamed_maps
                        key = None
                        for k, v in source_unnamed_maps[src].items():
                            if v == name:
                                key = k
                                break
                        if key:
                            if key not in unnamed_pages_map[src]:
                                unnamed_pages_map[src][key] = []
                            unnamed_pages_map[src][key].extend(pages_list[name_idx])
            
            for key in sorted(all_unnamed_keys):
                p_val = source_unnamed_maps["policy"].get(key)
                b_val = source_unnamed_maps["binder"].get(key)
                q_val = source_unnamed_maps["quote"].get(key)
                p_item_pages = sorted(list(set(unnamed_pages_map["policy"].get(key, []))))  # Remove duplicates and sort
                b_item_pages = sorted(list(set(unnamed_pages_map["binder"].get(key, []))))  # Remove duplicates and sort
                q_item_pages = sorted(list(set(unnamed_pages_map["quote"].get(key, []))))  # Remove duplicates and sort
                sub_path = f"{field_path}[{key}]"
                check_values(p_val, b_val, q_val, sub_path, p_item_pages, b_item_pages, q_item_pages)
            return

        # Build identifier mapping with 3-tier matching
        identifier_to_key: Dict[str, str] = {}  # normalized_identifier -> UUID
        key_to_identifier: Dict[str, str] = {}  # UUID -> original_identifier
        
        # Track identifiers established by each document for fuzzy matching
        policy_ids: Set[str] = set()
        binder_ids: Set[str] = set()
        
        async def get_or_create_key(identifier: str, item: Any, fuzzy_candidates: Set[str]) -> str:
            """Get existing key or create new one using 3-tier matching"""
            if not identifier or not identifier.strip():
                return get_unique_key(field_path, item)
            
            identifier = identifier.strip()
            normalized = normalize(identifier)
            
            # Tier 1: Exact match (normalized) - always allowed globally to group identical items
            if normalized in identifier_to_key:
                existing_key = identifier_to_key[normalized]
                return existing_key

            # Tier 1b: Case-insensitive exact match on original identifiers - also allowed globally
            identifier_lower = identifier.lower()
            for existing_key, existing_id in key_to_identifier.items():
                if existing_id.lower() == identifier_lower:
                    # Add normalized mapping to ensure future lookups work
                    identifier_to_key[normalized] = existing_key
                    return existing_key
            
            # Tier 2: Fuzzy match - ONLY allowed against identifiers from OTHER documents
            if fuzzy_candidates:
                best_match_norm = None
                best_match_key = None
                best_similarity = 0
                for existing_norm in fuzzy_candidates:
                    if existing_norm not in identifier_to_key:
                        continue
                    existing_key = identifier_to_key[existing_norm]
                    similarity = fuzz.ratio(normalized, existing_norm)
                    if similarity >= similarity_threshold and similarity > best_similarity:
                        best_similarity = similarity
                        best_match_norm = existing_norm
                        best_match_key = existing_key
                
                if best_match_key:
                    # Add this identifier to the mapping so future exact matches work
                    identifier_to_key[normalized] = best_match_key
                    return best_match_key
            
            # Tier 3: Semantic match via GPT (batch processing) - ONLY allowed against identifiers from OTHER documents
            # (Logic omitted for brevity as it was commented out anyway)
            
            # No match found - create new key
            new_key = get_unique_key(field_path, item)
            identifier_to_key[normalized] = new_key
            key_to_identifier[new_key] = identifier
            return new_key

        # Build lookup dictionaries
        lookup = {"policy": {}, "binder": {}, "quote": {}}
        all_keys = set()

        # Process policy items first to establish the initial identifier mappings
        if policy is not None and p_list:
            for item in p_list or []:
                if isinstance(item, dict) and set(item.keys()).issubset({"value", "pages"}):
                    inner_item = item.get("value")
                    item_pages = item.get("pages") or p_pages_ctx or []
                else:
                    inner_item = item
                    item_pages = p_pages_ctx or []
                
                identifier = get_identifier_value(field_path, inner_item)
                # Policy list items never fuzzy match each other
                key = await get_or_create_key(identifier, inner_item, fuzzy_candidates=set())
                if identifier:
                    policy_ids.add(normalize(identifier))
                
                lookup["policy"][key] = {"_value": inner_item, "_pages": item_pages}
                all_keys.add(key)
        
        # Process binder items - they will match against policy identifiers
        if binder is not None and b_list:
            for item in b_list or []:
                if isinstance(item, dict) and set(item.keys()).issubset({"value", "pages"}):
                    inner_item = item.get("value")
                    item_pages = item.get("pages") or b_pages_ctx or []
                else:
                    inner_item = item
                    item_pages = b_pages_ctx or []
                
                identifier = get_identifier_value(field_path, inner_item)
                # Binder items can fuzzy match Policy items, but not other Binder items
                key = await get_or_create_key(identifier, inner_item, fuzzy_candidates=policy_ids)
                if identifier:
                    binder_ids.add(normalize(identifier))
                
                lookup["binder"][key] = {"_value": inner_item, "_pages": item_pages}
                all_keys.add(key)
        
        # Process quote items - they will match against both policy and binder identifiers
        if quote is not None and q_list:
            for item in q_list or []:
                if isinstance(item, dict) and set(item.keys()).issubset({"value", "pages"}):
                    inner_item = item.get("value")
                    item_pages = item.get("pages") or q_pages_ctx or []
                else:
                    inner_item = item
                    item_pages = q_pages_ctx or []
                
                identifier = get_identifier_value(field_path, inner_item)
                # Quote items can fuzzy match Policy or Binder items, but not other Quote items
                key = await get_or_create_key(identifier, inner_item, fuzzy_candidates=(policy_ids | binder_ids))
                
                lookup["quote"][key] = {"_value": inner_item, "_pages": item_pages}
                all_keys.add(key)

        # Determine default item type
        item_type_is_dict = any(isinstance(item.get("_value") if isinstance(item, dict) else item, dict) for source in lookup.values() for item in source.values())
        default_item = {} if item_type_is_dict else ""

        # Compare matched items deeply
        for key in sorted(all_keys):
            p_entry = lookup["policy"].get(key)
            b_entry = lookup["binder"].get(key)
            q_entry = lookup["quote"].get(key)
            p_item = (p_entry.get("_value") if isinstance(p_entry, dict) else default_item) if policy is not None else default_item
            b_item = (b_entry.get("_value") if isinstance(b_entry, dict) else default_item) if binder is not None else default_item
            q_item = (q_entry.get("_value") if isinstance(q_entry, dict) else default_item) if quote is not None else default_item
            p_item_pages = (p_entry.get("_pages") if isinstance(p_entry, dict) else (p_pages_ctx or [])) if policy is not None else (p_pages_ctx or [])
            b_item_pages = (b_entry.get("_pages") if isinstance(b_entry, dict) else (b_pages_ctx or [])) if binder is not None else (b_pages_ctx or [])
            q_item_pages = (q_entry.get("_pages") if isinstance(q_entry, dict) else (q_pages_ctx or [])) if quote is not None else (q_pages_ctx or [])

            await compare_nested(p_item, b_item, q_item, f"{field_path}[{key}]", p_item_pages, b_item_pages, q_item_pages)
    async def compare_nested(p_obj: Any, b_obj: Any, q_obj: Any, path: str = "", p_pages_ctx: List[int] = None, b_pages_ctx: List[int] = None, q_pages_ctx: List[int] = None):
        """
        ENHANCED: Recursively compare nested objects ensuring EVERY field is evaluated.
        """
        # Initialize page contexts
        p_pages_ctx = p_pages_ctx or []
        b_pages_ctx = b_pages_ctx or []
        q_pages_ctx = q_pages_ctx or []

        # Unwrap wrapper objects and capture pages
        def unwrap(obj: Any, inherited_pages: List[int]) -> tuple:
            if isinstance(obj, dict) and set(obj.keys()).issubset({"value", "pages"}):
                pages = obj.get("pages") or inherited_pages or []
                return obj.get("value"), pages
            return obj, inherited_pages or []

        p_obj, p_pages_ctx = unwrap(p_obj, p_pages_ctx)
        b_obj, b_pages_ctx = unwrap(b_obj, b_pages_ctx)
        q_obj, q_pages_ctx = unwrap(q_obj, q_pages_ctx)

        # Determine which objects are provided
        provided_objs = []
        if policy is not None and p_obj is not None:
            provided_objs.append(("policy", p_obj))
        if binder is not None and b_obj is not None:
            provided_objs.append(("binder", b_obj))
        if quote is not None and q_obj is not None:
            provided_objs.append(("quote", q_obj))

        # Skip if all provided objects are empty dicts
        if all(isinstance(obj, dict) and not obj for _, obj in provided_objs):
            return

        # If all provided objects are dicts, compare their keys
        if all(isinstance(obj, dict) for _, obj in provided_objs):
            # Collect ALL keys from all provided dictionaries
            all_keys = set()
            for _, obj in provided_objs:
                if isinstance(obj, dict):
                    all_keys.update(obj.keys())

            # Compare each key
            for key in sorted(all_keys):
                current_path = f"{path}.{key}" if path else key
                
                # Skip if in skip list
                if current_path in SKIP_FIELDS:
                    continue

                # Extract values for this key
                p_val = p_obj.get(key) if policy is not None and isinstance(p_obj, dict) else None
                b_val = b_obj.get(key) if binder is not None and isinstance(b_obj, dict) else None
                q_val = q_obj.get(key) if quote is not None and isinstance(q_obj, dict) else None

                # Unwrap per-key wrapper shapes if present to propagate pages
                def unwrap_child(val: Any, pages_ctx: List[int]) -> tuple:
                    if isinstance(val, dict) and set(val.keys()).issubset({"value", "pages"}):
                        return val.get("value"), (val.get("pages") or pages_ctx or [])
                    return val, pages_ctx or []

                p_val, p_child_pages = unwrap_child(p_val, p_pages_ctx)
                b_val, b_child_pages = unwrap_child(b_val, b_pages_ctx)
                q_val, q_child_pages = unwrap_child(q_val, q_pages_ctx)

                # Handle lists (arrays)
                if any(isinstance(val, list) for val in [p_val, b_val, q_val] if val is not None):
                    # Check if all are lists or None
                    all_lists = all(isinstance(val, list) or val is None for val in [p_val, b_val, q_val])
                    if all_lists:
                        # Check if they are lists of strings
                        is_str_list = True
                        for val in [p_val, b_val, q_val]:
                            if val is not None:
                                if not all(isinstance(item, str) for item in val):
                                    is_str_list = False
                                    break
                        if is_str_list:
                            # Normalize by sorting and joining for comparison
                            def get_norm(val):
                                if val is None:
                                    return ''
                                return ','.join(sorted(item.strip().lower() for item in val))
                            p_norm = get_norm(p_val)
                            b_norm = get_norm(b_val)
                            q_norm = get_norm(q_val)
                            check_values(p_val, b_val, q_val, current_path, p_child_pages, b_child_pages, q_child_pages)
                            continue
                    
                    p_list = p_val if isinstance(p_val, list) else []
                    b_list = b_val if isinstance(b_val, list) else []
                    q_list = q_val if isinstance(q_val, list) else []
                    
                    # Use identifier-based matching
                    await match_items_by_key(p_list, b_list, q_list, current_path, p_child_pages, b_child_pages, q_child_pages)
                
                # Handle nested dicts
                elif any(isinstance(val, dict) for val in [p_val, b_val, q_val] if val is not None):
                    await compare_nested(
                        p_val if isinstance(p_val, dict) else {},
                        b_val if isinstance(b_val, dict) else {},
                        q_val if isinstance(q_val, dict) else {},
                        current_path,
                        p_child_pages,
                        b_child_pages,
                        q_child_pages
                    )
                
                # Handle leaf values
                else:
                    check_values(p_val, b_val, q_val, current_path, p_child_pages, b_child_pages, q_child_pages)

        # Handle cases where objects are not all dicts (leaf comparison)
        else:
            check_values(
                p_obj if policy is not None else None,
                b_obj if binder is not None else None,
                q_obj if quote is not None else None,
                path,
                p_pages_ctx,
                b_pages_ctx,
                q_pages_ctx
            )

    # Storage for pending field comparisons
    pending_fields = []
    processed_field_paths = set()

    def check_values(p_val: Any, b_val: Any, q_val: Any, field_path: str, p_pages: List[int] = None, b_pages: List[int] = None, q_pages: List[int] = None):
        """Store field for later processing"""
        if field_path not in processed_field_paths:
            pending_fields.append((field_path, p_val, b_val, q_val, (p_pages or []), (b_pages or []), (q_pages or [])))
            processed_field_paths.add(field_path)

    # === START COMPARISON ===
    
    # Collect all possible fields
    async def collect_fields_async():
        all_fields = []
        
        async def collect_from_json(json_data, json_name):
            fields = []
            get_all_subfields(json_data, "", fields, set())
            return fields
        
        field_tasks = []
        if policy:
            field_tasks.append(collect_from_json(policy, "policy"))
        if binder:
            field_tasks.append(collect_from_json(binder, "binder"))
        if quote:
            field_tasks.append(collect_from_json(quote, "quote"))
        
        if field_tasks:
            field_results = await asyncio.gather(*field_tasks)
            for fields in field_results:
                all_fields.extend(fields)
        
        return list(set(all_fields))

    all_fields = await collect_fields_async()
    logger.info(f"Collected {len(all_fields)} unique fields to compare")

    # Perform deep recursive comparison
    await compare_nested(
        policy if policy is not None else {},
        binder if binder is not None else {},
        quote if quote is not None else {}
    )

    # Add remaining fields to pending_fields
    async def process_remaining_fields():
        remaining_fields = []
        for field in all_fields:
            if field in SKIP_FIELDS or field in processed_field_paths:
                continue
            
            # Get the values for this field path
            field_parts = field.split('.')
            p_val = get_nested_value(policy, field_parts) if policy else None
            b_val = get_nested_value(binder, field_parts) if binder else None
            q_val = get_nested_value(quote, field_parts) if quote else None
            
            # Skip fields that contain complex objects
            values_to_check = []
            if policy is not None:
                values_to_check.append(p_val)
            if binder is not None:
                values_to_check.append(b_val) 
            if quote is not None:
                values_to_check.append(q_val)
            
            has_complex = any(isinstance(val, (dict, list)) and val for val in values_to_check)
            
            if not has_complex:
                # Extract pages if wrapped
                def extract_pages(val: Any) -> List[int]:
                    if isinstance(val, dict) and set(val.keys()).issubset({"value", "pages"}):
                        return val.get("pages") or []
                    return []
                remaining_fields.append((field, p_val, b_val, q_val, extract_pages(p_val), extract_pages(b_val), extract_pages(q_val)))
                processed_field_paths.add(field)
        return remaining_fields
    
    remaining_fields = await process_remaining_fields()
    pending_fields.extend(remaining_fields)

    # Enhanced filtering to exclude container fields (lists of dicts, dicts with lists)
    seen_fields = set()
    unique_pending_fields = []
    container_fields = set()
    
    def is_container_field(field_path: str, p_val: Any, b_val: Any, q_val: Any) -> bool:
        """Check if a field is a container (list of dicts or dict with lists) that should be excluded."""
        # Check if this field path ends with a container pattern
        if '[' in field_path and ']' in field_path and field_path.endswith(']'):
            # This could be an array container, but we need to check the actual values
            # If the values are strings, numbers, or simple types, it's NOT a container
            for val in [p_val, b_val, q_val]:
                if val is not None:
                    if isinstance(val, (str, int, float, bool)):
                        # This is a concrete value, not a container
                        return False
                    elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                        # List of dictionaries - this is a container
                        return True
                    elif isinstance(val, dict):
                        # Check if this dict contains lists or nested dicts
                        has_nested_containers = any(
                            isinstance(v, (list, dict)) and v 
                            for v in val.values()
                        )
                        if has_nested_containers:
                            return True
            # If we get here, it's likely a container
            return True
        
        # Check if any of the values are containers
        for val in [p_val, b_val, q_val]:
            if val is not None:
                if isinstance(val, list) and len(val) > 0 and isinstance(val[0], dict):
                    # List of dictionaries - this is a container
                    return True
                elif isinstance(val, dict):
                    # Check if this dict contains lists or nested dicts
                    has_nested_containers = any(
                        isinstance(v, (list, dict)) and v 
                        for v in val.values()
                    )
                    if has_nested_containers:
                        return True
        return False
    
    def has_concrete_child_fields(field_path: str) -> bool:
        """Check if this container field has concrete child fields."""
        # Look for fields that are children of this container
        for pf in pending_fields:
            other_field = pf[0]
            if other_field.startswith(field_path + '.') or other_field.startswith(field_path + '['):
                # Check if the child field is concrete (not a container)
                return True
        return False
    
    # First pass: identify all container fields
    for field_data in pending_fields:
        field_path, p_val, b_val, q_val, *_ = field_data
        if is_container_field(field_path, p_val, b_val, q_val):
            container_fields.add(field_path)
    
    # Second pass: filter out container fields that have concrete children
    for field_data in pending_fields:
        field_path, p_val, b_val, q_val, *_ = field_data
        
        # Skip container fields that have concrete child fields
        if field_path in container_fields and has_concrete_child_fields(field_path):
            continue

        # Skip fields that are containers themselves (list of dicts, dict with lists)
        if is_container_field(field_path, p_val, b_val, q_val):
            continue
        
        if field_path not in seen_fields:
            unique_pending_fields.append(field_data)
            seen_fields.add(field_path)
    
    pending_fields = unique_pending_fields

    # Process all fields with group-based filtering
    from collections import defaultdict
    
    async def group_fields_async():
        grouped_fields = defaultdict(list)
        
        for field, p_val, b_val, q_val, *_ in pending_fields:
            # Extract group key
            if '[' in field and ']' in field:
                group_key = field.split(']')[0] + ']'
            else:
                field_parts = field.split('.')
                if len(field_parts) > 1:
                    group_key = '.'.join(field_parts[:-1])
                else:
                    group_key = 'root'
            
            grouped_fields[group_key].append((field, p_val, b_val, q_val))
        
        return grouped_fields
    
    grouped_fields = await group_fields_async()
    
    def is_empty(val: Any) -> bool:
        if val is None:
            return True
        if isinstance(val, (dict, list, str)):
            return not bool(val)
        return False
    
    async def process_group(group_key: str, fields: List[tuple]) -> tuple:
        """Process a single group of fields asynchronously"""
        # Check if any field in the group has meaningful data
        has_meaningful_data = False
        for field, p_val, b_val, q_val in fields:
            if (policy is not None and not is_empty(p_val)) or \
               (binder is not None and not is_empty(b_val)) or \
               (quote is not None and not is_empty(q_val)):
                has_meaningful_data = True
                break
        
        group_anomalies = []
        group_matched = 0
        group_mismatched = 0
        
        if has_meaningful_data:
            async def process_field(field_data: tuple) -> tuple:
                field, p_val, b_val, q_val = field_data
                
                def normalize_val(val: Any) -> str:
                    if isinstance(val, list) and all(isinstance(x, str) for x in val):
                        return ','.join(sorted(x.strip().lower() for x in val))
                    if val is None or val == "":
                        return ""
                    str_val = str(val).strip().lower()
                    cleaned = re.sub(r'[\$\s,]', '', str_val)
                    try:
                        num = float(cleaned)
                        if num.is_integer():
                            return str(int(num))
                        else:
                            return str(num)
                    except ValueError:
                        return str_val

                # Normalize values for comparison
                normalized_values = []
                if policy is not None:
                    normalized_values.append(normalize_val(p_val))
                if binder is not None:
                    normalized_values.append(normalize_val(b_val))
                if quote is not None:
                    normalized_values.append(normalize_val(q_val))

                filtered_values = [v for v in normalized_values]

                # Determine status
                forms_special_handled = False
                if "forms_and_endorsements" in field:
                    bracket_match = re.search(r'\[([^\]]+)\]', field)
                    if bracket_match:
                        key = bracket_match.group(1)
                        # Fix: close regex string and parenthesis, correct logic
                        if len(key) < 36 and re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', key) is None:
                            num_present = sum(1 for val in [p_val, b_val, q_val] if val is not None)
                            status = "matched" if num_present >= 2 else "mismatched"
                            is_matched = status == "matched"
                            forms_special_handled = True

                if not forms_special_handled:
                    if len(filtered_values) < 2:
                        status = "mismatched"
                        is_matched = False
                    else:
                        raw_values: List[str] = []
                        if policy is not None:
                            raw_values.append("" if p_val is None else str(p_val))
                        if binder is not None:
                            raw_values.append("" if b_val is None else str(b_val))
                        if quote is not None:
                            raw_values.append("" if q_val is None else str(q_val))

                        numeric_tokens_per_value = [extract_numeric_tokens(v) for v in raw_values]
                        conflict_detected = False
                        for i in range(len(numeric_tokens_per_value)):
                            for j in range(i + 1, len(numeric_tokens_per_value)):
                                a = numeric_tokens_per_value[i]
                                b = numeric_tokens_per_value[j]
                                if a and b and a != b:
                                    conflict_detected = True
                                    break
                            if conflict_detected:
                                break

                        if conflict_detected:
                            status = "mismatched"
                            is_matched = False
                        else:
                            has_numeric = any(is_numeric(val) for val in filtered_values if val != "")
                            
                            if has_numeric:
                                all_equal = True
                                for i in range(len(filtered_values)):
                                    for j in range(i + 1, len(filtered_values)):
                                        if filtered_values[i] != filtered_values[j]:
                                            all_equal = False
                                            break
                                    if not all_equal:
                                        break
                                status = "matched" if all_equal else "mismatched"
                                is_matched = (status == "matched")
                            else:
                                has_comma_values = any(',' in val for val in filtered_values)
                                
                                if has_comma_values:
                                    parts_list = []
                                    for v in filtered_values:
                                        parts = sorted(part.strip() for part in v.split(',') if part.strip())
                                        parts_list.append(parts)
                                    
                                    if "common_exclusions" in field.lower():
                                        normalized_parts_list = []
                                        for parts in parts_list:
                                            normalized = []
                                            for part in parts:
                                                normalized_part = part.lower().replace(" exclusion", "").replace(" clause", "").strip()
                                                items = []
                                                for item in normalized_part.split(", "):
                                                    if " and " in item:
                                                        items.extend([x.strip() for x in item.split(" and ")])
                                                    elif " or " in item:
                                                        items.extend([x.strip() for x in item.split(" or ")])
                                                    else:
                                                        items.append(item.strip())
                                                normalized.extend(items)
                                            normalized_parts_list.append(sorted(normalized))
                                        
                                        all_sets = [set(parts) for parts in normalized_parts_list]
                                        all_similar = all(s == all_sets[0] for s in all_sets)
                                    else:
                                        lengths = [len(p) for p in parts_list]
                                        if len(set(lengths)) != 1:
                                            all_similar = False
                                        else:
                                            all_exact = True
                                            for i in range(lengths[0]):
                                                els = [p[i] for p in parts_list]
                                                if not all(e == els[0] for e in els):
                                                    all_exact = False
                                                    break
                                            
                                            if all_exact:
                                                all_similar = True
                                            else:
                                                all_fuzzy = True
                                                for i in range(lengths[0]):
                                                    els = [p[i] for p in parts_list]
                                                    for j in range(len(els)):
                                                        for k in range(j + 1, len(els)):
                                                            sim = fuzz.partial_ratio(els[j], els[k])
                                                            if sim < similarity_threshold:
                                                                all_fuzzy = False
                                                                break
                                                        if not all_fuzzy:
                                                            break
                                                all_similar = all_fuzzy
                                    status = "matched" if all_similar else "mismatched"
                                    is_matched = (status == "matched")
                                else:
                                    all_similar = True
                                    
                                    for i in range(len(normalized_values)):
                                        for j in range(i + 1, len(normalized_values)):
                                            similarity = fuzz.partial_ratio(normalized_values[i], normalized_values[j])
                                            if similarity < similarity_threshold:
                                                all_similar = False
                                                break
                                        if not all_similar:
                                            break
                                    
                                    status = "matched" if all_similar else "mismatched"
                                    is_matched = (status == "matched")

                # Format values for display
                def format_value(val: Any, provided_dict: Dict[str, Any], this_key: str, threshold: float, field_path: str) -> Any:
                    if val is None:
                        return ""  # Return empty string for missing field values, not None
                    if isinstance(val, dict):
                        return "{}" if not val else str(val)
                    
                    if "forms_and_endorsements" in field_path:
                        bracket_match = re.search(r'\[([^\]]+)\]', field_path)
                        if bracket_match:
                            bracket_key = bracket_match.group(1)
                            if not re.match(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', bracket_key):
                                if val != "":
                                    return f"{bracket_key}:{val}"
                                else:
                                    return f"{bracket_key}:"
                    
                    is_common_exclusions = "common_exclusions" in field_path.lower()
                    
                    items = None
                    original_items = []
                    apply_wrap = False
                    
                    if isinstance(val, list) and all(isinstance(x, str) for x in val):
                        original_items = val
                        apply_wrap = is_common_exclusions
                    elif isinstance(val, str) and ',' in val:
                        original_items = [part.strip() for part in val.split(',') if part.strip()]
                        if original_items:
                            apply_wrap = is_common_exclusions
                    else:
                        return str(val).strip().title() if '@' not in str(val) else str(val).strip()
                    
                    if not apply_wrap:
                        if isinstance(val, list):
                            # For list of strings, return the list as-is (not comma-separated string)
                            return val
                        return str(val).strip().title() if '@' not in str(val) else str(val).strip()
                    
                    other_vals = [provided_dict[k] for k in provided_dict if k != this_key and provided_dict[k] is not None]
                    if len(other_vals) == 0:
                        formatted_items = [str(x).strip().title() if '@' not in str(x) else str(x).strip() for x in original_items]
                        # For list of strings, return as list; for comma-separated strings, return as string
                        if isinstance(val, list):
                            return formatted_items
                        return ", ".join(formatted_items)
                    
                    other_norms = []
                    for ov in other_vals:
                        if isinstance(ov, list) and all(isinstance(x, str) for x in ov):
                            other_norms.append([str(x).strip().lower() for x in ov])
                        elif isinstance(ov, str):
                            other_norms.append([part.strip().lower() for part in ov.split(',') if part.strip()])
                        else:
                            other_norms.append([])
                    
                    formatted_items = []
                    for item in original_items:
                        norm = str(item).strip().lower()
                        
                        if is_common_exclusions:
                            normalized_item = norm.replace(" exclusion", "").replace(" clause", "").strip()
                            
                            def normalize_for_comparison(text):
                                text = text.replace(" exclusion", "").replace(" clause", "").strip()
                                items = []
                                for item in text.split(", "):
                                    if " and " in item:
                                        items.extend([x.strip() for x in item.split(" and ")])
                                    elif " or " in item:
                                        items.extend([x.strip() for x in item.split(" or ")])
                                    else:
                                        items.append(item.strip())
                                return items
                            
                            normalized_item_parts = normalize_for_comparison(norm)
                            
                            has_match_in_all_others = all(
                                any(any(part in normalize_for_comparison(o) for part in normalized_item_parts) for o in other_norm)
                                for other_norm in other_norms
                            )
                        else:
                            has_match_in_all_others = all(
                                any(fuzz.partial_ratio(norm, o) >= threshold for o in other_norm)
                                for other_norm in other_norms
                            )
                        
                        formatted_item = str(item).strip().title() if '@' not in str(item) else str(item).strip()
                        if not has_match_in_all_others:
                            formatted_item = f"***{formatted_item}***"
                        formatted_items.append(formatted_item)
                    
                    # For list of strings, return as list; for comma-separated strings, return as string
                    if isinstance(val, list):
                        return formatted_items
                    return ", ".join(formatted_items)

                p_display = p_val
                b_display = b_val
                q_display = q_val

                provided_dict = {"policy": p_display, "binder": b_display, "quote": q_display}
                p_val_str = format_value(p_display, provided_dict, "policy", similarity_threshold, field)
                b_val_str = format_value(b_display, provided_dict, "binder", similarity_threshold, field)
                q_val_str = format_value(q_display, provided_dict, "quote", similarity_threshold, field)

                if policy is None:
                    p_val_str = None
                if binder is None:
                    b_val_str = None
                if quote is None:
                    q_val_str = None

                # Lookup pages for this field from pending_fields
                p_pages = []
                b_pages = []
                q_pages = []
                try:
                    for pf in pending_fields:
                        if pf[0] == field:
                            _, _, _, _, p_pages, b_pages, q_pages = pf
                            break
                except Exception:
                    pass

                anomaly = Anomaly(
                    anomalyFieldName=field,
                    policy_value=p_val_str,
                    binder_value=b_val_str,
                    quote_value=q_val_str,
                    status=status,
                    field_name=get_descriptive_field_name(field, p_val, b_val, q_val),
                    policy_pages=p_pages or None,
                    binder_pages=b_pages or None,
                    quote_pages=q_pages or None
                )
                
                return anomaly, is_matched

            field_tasks = [process_field(field_data) for field_data in fields]
            field_results = await asyncio.gather(*field_tasks)
            
            for anomaly, is_matched in field_results:
                group_anomalies.append(anomaly)
                if is_matched:
                    group_matched += 1
                else:
                    group_mismatched += 1
        
        return group_anomalies, group_matched, group_mismatched

    # Process all groups concurrently
    group_tasks = [process_group(group_key, fields) for group_key, fields in grouped_fields.items()]
    group_results = await asyncio.gather(*group_tasks, return_exceptions=True)
    
    for result in group_results:
        if isinstance(result, Exception):
            logger.error(f"Error processing group: {result}")
            continue
        group_anomalies, group_matched, group_mismatched = result
        anomalies.extend(group_anomalies)
        matched_count += group_matched
        mismatched_count += group_mismatched

    # Verification with OpenAI or Gemini (placeholder for actual implementation)
    # Note: Add your actual verification logic here if needed
    # ===== Verification - Post Processing of MISMATCHED Anomalies =====
    async def verify_anomalies_with_gemini(all_anomalies: List[Anomaly]) -> List[Anomaly]:
        # Ensure shared gemini_client is available and configured
        try:
            _ = gemini_client.aio
        except Exception as e:
            logger.warning(f"Gemini client not available for verification: {e}")
            return all_anomalies

        # Collect indices of anomalies that are mismatched (case-insensitive safety)
        target_indices: List[int] = [idx for idx, a in enumerate(all_anomalies) if (a.status or '').lower() == 'mismatched']
        if not target_indices:
            return all_anomalies

        # Prepare batches
        BATCH_SIZE = 50
        batches: List[List[int]] = [target_indices[i:i + BATCH_SIZE] for i in range(0, len(target_indices), BATCH_SIZE)]

        system_instruction = (
            "You are a strict verifier. You will be given a list of anomalies from a field-by-field comparison of three sources: "
            "Policy, Binder, and Quote. Each anomaly has values and a current status=\"MISMATCHED\". Your job is to verify whether each anomaly "
            "is truly a mismatch. If values are effectively the same despite formatting (e.g., currency symbols, commas, case), mark it MATCHED; "
            "otherwise keep MISMATCHED. Return a compact JSON array where each item has: {index, status}. Status must be either MATCHED or MISMATCHED."
        )

        async def verify_batch(batch_indices: List[int]) -> List[dict]:
            nonlocal verification_input_tokens, verification_output_tokens, verification_cached_tokens
            payload = []
            for idx in batch_indices:
                a = all_anomalies[idx]
                payload.append({
                    "index": idx,
                    "anomalyFieldName": a.anomalyFieldName,
                    "policy_value": a.policy_value,
                    "binder_value": a.binder_value,
                    "quote_value": a.quote_value,
                    "current_status": a.status,
                    "field_name": a.field_name,
                })

            prompt = (
                f"System Instruction:\n{system_instruction}\n\n"
                f"Anomalies JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
                "Return ONLY a valid JSON array like: [{\"index\":0,\"status\":\"MATCHED\"}] with no extra text."
            )

            try:
                resp = await gemini_client.aio.models.generate_content(
                    model=(verification_model or GeminiModels.FLASH.value),
                    contents=[prompt]
                )
                text = (resp.text or '').strip()
                # Token usage aggregation (Gemini)
                try:
                    usage = getattr(resp, 'usage_metadata', None)
                    if usage is not None:
                        # Support various SDK field names
                        verification_input_tokens += int(getattr(usage, 'prompt_token_count', 0) or getattr(usage, 'input_tokens', 0) or 0)
                        verification_output_tokens += int(getattr(usage, 'candidates_token_count', 0) or getattr(usage, 'output_tokens', 0) or 0)
                        verification_cached_tokens += int(getattr(usage, 'cached_content_token_count', 0) or 0)
                except Exception:
                    pass
                # Attempt to extract JSON block
                parsed = None
                try:
                    parsed = json.loads(text)
                except Exception:
                    # Try to find JSON within text
                    import re as _re
                    match = _re.search(r"\[.*\]", text, _re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                if not isinstance(parsed, list):
                    logger.warning("Gemini response is not a list; skipping batch update")
                    return []
                return parsed
            except Exception as e:
                logger.warning(f"Gemini verification failed for a batch: {e}")
                return []

        # Run all batches in parallel
        batch_tasks = [verify_batch(b) for b in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Apply updates
        idx_to_status: Dict[int, str] = {}
        for res in batch_results:
            if isinstance(res, Exception):
                logger.warning(f"Gemini batch task error: {res}")
                continue
            for item in res or []:
                try:
                    idx = int(item.get("index"))
                    status = str(item.get("status") or '').upper()
                    if status in {"MATCHED", "MISMATCHED"} and idx in target_indices:
                        idx_to_status[idx] = status.lower()
                except Exception:
                    continue

        # Replace statuses for verified subset
        if idx_to_status:
            for idx, new_status in idx_to_status.items():
                old = all_anomalies[idx]
                # If status is MATCHED, strip *** from all string values
                if new_status == "matched":
                    def _clean(val):
                        if isinstance(val, str):
                            return val.replace("***", "")
                        return val
                    policy_val = _clean(old.policy_value)
                    binder_val = _clean(old.binder_value)
                    quote_val = _clean(old.quote_value)
                else:
                    policy_val = old.policy_value
                    binder_val = old.binder_value
                    quote_val = old.quote_value

                all_anomalies[idx] = Anomaly(
                    anomalyFieldName=old.anomalyFieldName,
                    policy_value=policy_val,
                    binder_value=binder_val,
                    quote_value=quote_val,
                    status=new_status,
                    field_name=old.field_name,
                    policy_pages=old.policy_pages,
                    binder_pages=old.binder_pages,
                    quote_pages=old.quote_pages,
                )


        return all_anomalies

    async def verify_anomalies_with_openai(all_anomalies: List[Anomaly]) -> List[Anomaly]:
        try:
            _ = AsyncOpenAI
        except Exception as e:
            logger.warning(f"OpenAI client not available for verification: {e}")
            return all_anomalies

        def is_non_empty(val):
            return str(val).strip() != ""

        target_indices: List[int] = [
            idx for idx, a in enumerate(all_anomalies)
            if (a.status or '').lower() == 'mismatched'
            and all(is_non_empty(v) for v in [a.policy_value, a.binder_value, a.quote_value])
        ]
        if not target_indices:
            return all_anomalies

        BATCH_SIZE = 25
        batches: List[List[int]] = [target_indices[i:i + BATCH_SIZE] for i in range(0, len(target_indices), BATCH_SIZE)]

        client = AsyncOpenAI(api_key=ENV_PROJECT.OPENAI_API_KEY)
        system_instruction = ('''
You are a **strict verifier**.  
You will be given a list of anomalies from a field-by-field comparison of three sources: **Policy, Binder, and Quote**.  

- Each anomaly has values (`policy_value`, `binder_value`, `quote_value`) and a current `status="MISMATCHED"`.  
- Your job is to **verify whether each anomaly is truly a mismatch** by comparing the values across all three sources.  

### Rules for Verification
1. **Handle Null Values**:  
   - Treat `null` and `-` as indicating no uploaded data. Exclude these values from the comparison process. Only compare non-`null` values among `policy_value`, `binder_value`, and `quote_value`.  
   - If all three values are `null` or `-`, mark as **MATCHED** (no data to compare).  
   - If at least one value is non-`null` and others are `null`, compare only the non-`null` values if there are at least two; otherwise, mark as **MISMATCHED** due to insufficient data.

2. **Normalize Values Before Comparison**:  
   - Remove currency symbols (e.g., `$`, `£`), commas, extra whitespace, and convert to lowercase.  
   - Expand common abbreviations (e.g., "BI" to "Bodily Injury", "PD" to "Property Damage") where contextually appropriate.  

3. **Mark as MATCHED if Values Are Effectively Equivalent**:  
   - If normalized non-`null` values are identical or convey the same meaning despite differences in formatting or punctuation, mark as **MATCHED**.  

4. **Mark as MISMATCHED if Values Differ in Meaning**:  
   - If normalized non-`null` values differ in content or intent (e.g., different amounts, different terms), mark as **MISMATCHED**.  

5. **Strict Rule for Legal/Contractual Clauses**:  
   - If one non-`null` value is a **long legal/contractual clause** and another is a **short label, abbreviation, or oversimplified phrase**, mark as **MISMATCHED** unless their meanings are **explicitly identical** after normalization.  

6. **Default to MISMATCHED Only When Unavoidable**:  
   - Only retain `status="MISMATCHED"` if no reasonable normalization or contextual equivalence can be established among non-`null` values.  

### Expected Output
Return a **compact JSON array** where each item follows the structure:  

```json
{ "index": <number>, "status": "MATCHED" | "MISMATCHED" }

        ''')

        async def verify_batch(batch_indices: List[int]) -> List[dict]:
            nonlocal verification_input_tokens, verification_output_tokens, verification_cached_tokens
            payload = []
            for idx in batch_indices:
                a = all_anomalies[idx]
                payload.append({
                    "index": idx,
                    "anomalyFieldName": a.anomalyFieldName,
                    "policy_value": a.policy_value,
                    "binder_value": a.binder_value,
                    "quote_value": a.quote_value,
                    "current_status": a.status,
                    "field_name": a.field_name,
                })

            user_prompt = (
                f"Instruction:\n{system_instruction}\n\n"
                f"Anomalies JSON:\n{json.dumps(payload, ensure_ascii=False)}\n\n"
            )

            try:
                resp = await client.chat.completions.create(
                    model=(verification_model or ENV_PROJECT.GPT_VERIFICATION_MODEL),
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=1
                )
                text = (resp.choices[0].message.content or '').strip()
                # Token usage aggregation (OpenAI)
                try:
                    usage = getattr(resp, 'usage', None)
                    if usage is not None:
                        pt = getattr(usage, 'prompt_tokens', None)
                        ct = getattr(usage, 'completion_tokens', None)
                        it = getattr(usage, 'input_tokens', None)
                        ot = getattr(usage, 'output_tokens', None)
                        verification_input_tokens += int((it if it is not None else (pt or 0)) or 0)
                        verification_output_tokens += int((ot if ot is not None else (ct or 0)) or 0)
                        verification_cached_tokens += int(getattr(usage, 'cache_creation_input_tokens', 0) or getattr(usage, 'cached_tokens', 0) or 0)
                except Exception:
                    pass
                parsed = None
                try:
                    parsed = json.loads(text)
                except Exception:
                    import re as _re
                    match = _re.search(r"$$         .*         $$", text, _re.DOTALL)
                    if match:
                        parsed = json.loads(match.group(0))
                if not isinstance(parsed, list):
                    logger.warning("OpenAI response is not a list; skipping batch update")
                    return []
                return parsed
            except Exception as e:
                logger.warning(f"OpenAI verification failed for a batch: {e}")
                return []

        batch_tasks = [verify_batch(b) for b in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        idx_to_status: Dict[int, str] = {}
        for res in batch_results:
            if isinstance(res, Exception):
                logger.warning(f"OpenAI batch task error: {res}")
                continue
            for item in res or []:
                try:
                    idx = int(item.get("index"))
                    status = str(item.get("status") or '').upper()
                    if status in {"MATCHED", "MISMATCHED"} and idx in target_indices:
                        idx_to_status[idx] = status.lower()
                except Exception:
                    continue

        if idx_to_status:
            for idx, new_status in idx_to_status.items():
                old = all_anomalies[idx]
                # If status is MATCHED, strip *** from all string values
                if new_status == "matched":
                    def _clean(val):
                        if isinstance(val, str):
                            return val.replace("***", "")
                        return val
                    policy_val = _clean(old.policy_value)
                    binder_val = _clean(old.binder_value)
                    quote_val = _clean(old.quote_value)
                else:
                    policy_val = old.policy_value
                    binder_val = old.binder_value
                    quote_val = old.quote_value

                all_anomalies[idx] = Anomaly(
                    anomalyFieldName=old.anomalyFieldName,
                    policy_value=policy_val,
                    binder_value=binder_val,
                    quote_value=quote_val,
                    status=new_status,
                    field_name=old.field_name,
                    policy_pages=old.policy_pages,
                    binder_pages=old.binder_pages,
                    quote_pages=old.quote_pages,
                )

        return all_anomalies

    # Execute Gemini verification (best-effort; non-fatal on failure)
    try:
        if (verification_provider or "gemini").lower() == "openai":
            anomalies = await verify_anomalies_with_openai(anomalies)
        else:
            anomalies = await verify_anomalies_with_gemini(anomalies)
            try:
                await gemini_client.aio.aclose()
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"Gemini verification step skipped due to error: {e}")
    
    # Recount matched/mismatched after verification updates
    matched_count = sum(1 for a in anomalies if a.status == "matched")
    mismatched_count = sum(1 for a in anomalies if a.status == "mismatched")
    
    end_time = time.time()
    logger.info(f"Comparison completed in {end_time - start_time:.2f} seconds")
    logger.info(f"Total anomalies: {len(anomalies)}, Matched: {matched_count}, Mismatched: {mismatched_count}")

    # Aggregate semantic similarity check tokens into verification tokens
    verification_input_tokens += semantic_token_tracker.get('input_token', 0)
    verification_output_tokens += semantic_token_tracker.get('output_token', 0)
    verification_cached_tokens += semantic_token_tracker.get('cached_token', 0)
    
    logger.info(f"Semantic similarity tokens - Input: {semantic_token_tracker.get('input_token', 0)}, Output: {semantic_token_tracker.get('output_token', 0)}, Cached: {semantic_token_tracker.get('cached_token', 0)}")
    logger.info(f"Semantic matching stats - Total calls: {semantic_match_stats['total_calls']}, Total comparisons: {semantic_match_stats['total_comparisons']}, Matches found: {semantic_match_stats['matches_found']}")
    if semantic_match_stats['fields_checked']:
        logger.info(f"Fields that triggered semantic checks ({len(semantic_match_stats['fields_checked'])}): {', '.join(semantic_match_stats['fields_checked'])}")
    
    # Build verification token map for return
    model_key = (
        (verification_model or ENV_PROJECT.GPT_VERIFICATION_MODEL)
        if (verification_provider or "gemini").lower() == "openai"
        else (verification_model or GeminiModels.FLASH.value)
    )

    token_map = {
        model_key: {
            "input_token": verification_input_tokens,
            "output_token": verification_output_tokens,
            "cached_token": verification_cached_tokens,
        }
    }

    return {
        "anomalies": anomalies,
        "matched_count": matched_count,
        "mismatched_count": mismatched_count,
        "total_fields": matched_count + mismatched_count,
        "token": token_map,
    }


# Example usage
# from commercial_data import sample_data_12

# async def main():
#     result = await commercial_compare_jsons(
#         sample_data_12.get("Policy").get("commercial_extracted_data"),
#         sample_data_12.get("Binder").get("commercial_extracted_data")
#     )

#     logger.info("token usage: ", result["token"])
#     logger.info("matched count: ", result["matched_count"])
#     logger.info("mismatched count: ", result["mismatched_count"])
#     logger.info("total fields: ", result["total_fields"])

#     # Serialize anomalies safely in a thread
#     serialized_anomalies = await asyncio.to_thread(
#         lambda: [anomaly.model_dump() for anomaly in result["anomalies"]]
#     )

#     # Sort anomalies by anomalyFieldName (ascending)
#     sorted_anomalies = sorted(
#         serialized_anomalies,
#         key=lambda x: x.get("anomalyFieldName", "")
#     )

#     # Write sorted anomalies to file
#     with open("anomalies_15.json", "w") as f:
#         json.dump(sorted_anomalies, f, indent=4)
    
# if __name__ == "__main__":
#     asyncio.run(main())