import asyncio
import sys

from app.insurance_policy_processor.policy_data_extractor.policy_direct_field_extractor import (
    extract_policy_direct_fields,
)
from app.insurance_policy_processor.policy_data_extractor.policy_list_extractor import (
    extract_list_fields_with_new_mapper,
    extract_policy_list_fields,
)
from app.insurance_policy_processor.policy_data_extractor.policy_object_extractor import (
    extract_policy_object_fields,
)
from app.insurance_policy_processor.policy_orchestrator.models.state_models import (
    InsuranceDocumentState,
)
from app.utils.core import (
    generate_short_schema_for_context,
)
from app.logging_config import get_logger
from app.utils.helpers import save_json, load_json
from app.insurance_policy_processor.policy_orchestrator.enums_new import (
    CUSTOM_FIELD_PROMPT_EXTRACTION,
)

logger = get_logger(__name__)


async def bifurcate_object(
    pdf_path: str,
    parent_key: str,
    extraction_schema: dict,
    tmp_dir: str,
    extraction_instruction: str,
    page_numbers=list,
    original_pages=list,
    schema_context: dict = {},
):
    result = {
        parent_key: None,
        "token_consumptions": [],
    }
    try:
        list_separation_needed = False
        list_counts = 0
        object_counts = 0
        total_keys = len(extraction_schema.keys())
        for key, val in extraction_schema.items():
            if isinstance(val, list):
                list_counts += 1
            elif isinstance(val, dict):
                object_counts += 1

        if list_counts > 1:
            list_separation_needed = True
        elif list_counts == 1 and object_counts >= 1:
            list_separation_needed = True
        elif list_counts == 1 and total_keys > 5:
            list_separation_needed = True
        else:
            list_separation_needed = False

        all_list_schemas = []
        new_extraction_schema = {}
        if list_separation_needed:
            for key, val in extraction_schema.items():
                if isinstance(val, list):
                    all_list_schemas.append({key: val})
                else:
                    new_extraction_schema[key] = val
        else:
            new_extraction_schema = extraction_schema

        run_extraction = []
        final_result = {}

        if list_separation_needed and all_list_schemas:
            list_schema_context = await generate_short_schema_for_context(
                extraction_schema=extraction_schema
            )
            for list_schema in all_list_schemas:
                run_extraction.append(
                    extract_list_fields_with_new_mapper(
                        pdf_path=pdf_path,
                        parent_key=list(list_schema.keys())[0],
                        extraction_schema=list_schema,
                        page_numbers=page_numbers,
                        original_pages=original_pages,
                        tmp_dir=tmp_dir,
                        extraction_instruction=extraction_instruction,
                        schema_context=list_schema_context,
                    )
                )
        if new_extraction_schema:
            run_extraction.append(
                extract_policy_object_fields(
                    pdf_path=pdf_path,
                    parent_key=parent_key,
                    extraction_schema={parent_key: new_extraction_schema},
                    page_numbers=page_numbers,
                    original_pages=original_pages,
                    tmp_dir=tmp_dir,
                    extraction_instruction=extraction_instruction,
                    schema_context=schema_context,
                )
            )

        tokens_used = []
        final_result = {}

        object_results = await asyncio.gather(*run_extraction, return_exceptions=True)
        for obj_result in object_results:
            if isinstance(obj_result, Exception):
                continue
            is_object = obj_result.get("object", False)
            if is_object:
                # Ensure final_result is always a dict, never None
                extracted_data = obj_result.get(parent_key, {})
                final_result = extracted_data if extracted_data is not None else {}
                tokens_used.extend(obj_result.get("token_consumptions", []))
                break

        if not final_result:
            final_result = {}

        if list_separation_needed:
            for obj_result in object_results:
                if isinstance(obj_result, Exception):
                    continue
                is_object = obj_result.get("object", False)
                if not is_object:
                    for key, val in obj_result.items():
                        if key and key == "token_consumptions" and val:
                            tokens_used.extend(val)
                        elif key and key != "token_consumptions" and key != "object":
                            final_result[key] = val

        result["token_consumptions"] = tokens_used
        result[parent_key] = final_result
        return result
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(
            f"Error in bifurcation for '{parent_key}': {e} (Line {lineno})",
            exc_info=True,
        )
        return {parent_key: None, "token_consumptions": []}


async def extract_nested_coverage_object(
    pdf_path: str,
    parent_key: str,
    extraction_schema: dict,
    tmp_dir: str,
    extraction_instruction: str,
    nested_fields_mapping: dict,
    schema_context: dict = {},
):
    result = {
        parent_key: None,
        "token_consumptions": [],
    }
    try:
        run_extraction = []
        final_result = {}
        for key, val in nested_fields_mapping.items():
            # if not key == "forms_and_endorsements":
            #     continue

            if key not in extraction_schema:
                continue

            field_instruction = CUSTOM_FIELD_PROMPT_EXTRACTION.get(key, None)

            if isinstance(extraction_schema.get(key), list):
                run_extraction.append(
                    extract_policy_list_fields(
                        pdf_path=pdf_path,
                        parent_key=key,
                        extraction_schema={key: extraction_schema.get(key)},
                        page_numbers=val.get("relative_pages"),
                        original_pages=val.get("original_pages"),
                        tmp_dir=tmp_dir,
                        extraction_instruction=field_instruction
                        or extraction_instruction,
                        schema_context=schema_context,
                    )
                )
            elif not isinstance(extraction_schema.get(key), dict):
                run_extraction.append(
                    extract_policy_direct_fields(
                        pdf_path=pdf_path,
                        parent_key=key,
                        extraction_schema={key: extraction_schema.get(key)},
                        page_numbers=val.get("relative_pages"),
                        original_pages=val.get("original_pages"),
                        tmp_dir=tmp_dir,
                        extraction_instruction=field_instruction
                        or extraction_instruction,
                        schema_context=schema_context,
                    )
                )
            else:
                run_extraction.append(
                    bifurcate_object(
                        pdf_path=pdf_path,
                        parent_key=key,
                        extraction_schema=extraction_schema.get(key),
                        page_numbers=val.get("relative_pages"),
                        tmp_dir=tmp_dir,
                        extraction_instruction=field_instruction
                        or extraction_instruction,
                        original_pages=val.get("original_pages"),
                        schema_context=schema_context,
                    )
                )

        results = await asyncio.gather(*run_extraction, return_exceptions=True)
        tokens_used = []
        for res in results:
            if isinstance(res, Exception):
                continue
            for key, val in res.items():
                if key and key == "token_consumptions" and val:
                    tokens_used.extend(val)
                elif key and key != "token_consumptions":
                    final_result[key] = val

        result[parent_key] = final_result
        result["token_consumptions"] = tokens_used
        return result
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(
            f"Error in nested extraction for '{parent_key}' at (Line {lineno}): {e}",
            exc_info=True,
        )
        return {
            parent_key: None,
            "token_consumptions": [],
        }


async def run_extraction_for_individual_coverage_section(
    first_level_key,
    data,
    tmp_dir,
    idx,
    extraction_instruction: str,
):
    try:
        # Adding sleep
        await asyncio.sleep(delay=idx * 1.5)

        logger.info(f"Running extraction for: {first_level_key}")

        pdf_path = data.get("pdf_path", "")
        schema = data.get("schema", {})
        pages = data.get("pages")

        # Different Extraction Flow based on schema type
        if data.get("is_list"):
            result = await extract_policy_list_fields(
                pdf_path=pdf_path,
                parent_key=first_level_key,
                extraction_schema=schema,
                tmp_dir=tmp_dir,
                extraction_instruction=extraction_instruction,
                page_numbers=pages,
                original_pages=pages,
                is_first_level=True,
            )
            return result

        elif data.get("is_field"):
            result = await extract_policy_direct_fields(
                pdf_path=pdf_path,
                parent_key=first_level_key,
                extraction_schema=schema,
                tmp_dir=tmp_dir,
                extraction_instruction=extraction_instruction,
                page_numbers=pages,
                original_pages=pages,
                is_first_level=True,
            )
            return result

        else:
            nested_fields_mapping = data.get("nested_fields_mapping", {})
            schema_context = data.get("schema_context", {})

            result = await extract_nested_coverage_object(
                pdf_path=pdf_path,
                parent_key=first_level_key,
                extraction_schema=schema,
                tmp_dir=tmp_dir,
                extraction_instruction=extraction_instruction,
                nested_fields_mapping=nested_fields_mapping,
                schema_context=schema_context,
            )
            return result

    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(
            f"Error in extraction for {first_level_key} at (Line {lineno}): {str(e)}",
            exc_info=True,
        )
        return {
            first_level_key: {},
            "token_consumptions": [],
            "_error": str(e),
        }


async def extract_policy_data(state: InsuranceDocumentState):
    doc_id = state.policy_document_id
    
    # Get Stored Response
    # final_result = await load_json(file_path="final_result.json")
    # state.extracted_policy_data = final_result.get("extracted_policy_data")
    # logger.info("Extraction result loaded from file")
    # return state

    logger.info(f"[document:{doc_id}] policy_data_extractor: starting")
    coverage_detail_mapping = state.coverage_detail_mapping_result

    if not coverage_detail_mapping:
        logger.warning(f"[document:{doc_id}] policy_data_extractor: coverage_detail_mapping is empty — skipping extraction")
        state.extracted_policy_data = {}
        return state

    final_result = {}
    all_extractions = []
    lob_keys = list(coverage_detail_mapping.keys())

    logger.info(f"[document:{doc_id}] policy_data_extractor: starting parallel extraction for {len(lob_keys)} lobs: {lob_keys}")

    idx = 0
    for first_level_key, val in coverage_detail_mapping.items():
        all_extractions.append(
            run_extraction_for_individual_coverage_section(
                first_level_key=first_level_key,
                data=val,
                tmp_dir=state.tmp_dir,
                idx=idx,
                extraction_instruction=state.extraction_instruction,
            )
        )
        idx += 1

    logger.info(f"Running {len(all_extractions)} extraction tasks in parallel")
    results = await asyncio.gather(*all_extractions, return_exceptions=True)

    for result in results:
        if isinstance(result, Exception):
            logger.error(f"[document:{doc_id}] policy_data_extractor: extraction task raised exception: {result}")
            continue
        for key, val in result.items():
            if key == "token_consumptions" and val:
                state.extraction_cost.extend(val)
            elif key == "_error":
                logger.error(f"[document:{doc_id}] policy_data_extractor: extraction failed for a lob: {val}")
            elif key and key != "token_consumptions":
                final_result[key] = val

    state.extracted_policy_data = final_result
    logger.info(f"[document:{doc_id}] policy_data_extractor: all extractions done | result keys={list(final_result.keys())}")

    # Save Response
    # await save_json(file_path="extraction_result.json", data=final_result)
    # logger.info("Policy data extraction result saved to file")

    return state
