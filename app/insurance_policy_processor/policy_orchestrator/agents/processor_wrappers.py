import sys

from app.insurance_policy_processor.coverage_detail_mapper.mapper import (
    map_coverage_details,
)
from app.insurance_policy_processor.coverage_mapper.mapper import (
    map_coverage_sections,
)
from app.insurance_policy_processor.policy_data_extractor.policy_extractor import (
    extract_policy_data,
)
from app.insurance_policy_processor.policy_orchestrator.models.state_models import (
    InsuranceDocumentState,
)
from app.insurance_policy_processor.processing_cost_calculator.metrics_calculator import (
    calculate_insurance_cost,
)
from app.logging_config import get_logger
from app.utils.helpers import save_json, load_json

logger = get_logger(__name__)


async def coverage_mapping_wrapper(state: InsuranceDocumentState):
    doc_id = state.policy_document_id
    try:
        # return state
        logger.info(f"[document:{doc_id}] Node: coverage_mapper — started | lobs={list(state.insurance_extraction_schema.keys())}")
        state = await map_coverage_sections(state=state)
        logger.info(f"[document:{doc_id}] Node: coverage_mapper — completed")
        return state
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(f"[document:{doc_id}] Node: coverage_mapper — FAILED at line {lineno}: {e}", exc_info=True)
        raise e


async def coverage_detail_mapping_wrapper(state: InsuranceDocumentState):
    doc_id = state.policy_document_id
    try:
        # return state
        logger.info(f"[document:{doc_id}] Node: coverage_detail_mapping — started")
        state = await map_coverage_details(state=state)
        logger.info(f"[document:{doc_id}] Node: coverage_detail_mapping — completed")
        return state
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(f"[document:{doc_id}] Node: coverage_detail_mapping — FAILED at line {lineno}: {e}", exc_info=True)
        raise e


async def policy_data_extraction_wrapper(state: InsuranceDocumentState):
    doc_id = state.policy_document_id
    try:
        # return state
        logger.info(f"[document:{doc_id}] Node: policy_data_extractor — started")
        state = await extract_policy_data(state=state)
        extracted_keys = list(state.extracted_policy_data.keys()) if state.extracted_policy_data else []
        logger.info(f"[document:{doc_id}] Node: policy_data_extractor — completed | extracted lobs={extracted_keys}")
        return state
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(f"[document:{doc_id}] Node: policy_data_extractor — FAILED at line {lineno}: {e}", exc_info=True)
        raise e


async def processing_cost_calculator(state: InsuranceDocumentState):
    doc_id = state.policy_document_id
    try:
        # return state
        logger.info(f"[document:{doc_id}] Node: processing_cost_calculator — started")
        state = await calculate_insurance_cost(state=state)
        logger.info(f"[document:{doc_id}] Node: processing_cost_calculator — completed")
 
        # Save Final Response
        # data = state.model_dump(mode="json")
        # await save_json(file_path="final_result_1.json", data=data)
        # logger.info("Final result saved to file")
        # get_cost(data=data)

        return state
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(f"[document:{doc_id}] Node: processing_cost_calculator — FAILED at line {lineno}: {e}", exc_info=True)
        raise e
