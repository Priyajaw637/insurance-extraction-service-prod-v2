import datetime
import os
import shutil
from typing import Any, Dict, List, Optional
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from app.insurance_policy_processor.policy_document_handler.policy_file_manager import (
    delete_policy_file_from_gemini,
)
from app.insurance_policy_processor.policy_orchestrator.agents.processor_wrappers import (
    coverage_detail_mapping_wrapper,
    coverage_mapping_wrapper,
    policy_data_extraction_wrapper,
    processing_cost_calculator,
)
from app.insurance_policy_processor.policy_orchestrator.models.state_models import (
    InsuranceDocumentState,
)
from app.modules.async_redis import redis_client
from app.logging_config import get_logger
from app.insurance_policy_processor.policy_document_handler.policy_file_manager import (
    upload_policy_file,
)
from app.insurance_policy_processor.lob_indentifier.lob_indentifier import (
    get_lob_name_and_form_type,
)

logger = get_logger(__name__)


def serialize_datetime_objects(data: Any) -> Any:
    """Recursively convert datetime objects to ISO format strings for JSON serialization."""
    if isinstance(data, datetime.datetime):
        return data.isoformat()
    elif isinstance(data, datetime.date):
        return data.isoformat()
    elif isinstance(data, datetime.time):
        return data.isoformat()
    elif isinstance(data, dict):
        return {k: serialize_datetime_objects(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [serialize_datetime_objects(item) for item in data]
    else:
        return data


async def insurance_document_router(state: InsuranceDocumentState) -> str:
    if state.pdf_length < 100:
        return "coverage_mapper"
    else:
        return "structural_analysis"


async def create_insurance_workflow() -> StateGraph:
    """Create and configure the LangGraph workflow for insurance document processing."""
    logger.info("Creating insurance Document Processing workflow")
    workflow = StateGraph(InsuranceDocumentState)

    workflow.add_node("lob_identifier", get_lob_name_and_form_type)
    workflow.add_node("coverage_mapper", coverage_mapping_wrapper)
    workflow.add_node("coverage_detail_mapping", coverage_detail_mapping_wrapper)
    workflow.add_node("policy_data_extractor", policy_data_extraction_wrapper)
    workflow.add_node("processing_cost_calculator", processing_cost_calculator)

    workflow.set_entry_point("lob_identifier")

    workflow.add_edge("lob_identifier", "coverage_mapper")
    workflow.add_edge("coverage_mapper", "coverage_detail_mapping")
    workflow.add_edge("coverage_detail_mapping", "policy_data_extractor")
    workflow.add_edge("policy_data_extractor", "processing_cost_calculator")
    workflow.add_edge("processing_cost_calculator", END)

    memory = MemorySaver()
    logger.info("LangGraph workflow created successfully")
    return workflow.compile(checkpointer=memory)


async def process_insurance_document(
    pdf_path: str,
    insurance_extraction_schema: Dict[str, Any],
    tmp_dir="/tmp",
    line_of_business: str = "Commercial",
    country: str = "US",
    policy_document_id="",
    list_lobs: Optional[List[str]] = None,
    tool_name: Optional[str] = None,
):
    try:
        logger.info(
            f"[document:{policy_document_id}] Workflow started | lob={line_of_business} country={country} tool={tool_name}"
        )

        workflow = await create_insurance_workflow()
        id_val = uuid4()
        config = {"configurable": {"thread_id": f"insurance_{id_val}"}}

        path = os.path.join(tmp_dir, str(id_val))
        if not os.path.exists(path=path):
            os.makedirs(name=path)

        # Upload main File
        logger.info(f"[document:{policy_document_id}] Uploading PDF to Gemini")
        full_pdf_file = await upload_policy_file(
            pdf_path=pdf_path, mime_type="application/pdf"
        )
        logger.info(f"[document:{policy_document_id}] PDF uploaded successfully")

        new_schema = {}
        # Remove US personal from US commercial
        if line_of_business == "Commercial" and country == "US":
            insurance_extraction_schema.pop("homeowners", None)
            insurance_extraction_schema.pop("personal_auto", None)
        elif line_of_business == "Personal" and not list_lobs and tool_name == "policy_checking":
            new_schema = {
                "general": insurance_extraction_schema.get("general", None),
                "homeowners": insurance_extraction_schema.get("homeowners", None),
                "personal_auto": insurance_extraction_schema.get("personal_auto", None),
            }
        
        elif line_of_business == "Personal" and not list_lobs and tool_name == "proposal_generation":
            new_schema = {
                "insured_profile": insurance_extraction_schema.get("insured_profile", None),
                "homeowners": insurance_extraction_schema.get("homeowners", None),
                "personal_auto": insurance_extraction_schema.get("personal_auto", None),
            }

        initial_state = InsuranceDocumentState(
            pdf_path=pdf_path,
            tmp_dir=path,
            insurance_extraction_schema=new_schema or insurance_extraction_schema,
            policy_document_id=policy_document_id,
            list_lobs=list_lobs,
            line_of_business=line_of_business,
            country=country,
            tool_name=tool_name,
            uploaded_pdf=full_pdf_file,
        )

        logger.info(f"[document:{policy_document_id}] LangGraph workflow invoking — nodes: lob_identifier → coverage_mapper → coverage_detail_mapping → policy_data_extractor → processing_cost_calculator")
        final_state = await workflow.ainvoke(initial_state, config)
        logger.info(f"[document:{policy_document_id}] LangGraph workflow completed successfully")
        state = InsuranceDocumentState(**final_state)

        await redis_client.create(
            key=policy_document_id,
            value={
                "extracted_policy_data": state.extracted_policy_data,
                "token_usage": state.token_consumption,
            },
        )

        state_data = serialize_datetime_objects(state.model_dump())

        # Publish completion message to Redis channel
        channel_name = f"insurance_document_processing_complete:{policy_document_id}"
        await redis_client.publish(
            channel_name,
            {
                "status": "completed",
                "policy_document_id": policy_document_id,
                "data": state.extracted_policy_data,
                "token_usage": state.token_consumption,
                "state_data": state_data,
            },
        )

        return state
    except Exception as e:
        error_message = str(e)
        logger.error(
            f"Insurance document processing workflow execution failed: {error_message}",
            exc_info=True,
        )
        logger.error(f"[document:{policy_document_id}] Workflow FAILED: {error_message}")
        logger.info(f"Saving empty result in Redis at key: '{policy_document_id}'")
        await redis_client.create(key=policy_document_id, value={})

        # Publish task failure to Redis
        await redis_client.publish_task_failure(
            task_id=policy_document_id,
            error_message=error_message,
            additional_data={
                "policy_document_id": policy_document_id,
                "data": {},
                "token_usage": 0,
            },
        )

        return initial_state
    finally:
        # pass
        if os.path.exists(path=path):
            shutil.rmtree(path=path)

        try:
            # Use initial_state instead of state in finally block
            if hasattr(initial_state, "uploaded_pdf") and initial_state.uploaded_pdf:
                logger.info(
                    f"Cleaning up uploaded PDF: {initial_state.uploaded_pdf.name}"
                )
                await delete_policy_file_from_gemini(initial_state.uploaded_pdf.name)
        except Exception as e:
            logger.warning(f"Failed to delete uploaded PDF: {str(e)}")
