import json
from uuid import uuid4
import time
from google.genai import types
from app.insurance_policy_processor.policy_document_handler.policy_file_manager import (
    upload_policy_file,
)
from app.insurance_policy_processor.policy_orchestrator.enums import (
    GeminiModels,
    MANDATORY_FIELDS,
)
from app.insurance_policy_processor.policy_orchestrator.models.state_models import (
    InsuranceDocumentState,
    LobIdentifierResponse,
    FormTypeIdentifierResponse,
)
from app.insurance_policy_processor.policy_orchestrator.modules.gemini import (
    gemini_client,
)
from app.modules.gemini_processor import (
    get_gemini_result_with_retry,
    submit_gemini_task,
)
from app.utils.core import clean_response, create_output_schema
from app.logging_config import get_logger
import sys
from app.utils.helpers import save_json, load_json

logger = get_logger("lob_identifier")


async def get_lob_names(state: InsuranceDocumentState, file_name: str):
    logger.info("LOB Identifier Started...")
    # final_result = await load_json(file_path="final_result.json")

    # return {
    #     "identified_lobs": final_result.get("identified_lobs"),
    #     "identified_forms_type": final_result.get("identified_forms_type"),
    # }

    # Upload main File
    full_pdf_file = state.uploaded_pdf

    lobs_description = await load_json(file_path=f"{file_name.lower()}.json")

    system_prompt = f"""
# ROLE
You are an expert Commercial Insurance Policy Analyst. Your sole purpose is to audit insurance policy documents to identify the active Lines of Business (LOB) and the Types of Forms/Endorsements present.

---

# EXTRACTION GUIDELINES

You must follow a strict "Index First" approach to ensure accuracy.

## 1. Identifying Lines of Business (LOB) covered by this Policy Document

- **Task:** Identify all Lines of Business (LOB) which are covered by this policy.
- **Primary Source (Priority):**
    - Look for **Common Declaration Page / Premium Summary Page** pages.
    - This page is usually present in start of the PDF-Document. Containing the high-level summary of all the coverages and their premiums.
    - It may be titled differently for example: "Common Policy Declaration", "Commercial Policy Common Declaration", "Schedule of Coverages", "Coverages Summary", etc.
    - This single page lists all active coverages with their associated premiums and breakdowns.
    - It is usually found at the beginning of the document (before < 30% of the document).
    - This only includes a high-level coverage breakdown of the policy.
- **Secondary Source:**
    - Only if NO summary page exists, look for individual "Declaration" for each LOB.
    - This will contain a more detailed breakdown of each LOB, including premiums, coverages, deductibles, etc.
- **Logic:**
    - If a Summary page exists, IGNORE individual declaration pages for identification purposes. Use only the Summary.
- **No-Coverage / Not Included:**:
    - If a Declaration page clearly states/indicates that LOB is not covered in the policy then it is confirmed.
    - Look for if for terms like "no coverage" or "not included", "Coverage - Nil", etc.
- **Mapping:**
    - Compare identified LOBs against the provided "Known LOB List".
    - Map the identified LOBs to the closest canonical LOB name from the list.
- If their is an ambiguity or if you are unsure about a LOB, its best to add it to the list and map it to the closest canonical LOB. As once the LOB is excluded at this stage, it is completely ignored in further analysis.

## 2. Identifying Forms and Endorsements Type

- From the policy document indentify which of below type of Forms/Endorsements are present.
    - "Index_Schedule_Type_Forms": When Forms/Endorsements are listed in an index/schedule type format.
    - "Individual_Pages_Type_Forms": When instead of an index/schedule, document contains individual pages for each Form/Endorsement.
    - When both Index type forms along with individual pages are present, return the "Index_Schedule_Type_Forms".
- **Primary Source (Priority):**
    - Look for a "Schedule of Forms,", "Forms list", "Forms Index", "Forms and Endorsements Schedule", "Endorsements Schedule", "Forms Applicable", etc.
    - The best way to verify that a Schedule exists is to look for a table/list like structure where Form Number are listed alongside its Form Name.
    - Usual identifiers for this field are: tabular/listed set of form numbers with form names.
    - These pages usually span MAX of 3 pages and are present at start of the document.
- **Secondary Source:** Only if NO schedule/index exists for forms, scan the document to look for individual form footers/headers.
- Do not confuse between a general (Index/table of contents) and a Schedule of Forms.
- **Logic:** Based on your analysis, return the type of forms and endorsements present.

---

# RESPONSE FORMAT
You must return a valid JSON object with no additional markdown formatting
"""

    user_prompt = f"""
# Context Data
Below is the list of standard Lines of Business (LOBs) we support, followed by the full text of the insurance policy.

<known_lobs_list>
{lobs_description}
</known_lobs_list>

# Task
Analyze the attached policy document according to the System Instructions.
1. Locate the LOBs covered using the Premium Summary or Declaration table if present else identify from scaning the document.
2. Then map which LOBs are matching from the <known_lobs_list>
2. Locate the Schedule of Forms. Extract the type of forms and endorsements.
3. Return the JSON output.
"""

    task_id = str(uuid4())
    model_name = GeminiModels.FLASH.value

    original_contents = [
        full_pdf_file,
        types.Part.from_text(text=user_prompt),
    ]

    config_dict = {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "response_schema": LobIdentifierResponse.model_json_schema(),
        "temperature": 0.0,
        "top_p": 1,
        "top_k": 0,
        "seed": 93,
        "thinking_config": {"thinking_budget": 24000},
    }

    token_response = await gemini_client.aio.models.count_tokens(
        contents=original_contents, model=model_name
    )
    estimated_tokens = token_response.total_tokens

    await submit_gemini_task(
        task_id=task_id,
        model=model_name,
        contents=[],
        config=config_dict,
        estimated_tokens=estimated_tokens,
        file_path=state.pdf_path,
        file_processing="upload",
        mime_type="application/pdf",
        text_contents=[user_prompt],
    )
    response = await get_gemini_result_with_retry(task_id)

    if response and response.usage_metadata:
        usage_info = response.usage_metadata
        state.coverage_mapping_cost.append(
            {
                "model": model_name,
                "input_tokens": usage_info.prompt_token_count or 0,
                "output_tokens": usage_info.candidates_token_count or 0,
                "cached_tokens": usage_info.cached_content_token_count or 0,
                "thinking_tokens": usage_info.thoughts_token_count or 0,
            }
        )

    if response and (response.parsed is not None or response.text is not None):
        if response.parsed:
            return response.parsed
        else:
            doc_id = state.policy_document_id
            logger.warning(f"[document:{doc_id}] lob_identifier: Gemini returned raw text (not parsed) — attempting clean_response. raw={response.text[:500] if response.text else 'None'}")
            return await clean_response(response.text)

    doc_id = state.policy_document_id
    logger.warning(f"[document:{doc_id}] lob_identifier: Gemini returned no response")
    return {}


async def get_only_form_type(state: InsuranceDocumentState):
    logger.info("Form Type Identifier Started...")

    full_pdf_file = state.uploaded_pdf

    system_prompt = f"""
# ROLE
- You are an expert Commercial Insurance Policy Analyst.
- Your sole purpose is to audit insurance policy documents to identify the types of Forms/Endorsements present in the policy document.

---

# GUIDELINES

- From the policy document indentify which of below type of Forms/Endorsements are present.
    - "Index_Schedule_Type_Forms": When Forms/Endorsements are listed in an index/schedule type format.
    - "Individual_Pages_Type_Forms": When instead of an index/schedule, document contains individual pages for each Form/Endorsement.
    - When both Index type forms along with individual pages are present, return the "Index_Schedule_Type_Forms".
- **Primary Source (Priority):**
    - Look for a "Schedule of Forms,", "Forms list", "Forms Index,", "Forms and Endorsements Schedule", "Endorsements Schedule", "Forms Applicable", etc.
    - The best way to verify that a Schedule exists is to look for a table/list like structure where Form Number are listed alongside its Form Name.
    - Usual identifiers for this field are: tabular/listed set of form numbers with form names/description.
    - These pages usually span MAX of 3 pages and are present at start of the document.
- **Secondary Source:** Only if NO schedule/index exists for forms, scan the document to look for individual form footers/headers.
- Do not confuse between a general (Index/table of contents) and a Schedule of Forms.
- **Logic:** Based on your analysis, return the type of forms and endorsements present.

---

# RESPONSE FORMAT
You must return a valid JSON object with no additional markdown formatting
"""

    user_prompt = f"""
Based on the attached policy document, identify if the forms and endorsements are listed in Index or individual pages.
"""

    task_id = str(uuid4())
    model_name = GeminiModels.FLASH_LITE.value

    original_contents = [
        full_pdf_file,
        types.Part.from_text(text=user_prompt),
    ]

    config_dict = {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "response_schema": FormTypeIdentifierResponse.model_json_schema(),
        "temperature": 0.0,
        "top_p": 1,
        "top_k": 0,
        "seed": 45,
    }

    token_response = await gemini_client.aio.models.count_tokens(
        contents=original_contents, model=model_name
    )
    estimated_tokens = token_response.total_tokens

    await submit_gemini_task(
        task_id=task_id,
        model=model_name,
        contents=[],
        config=config_dict,
        estimated_tokens=estimated_tokens,
        file_path=state.pdf_path,
        file_processing="upload",
        mime_type="application/pdf",
        text_contents=[user_prompt],
    )
    response = await get_gemini_result_with_retry(task_id)

    if response and response.usage_metadata:
        usage_info = response.usage_metadata
        state.coverage_mapping_cost.append(
            {
                "model": model_name,
                "input_tokens": usage_info.prompt_token_count or 0,
                "output_tokens": usage_info.candidates_token_count or 0,
                "cached_tokens": usage_info.cached_content_token_count or 0,
                "thinking_tokens": usage_info.thoughts_token_count or 0,
            }
        )

    if response and (response.parsed is not None or response.text is not None):
        if response.parsed:
            return response.parsed
        else:
            return await clean_response(response.text)

    logger.warning("Form Type Identifier - no response received")
    return {}


async def get_lob_name_and_form_type(state: InsuranceDocumentState):
    doc_id = state.policy_document_id
    try:
        logger.info(f"[document:{doc_id}] Node: lob_identifier — started")
        t1 = time.perf_counter()

        if state.line_of_business == "Personal" and state.country == "US":
            logger.info(f"[document:{doc_id}] Node: lob_identifier — skipped (US Personal)")
            return state

        if state.list_lobs and state.tool_name == "policy_checking":
            logger.info(f"[document:{doc_id}] Node: lob_identifier — running form type only (lobs pre-supplied: {state.list_lobs})")
            form_type_result = await get_only_form_type(state=state)
            state.identified_forms_type = form_type_result.get(
                "identified_forms_type", "Index_Schedule_Type_Forms"
            )
            state.identified_lobs = state.list_lobs
            logger.info(f"[document:{doc_id}] Node: lob_identifier — form_type={state.identified_forms_type}")
            return state
        elif state.list_lobs and state.tool_name == "proposal_generation":
            logger.info(f"[document:{doc_id}] Node: lob_identifier — skipped (proposal with pre-supplied lobs)")
            return state

        file_name = f"{state.country}_lobs"
        logger.info(f"[document:{doc_id}] Node: lob_identifier — calling Gemini for LOB identification")

        lob_identifier_result = await get_lob_names(state=state, file_name=file_name)

        state.identified_lobs = lob_identifier_result.get("identified_lobs")
        state.identified_forms_type = lob_identifier_result.get("identified_forms_type")

        logger.info(f"[document:{doc_id}] Node: lob_identifier — identified_lobs={state.identified_lobs} form_type={state.identified_forms_type}")

        final_schema = {}
        for lob in state.insurance_extraction_schema.keys():
            if lob in state.identified_lobs:
                final_schema[lob] = state.insurance_extraction_schema[lob]

        if final_schema:
            for key in MANDATORY_FIELDS:
                if key in state.insurance_extraction_schema.keys():
                    final_schema[key] = state.insurance_extraction_schema[key]
                    if key not in state.identified_lobs:
                        state.identified_lobs.append(key)
            state.insurance_extraction_schema = final_schema

        elapsed = time.perf_counter() - t1
        logger.info(f"[document:{doc_id}] Node: lob_identifier — completed in {elapsed:.2f}s | final schema keys={list(state.insurance_extraction_schema.keys())}")

        return state
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(f"[document:{doc_id}] Node: lob_identifier — FAILED at line {lineno}: {e}", exc_info=True)
        raise e
