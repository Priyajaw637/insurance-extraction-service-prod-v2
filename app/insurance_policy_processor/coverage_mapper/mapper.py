import json
from uuid import uuid4
import time
from google.genai import types

from app.insurance_policy_processor.policy_document_handler.policy_file_manager import (
    upload_policy_file,
)
from app.insurance_policy_processor.policy_orchestrator.enums import GeminiModels
from app.insurance_policy_processor.policy_orchestrator.models.state_models import (
    InsuranceDocumentState,
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
from app.insurance_policy_processor.policy_orchestrator.enums_new import (
    CUSTOM_PROMPT_FIRST,
    CUSTOM_PROMPT_SECOND,
    CUSTOM_PROMPT_EXTRACTION,
)

logger = get_logger("coverage_mapper")


async def get_coverage_mapping(state: InsuranceDocumentState):
    logger.info("Coverage Mapping Started...")
    logger.info(f"Extracting Mappings for:  {state.insurance_extraction_schema.keys()}")

    # Upload main File
    full_pdf_file = state.uploaded_pdf
    # print("Full PDF File in get_coverage_mapping: ", full_pdf_file)

    # Create Output Schema for prompt
    output_schema = await create_output_schema(
        full_schema=state.insurance_extraction_schema
    )

    # System Prompt
    system_prompt = f"""
# ROLE:
- You are a specialized Insurance Document Analyst with expertise in ubderstanding policy document and map relevant pages that contains data to the individual Lines of Business. 
- Your primary goal is to analyze the provided insurance policy document and map its pages to the **LOB (Lines of Business)** mentioned in the `insurance_schema` provided by the user.

---

# INPUTS:
1. **PDF Document**: The policy document that is to be analyzed.
2. **insurance_schema**: The JSON schema to be used for classification. The first-level keys represent the LOB names, and each LOB will have nested level of fields.

---

# TASK:

### Primary Objective: 
- STEP 1: First Analyze the policy document content.
- STEP 2: Understand the structure of each LOB from the `insurance_schema`.
- STEP 3: Then identify which LOB's from the `insurance_schema` are covered by this policy document.
- STEP 4: Then classify pages against each identified **LOB Names**.
- STEP 5: For each identified LOB, provide a list of page numbers that contains values/data against that LOB Schema.

### Page Classification Criteria:
- Base the classification on the semantic probability of finding the *values* for fields from `insurance_schema` for that LOB, not merely on a string match of the field's name.
- Do not expect document to have exact matches for fields. Document may have different wordings, aliases, or variations of the same field. Use your insurance-industry knowledge to make best effort to find matches.

### Mapping Tabular Data:
- When content is in a tabular format, treat the entire table as a single logical unit for mapping.
- Map all pages containing table from start to end, even if it extends across multiple pages.

### NO Forced LOB Matches:
If no pages are relevant to a LOB, return an empty list `[]` for that field. Do not force a classification.
 
---

{state.first_level_instruction}

---

# DOCUMENT STRUCTURE:

- **Type of Policy Document**: The policy document could be in one of the following formats:
  - **Merged Document**: The document is a collection of individual policy documents merged into a single file.
  - **Single Continuous Document**: The document is a single file from start to end.
- **What to Do in both Cases**: 
  - In both cases, treat the provided policy document as a new, self-contained file.
  - Completely discard any page numbers written within the pages of the policy document's and strictly follow `Page Mapping Rules` below.

---

# PAGE MAPPING RULES:

- **Use Absolute Page Order**: You are required to use the policy document's **actual, sequential page number** for mapping. The first page of the PDF is page 1, the second is page 2, and so on.
- **Strictly Ignore All Internal Page Numbers**: Completely discard any page numbers written within the policy document's content. This includes:
  - Page numbers in headers or footers.
  - Page numbers found within a Table of Contents.
- **Rationale**: These internal numbers are unreliable due as policy document can be a merged document and must not be used in your output.
- **Example**: If the document's 25th physical page has "Page 63" printed in its footer, then the correct page number to use in output will be **25** and not **63**.
- **Final Output Rule**: All page numbers in your final mapping must correspond exclusively to the sequential order of pages from the beginning to the end of the policy document.

---

# OUTPUT FORMAT:
- Return your response as a JSON object, with mapping of page numbers for each **LOB** field.
---
"""

    user_prompt = ""
    if state.list_lobs:
        lob_list = ", ".join(state.list_lobs)

        user_prompt = f"""
Carefully analyse the provided policy document and generate a page-wise-mapping against each LOB key from the below schema.

NOTE: This is confirm that this policy document contains all these following LOBs: {lob_list}. So make sure to map all these LOBs.
"""
    else:
        user_prompt = f"""
Carefully analyse the provided policy document and generate a page-wise-mapping against identified LOB (Lines of Business) from the below schema.
"""

    user_prompt += f"""

<insurance_schema>

{json.dumps(state.insurance_extraction_schema, indent=2)}

</insurance_schema>
"""

    task_id = str(uuid4())
    model_name = GeminiModels.FLASH.value

    original_contents = [
        full_pdf_file,
        # types.Part.from_text(text=schema_prompt),
        types.Part.from_text(text=user_prompt),
    ]

    config_dict = {
        "system_instruction": system_prompt,
        "response_mime_type": "application/json",
        "response_schema": output_schema,
        "temperature": 0.0,
        "top_p": 1,
        "top_k": 0,
        "seed": 77,
        "thinking_config": {
            "thinking_budget": 24000,
            # "include_thoughts": True,
        },
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
        text_contents=[
            # schema_prompt,
            user_prompt
        ],
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

    logger.warning("Coverage mapping failed - no response received")
    return {}


async def map_coverage_sections(state: InsuranceDocumentState):
    doc_id = state.policy_document_id
    try:
        # Get Stored Response
        # final_result = await load_json(file_path="final_result.json")
        # state.coverage_mapping_result = final_result.get("coverage_mapping_result")
        # logger.info("Coverage mapping result loaded from file")

        # return state

        # SET custom prompt for each level
        if state.country == "US" and state.line_of_business == "Personal":
            custom_prompt_key = "US_Personal"
        else:
            custom_prompt_key = f"{state.country}_{state.line_of_business}_{state.identified_forms_type}"

        state.first_level_instruction = CUSTOM_PROMPT_FIRST.get(custom_prompt_key)
        state.second_level_instruction = CUSTOM_PROMPT_SECOND.get(custom_prompt_key)
        state.extraction_instruction = CUSTOM_PROMPT_EXTRACTION.get(custom_prompt_key)

        logger.info(f"[document:{doc_id}] coverage_mapper: calling Gemini for page mapping | prompt_key={custom_prompt_key}")
        # print("=" * 40)
        # print("\n\n")
        # print("first_level_instruction: ", state.first_level_instruction)
        # print("second_level_instruction: ", state.second_level_instruction)
        # print("extraction_instruction: ", state.extraction_instruction)
        # print("\n\n")
        # print("=" * 40)
        
        t1 = time.perf_counter()
        coverage_mapping = await get_coverage_mapping(state=state)
        state.coverage_mapping_result = coverage_mapping
        elapsed = time.perf_counter() - t1

        mapped_lobs = list(coverage_mapping.keys()) if isinstance(coverage_mapping, dict) else []
        logger.info(f"[document:{doc_id}] coverage_mapper: completed in {elapsed:.2f}s | mapped_lobs={mapped_lobs}")

        return state
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(f"[document:{doc_id}] coverage_mapper: FAILED at line {lineno}: {e}", exc_info=True)
        raise e
