import json
import os
import sys
from uuid import uuid4

from google.genai import types

from app.insurance_policy_processor.policy_document_handler.policy_page_extractor import (
    filter_policy_pdf_pages,
)
from app.insurance_policy_processor.policy_orchestrator.enums import GeminiModels
from app.insurance_policy_processor.policy_orchestrator.modules.gemini import (
    gemini_client,
)
from app.logging_config import get_logger


from app.modules.gemini_processor import (
    get_gemini_result_with_retry,
    submit_gemini_task,
)
from app.utils.core import (
    clean_response,
    create_gemini_schema,
    map_relative_pages_to_original,
    pdf_to_bytes,
)

logger = get_logger(__name__)


async def extract_policy_object_fields(
    pdf_path: str,
    parent_key: str,
    extraction_schema: dict,
    page_numbers: list,
    original_pages: list,
    tmp_dir: str,
    extraction_instruction: str,
    schema_context: dict = {},
):
    result = {
        parent_key: None,
        "token_consumptions": [],
        "object": True,
    }
    if not page_numbers:
        logger.info(f"No pages provided for object extraction of {parent_key}")
        return result

    try:
        # Upload Pdf File
        new_pdf_path = os.path.join(tmp_dir, f"{uuid4()}.pdf")
        new_pdf_path = await filter_policy_pdf_pages(
            pdf_path=pdf_path, page_numbers=page_numbers, output_path=new_pdf_path
        )
        new_pdf_file_bytes = await pdf_to_bytes(pdf_path=new_pdf_path)

        if (
            schema_context
            and isinstance(schema_context, dict)
            and len(schema_context) == 1
        ):
            first_parent = list(schema_context.keys())[0]
            schema_context_str = f"""
PARENT FIELD CONTEXT: The provided (`{parent_key}`) object is a sub-field of inside the parent field `{first_parent}`. So keep this in mind when extracting data.
"""
        else:
            schema_context_str = ""

        system_prompt = f"""
You are an expert Commercial Insurance Policy Data Extractor. Your task is to fully populate a complex, nested JSON object using only the provided policy document.

# EXTRACTION STRATEGY:
1. **Hierarchical Search:** For each key in the schema (including nested fields and lists), search the entire policy document to find its value. Do not stop after the first match.
2. **Field Type Logic:**
    - For direct fields: Extract the exact value as found.
    - For lists: Extract all relevant items, not just the first.
    - For nested objects: Apply this strategy recursively.
3. **Document Continuity:** Treat all provided pages as a single, continuous text stream. Data may be spread across multiple pages.
4. **Field Description Priority:** Use field descriptions as your main guide; field names are secondary.

# DATA RULES:
1. **Source Restriction:** Use only the policy document text. Do not infer or create data.
2. **Missing Data Defaults:**
    - Boolean fields: Use "false" (as a string).
    - Other fields: Use the schema's default (e.g., "" for strings).
    - Nested objects: If no fields found, return `{{}}`.
    - Lists: If no items found, return `[]`. Do NOT return a list with one object of default/empty values.
3. **Type Casting:** All leaf field values must be strings, regardless of described type (e.g., "false", "1234", "23.4").
4. **Value Purity:** Return only the raw value as found, without added explanation.
    - Exception: If the description explicitly requests context, include it as part of the value. (e.g., a field asking for a summary).

---

{extraction_instruction}

---

# PAGE NUMBER CITATION:
1. **Source Pages:** For every extracted `value`, you MUST populate the corresponding `pages` field with an array of unique integers. These integers represent the page numbers where the information for the `value` was found.
2. **Page Numbering:** Use the absolute page order of the provided PDF. The first page is page 1, the second is page 2, and so on. Ignore any page numbers printed in the policy document's header or footer.
3. **Spanning Pages:** If a single piece of data is derived from information spanning multiple pages, include all relevant page numbers in the array (e.g., `[2, 3]`).
4. **No Data:** If a value is not found in the policy document and you are using a default value, the `pages` array must be empty (`[]`).

# OUTPUT:
Return a single, valid JSON object matching the response schema. No extra text.

---
        """

        user_prompt = f"""
TASK: Extract and fully populate the nested JSON object below using only the attached PDF. For each field (including lists and sub-objects), search all provided pages for relevant information.

DOCUMENT: The PDF contains only pages that are relevance to this schema. Read all pages as a continuous stream-do not miss items that cross page breaks.

OUTPUT: Return a single, valid JSON object matching the schema. No extra text.

{schema_context_str}
    """

        all_token_consumptions = []
        extracted_data = None

        # Create output schema for object
        output_schema = create_gemini_schema(descriptive_schema=extraction_schema)

        task_id = str(uuid4())
        model_name = GeminiModels.FLASH.value

        original_contents = [
            types.Part.from_bytes(data=new_pdf_file_bytes, mime_type="application/pdf"),
            types.Part.from_text(text=user_prompt),
        ]

        config_dict = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": output_schema,
            "temperature": 0.0,
            "top_p": 1,
            "top_k": 0,
            "seed": 32,
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
            file_path=new_pdf_path,
            file_processing="bytes",
            mime_type="application/pdf",
            text_contents=[user_prompt],
        )
        response = await get_gemini_result_with_retry(task_id)

        all_token_consumptions = []
        extracted_data = None

        if response and response.usage_metadata:
            usage_info = response.usage_metadata
            token_consumptions = {
                "model": model_name,
                "input_tokens": usage_info.prompt_token_count or 0,
                "output_tokens": usage_info.candidates_token_count or 0,
                "cached_tokens": (
                    usage_info.cached_content_token_count
                    if usage_info.cached_content_token_count
                    else 0
                ),
                "thinking_tokens": usage_info.thoughts_token_count or 0,
            }
            all_token_consumptions.append(token_consumptions)

        if response and (response.parsed is not None or response.text is not None):
            if response.parsed is not None:
                extracted_data = response.parsed
            else:
                extracted_data = await clean_response(response.text)

        if extracted_data:
            extracted_data = await map_relative_pages_to_original(
                data=extracted_data, original_pages_map=original_pages
            )

        result["token_consumptions"] = all_token_consumptions

        if len(extracted_data.keys()) == 1:
            result[parent_key] = (
                extracted_data.get(parent_key) if extracted_data is not None else None
            )
        else:
            result[parent_key] = extracted_data if extracted_data is not None else {}

        return result
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(
            f"Error in nested extraction for '{parent_key}' at (Line {lineno}): {str(e)}",
            exc_info=True,
        )
        return {parent_key: None, "token_consumptions": []}
