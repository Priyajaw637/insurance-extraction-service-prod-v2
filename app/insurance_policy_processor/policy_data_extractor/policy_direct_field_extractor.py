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
from app.logging_config import get_logger

logger = get_logger(__name__)


async def extract_policy_direct_fields(
    pdf_path: str,
    parent_key: str,
    extraction_schema: dict,
    page_numbers: list,
    original_pages: list,
    tmp_dir: str,
    extraction_instruction: str,
    is_first_level: bool = False,
    schema_context: dict = {},
):
    if not page_numbers:
        logger.info(f"No pages provided for direct field extraction of {parent_key}")
        return {
            parent_key: None,
            "token_consumptions": [],
        }

    try:
        if not is_first_level:
            new_pdf_path = os.path.join(tmp_dir, f"{uuid4()}.pdf")
            new_pdf_path = await filter_policy_pdf_pages(
                pdf_path=pdf_path, page_numbers=page_numbers, output_path=new_pdf_path
            )
        else:
            new_pdf_path = pdf_path

        new_pdf_file_bytes = await pdf_to_bytes(pdf_path=new_pdf_path)
        output_schema = create_gemini_schema(descriptive_schema=extraction_schema)

        if (
            schema_context
            and isinstance(schema_context, dict)
            and len(schema_context) == 1
        ):
            first_parent = list(schema_context.keys())[0]
            schema_context_str = f"""
PARENT FIELD CONTEXT: The provided (`{parent_key}`) list is a sub-field of inside the parent field `{first_parent}`. So keep this in mind when extracting data.
"""
        else:
            schema_context_str = ""

        system_prompt = f"""
You are a precise Insurance Policy Data Extractor specializing in commercial insurance documents. Your sole task is to extract and populate a single field in a JSON schema using only the provided policy document.

# EXTRACTION RULES:
1. **Source Restriction:** Use only the policy document text. Do not infer, calculate, or use external knowledge.
2. **Field Description Priority:** Use the field's description as your main guide; field name is secondary.
3. **Type Casting:** All extracted values must be strings, regardless of the described type (e.g., "false", "1234", "45.67").
4. **Value Purity:** Return only the raw value as found, without added explanation.
    - **Correct:** `{{"policy_number": "A-54321"}}`
    - **Incorrect:** `{{"policy_number": "The policy number is A-54321"}}`
    - **Exception:** If the description requests context, include it as part of the value. (e.g., a field asking for a summary).
5. **Missing Data:** If the value is not found, use the default from the response schema:
    - For booleans, use "false".
    - For other types, use the schema's default. (e.g., "" for strings).

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
        """

        user_prompt = f"""
TASK: Extract the exact value for the field in the JSON schema below from the attached PDF. Use only the policy document text and follow the field's description for context and data type.

DOCUMENT: The PDF contains only the most relevant pages for this field.

OUTPUT: Return a single, valid JSON object matching the schema. No extra text.

{schema_context_str}
    """

        task_id = str(uuid4())
        model_name = GeminiModels.FLASH_LITE.value

        original_contents = [
            types.Part.from_bytes(data=new_pdf_file_bytes, mime_type="application/pdf"),
            types.Part.from_text(text=user_prompt),
        ]

        config_dict = {
            "system_instruction": system_prompt,
            "response_mime_type": "application/json",
            "response_schema": output_schema,
            "temperature": 0.0,
            "seed": 28,
            "thinking_config": {"thinking_budget": -1},
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

        return {
            parent_key: (
                extracted_data.get(parent_key) if extracted_data is not None else None
            ),
            "token_consumptions": all_token_consumptions,
        }
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(
            f"Error in direct extraction for {parent_key} at (Line {lineno}): {str(e)}",
            exc_info=True,
        )
        return {
            parent_key: None,
            "token_consumptions": [],
        }
