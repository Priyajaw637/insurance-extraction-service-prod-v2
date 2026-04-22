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
    bind_new_pages_to_original,
    clean_response,
    create_gemini_schema,
    create_output_schema,
    map_relative_pages_to_original,
    pdf_to_bytes,
)

logger = get_logger(__name__)


async def extract_policy_list_fields(
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
        logger.info(f"No pages provided for list extraction of {parent_key}")
        return {
            parent_key: [],
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
You are an expert Insurance Policy Data Extractor. Your sole function is to populate a list within a given JSON schema.

# CORE RULES:
1. **List Extraction:** Find and extract **all** relevant items matching the schema's description; do not stop after the first match.
2. **Document Continuity:** Treat all provided pages as a single, continuous text stream. Items may span multiple pages—synthesize across page breaks.
3. **Source Restriction:** Extract data only from the provided policy document; do not infer or use external knowledge.
4. **Field Description Priority:** Use field descriptions as your primary guide; field names are secondary.

---

# LIST EXTRACTION RULES:

## LIST OF OBJECTS:
1. **Type Casting:** All leaf field values in each object must be strings, regardless of the description's suggested type (e.g., "false", "1234", "3.14").
2. **Value Purity:** Extract values exactly as they appear, preserving original formatting. Do not add explanations or extra text.
    - **Correct:** `[{{"policy_number": "A-54321"}}]`
    - **Incorrect:** `[{{"policy_number": "The policy number is A-54321"}}]`
3. **Missing Data in Object:** If a field's value is missing, use the default from the response schema ("false" for booleans, "" for strings).
4. **Missing List Items:** If no items are found, return an empty list `[]`. Do not return a list with one object containing only default values.

## LIST OF STRINGS:
1. **Exact Matches:** Extract values exactly as they appear, preserving formatting and capitalization.
2. Use field descriptions to guide item selection.

---

{extraction_instruction}

---

# PAGE NUMBER CITATION:
1. **Source Pages:** For every extracted `value`, you MUST populate the corresponding `pages` field with an array of unique integers. These integers represent the page numbers where the information for the `value` was found.
2. **Page Numbering:** Use the absolute page order of the provided PDF. The first page is page 1, the second is page 2, and so on. Ignore any page numbers printed in the policy document's header or footer.
3. **Spanning Pages:** If a single piece of data is derived from information spanning multiple pages, include all relevant page numbers in the array (e.g., `[2, 3]`).
4. **No Data:** If a value is not found in the policy document and you are using a default value, the `pages` array must be empty (`[]`).

# OUTPUT FORMAT:
Return a single, valid JSON object matching the response schema. If no items are found, return an empty list for the field.
---
        """

        user_prompt = f"""
TASK: Extract every relevant item from the attached PDF to fully populate the list in the JSON schema below. Items may appear in tables, lists, or within paragraphs, and may span multiple pages.

DOCUMENT: The PDF contains only pages that are relevance to this schema. Read all pages as a continuous stream-do not miss items that cross page breaks.

OUTPUT: Return a single, valid JSON object matching the schema. No extra text.

{schema_context_str}
        """

        all_token_consumptions = []
        extracted_data = None

        # Create output schema for list
        output_schema = create_gemini_schema(descriptive_schema=extraction_schema)

        # print(f"\n\noutput_schema for list {parent_key}: ", output_schema)

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
            "seed": 23,
            # "thinking_config": {"thinking_budget": 0},
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
            all_token_consumptions = [token_consumptions]

        if response and (response.parsed is not None or response.text is not None):
            if response.parsed is not None:
                extracted_data = response.parsed
            else:
                extracted_data = await clean_response(response.text)

        if extracted_data:
            extracted_data = await map_relative_pages_to_original(
                data=extracted_data, original_pages_map=original_pages
            )

        if isinstance(extracted_data, dict):
            return {
                parent_key: extracted_data.get(parent_key, []),
                "token_consumptions": all_token_consumptions,
            }
        elif isinstance(extracted_data, list):
            return {
                parent_key: extracted_data,
                "token_consumptions": all_token_consumptions,
            }

        return {
            parent_key: [],
            "token_consumptions": all_token_consumptions,
        }
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(
            f"Error in list extraction for {parent_key} at (Line {lineno}): {str(e)}",
            exc_info=True,
        )
        return {
            parent_key: [],
            "token_consumptions": [],
        }


async def extract_list_fields_with_new_mapper(
    pdf_path: str,
    parent_key: str,
    extraction_schema: dict,
    page_numbers: list,
    tmp_dir: str,
    original_pages: list,
    extraction_instruction: str,
    schema_context: dict = {},
):
    if not page_numbers:
        logger.info(f"No pages provided for new mapper list extraction of {parent_key}")
        return {
            parent_key: [],
            "token_consumptions": [],
            "object": False,
        }
    try:
        # print(f"Path at mapper for {parent_key}: ", pdf_path)

        # Upload Pdf File
        new_pdf_path = os.path.join(tmp_dir, f"{uuid4()}.pdf")
        new_pdf_path = await filter_policy_pdf_pages(
            pdf_path=pdf_path, page_numbers=page_numbers, output_path=new_pdf_path
        )
        new_pdf_file_bytes = await pdf_to_bytes(pdf_path=new_pdf_path)

        # print(f"New Pdf Path at mapper for {parent_key}: ", new_pdf_path)

        output_schema = await create_output_schema(full_schema=extraction_schema)

        system_prompt = """
You are a highly efficient Document Relevance Analyzer. Your sole function is to identify which pages in a policy document contain information relevant to a specific data extraction schema. You are a pre-processor; you do not extract any data yourself.

## CRITICAL INSTRUCTION:
### Understanding the Document's Structure

- **Context**: The PDF you are analyzing can also be a filtered pdf from a large policy document. As a result, the pages provided may appear non-sequential based on the numbers printed in headers or footers (e.g., page 100 might be followed by page 200).
- **Core Instruction**: Treat the provided policy document as a new, self-contained file. Your task is to map content based on the page order within this specific file, not the original source policy document.

### Page Numbering Rule

- **Use Absolute Page Order**: You are required to use the policy document's **actual, sequential page number** for mapping. The first page of the PDF is page 1, the second is page 2, and so on.
- **Strictly Ignore Printed Page Numbers**: Completely disregard any page numbers written within the policy document's headers and footers. These are unreliable due to policy document merging and must not be used in your output.
- **Example**: If the policy document's 25th physical page has "Page 8" printed in its footer, the correct page number for your mapping is **25**.
- **Final Output Rule**: All page numbers in your final mapping must correspond exclusively to the sequential order of pages from the beginning to the end of the policy document.

# CORE MISSION: PAGE IDENTIFICATION
1.  **Schema-Driven Search:** Your task is guided by the fields described in the user's JSON schema. Read the schema first to understand what kind of information you are looking for.
2.  **Content Keyword Analysis:** Scan each page of the policy document for keywords, phrases, table headers, or contextual clues that are directly related to the fields in the schema. For example, if the schema asks for `"building_limit"`, look for pages containing tables or text about "coverage," "limits," "schedules," or specific building identifiers.
3.  **Holistic Page Evaluation:** A page is considered relevant if it contains ANY information that would help fill out ANY part of the provided schema. It does not need to contain all the information. The presence of just one relevant data point makes a page relevant.
4.  **No Data Extraction:** Your only job is to output a list of relevant page numbers. You MUST NOT attempt to extract the data or fill the schema. You are only identifying the *location* of the data.

# RATIONALE FOR PAGE IDENTIFICATION:
- The current extraction engine (a separate agent, that runs after you) is capable of accurately extracting data from a page if it provided with relevant pages to the schema.
- By filtering out irrelevant pages, you help the extraction engine focus on a smaller set of pages, improving both speed and accuracy.
- Now, while filtering keep in mind that if you miss a relevant page, the extraction engine will not be able to extract that data at all. So, it is better to be slightly over-inclusive rather than miss a critical page.

# OUTPUT REQUIREMENTS:
1.  **Strict JSON Format:** Your response MUST be a single, valid JSON object and nothing else.
2.  The value for this key must be a JSON array of unique integers, representing the page numbers that contain relevant information. The page numbers must be sorted in ascending order.
3.  **Handling No Matches:** If you analyze the entire policy document and find no pages relevant to the schema, the value for `"relevant_pages"` MUST be an empty list (`[]`).
4.  **No Extra Text:** Do NOT include any text, explanations, or markdown formatting.
    """

        user_prompt = f"""
TASK: Analyze the attached PDF policy document and identify all pages that contain information relevant to the fields in the JSON schema below. Your goal is to create a list of page numbers to be used by a subsequent extraction agent.

---

# JSON SCHEMA TO FIND DATA FOR:
Use the fields in this schema to guide your search for relevant pages.

```
{json.dumps(extraction_schema, indent=2)}
```

Return the completed JSON object containing the list of relevant page numbers.
---
    """
        result = {}
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
            "top_p": 1,
            "top_k": 0,
            "seed": 25,
            "thinking_config": {"thinking_budget": 0},
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

        # logger.info(f"Completed new mapper processing for '{parent_key}'")

        final_result = None
        all_token_consumptions = []

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
            if response.parsed:
                final_result = response.parsed
            else:
                final_result = await clean_response(response.text)
        # print("Final Result from list mapper: ", final_result)

        new_pages = final_result.get(parent_key, [])
        if not new_pages:
            new_pages = []
            for i in range(len(page_numbers)):
                new_pages.append(i + 1)

        original_pages = await bind_new_pages_to_original(original_pages, new_pages)

        list_result = await extract_policy_list_fields(
            pdf_path=new_pdf_path,
            parent_key=parent_key,
            extraction_schema=extraction_schema,
            page_numbers=new_pages,
            tmp_dir=tmp_dir,
            extraction_instruction=extraction_instruction,
            schema_context=schema_context,
            original_pages=original_pages,
        )

        all_token_consumptions.extend(list_result.get("token_consumptions", []))
        result["token_consumptions"] = all_token_consumptions

        result[parent_key] = list_result.get(parent_key, [])
        return result
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(
            f"Error in list mapper for {parent_key} at (Line {lineno}) | new pages ({new_pages}) and original pages ({original_pages}): {str(e)}",
            exc_info=True,
        )
        return {
            parent_key: [],
            "token_consumptions": [],
            "object": False,
        }
