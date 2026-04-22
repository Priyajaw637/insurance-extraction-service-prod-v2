import asyncio
import json
import os
import sys
import time
from uuid import uuid4

from google.genai import types

from app.insurance_policy_processor.policy_document_handler.policy_page_extractor import (
    filter_policy_pdf_pages,
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
from app.utils.core import (
    bind_new_pages_to_original,
    clean_response,
    create_output_schema,
    generate_short_schema_for_context,
    pdf_to_bytes,
)
from app.logging_config import get_logger
from app.utils.helpers import save_json, load_json

logger = get_logger("coverage_detail_mapper")


async def nested_level_mapping(
    extraction_schema,
    parent_key,
    page_numbers,
    tmp_dir,
    second_level_instruction,
    main_pdf_path,
    idx,
    original_pdf_length=None,
):
    result = {
        "pages": page_numbers,
        "pdf_path": main_pdf_path,
        "tmp_dir": tmp_dir,
        "nested_fields_mapping": {},
        "schema": extraction_schema,
        "token_consumptions": None,
        "is_list": False,
        "is_field": False,
        "is_nested": False,
        "schema_context": None,
    }
    if not page_numbers:
        logger.info(f"No pages provided for nested level mapping of {parent_key}")
        return {parent_key: result}
    try:
        await asyncio.sleep(delay=idx * 0.6)

        # Upload Pdf File
        new_pdf_path = os.path.join(tmp_dir, f"{uuid4()}.pdf")
        new_pdf_path = await filter_policy_pdf_pages(
            pdf_path=main_pdf_path, page_numbers=page_numbers, output_path=new_pdf_path
        )
        new_pdf_file_bytes = await pdf_to_bytes(pdf_path=new_pdf_path)
        result["pdf_path"] = new_pdf_path
        common_exclusions = None

        # If the schema itself is list
        if isinstance(extraction_schema, list):
            extraction_schema = {parent_key: extraction_schema}
            result["schema"] = extraction_schema
            result["is_list"] = True
        elif not isinstance(extraction_schema, dict):
            result["is_field"] = True
            extraction_schema = {parent_key: extraction_schema}
            result["schema"] = extraction_schema
            return {parent_key: result}
        else:
            result["is_nested"] = True
            schema_context = await generate_short_schema_for_context(
                extraction_schema=extraction_schema
            )
            result["schema_context"] = {parent_key: schema_context}

            if "forms_and_endorsements" in extraction_schema:
                common_exclusions = extraction_schema.pop("common_exclusions", None)
            elif "common_exclusions" in extraction_schema:
                second_level_instruction += """
- for `common_exclusions` map the pages that have Header/Title having exclusions for this given LOB.
"""

        # Create Output Schema for prompt
        output_schema = await create_output_schema(full_schema=extraction_schema)

        if common_exclusions:
            extraction_schema["common_exclusions"] = common_exclusions

        system_prompt = f"""
# ROLE:
You are a specialized Insurance Document Analyst with expertise in understanding policy document and map relevant pages to schema fields for the provided Lines of Business.

---

# INPUTS:
1. PDF Document: A filtered PDF generated from the package policy document. The document is filtered to include only the pages that are relevant to the Lines of Business (LOB) provided in the `name_of_lob` field.
2. name_of_lob: The name of the Lines of Business (type of policy) for which the mapping is to be done.
3. lob_schema: The JSON schema to be used for classification. You only have to map pages to first-level keys from this schema.

---

# TASK:

### Primary Objective: 
- STEP 1: First Analyze the provided document content.
- STEP 2: Go through the first-level keys of `lob_schema` to see what overall structure the schema has.
- STEP 3: Then go through the sub-fields of each first-level key from the provided schema to understand what data they need and understand what to map.
- STEP 4: Then classify pages from this document against each **First-Level Key** from the provided schema.
- NOTE: For each first-level key, provide a list of page numbers that contains values for all fields for that first-level key.

### Page Classification Criteria:
- Base the classification on the semantic probability of finding the *values* for fields from `insurance_schema` for this LOB, not merely on a string match of the field's name.
- Do not expect document to have exact matches for fields. Document may have different wordings, synonyms, aliases, or variations of the same field. Use your insurance-industry knowledge to make best effort to find matches.
- Identify Tabular Data: When content is in a table, treat the entire table as a single logical unit for mapping, even if it extends across multiple pages.
- Detect Page Spans: When mapping across consecutive pages (table rows, etc), Actively look for indicators that a table or section continues onto the next page, such as repeated headers, "(Continued)" notices, or sentences and rows split between pages.
- Avoid Forced Matches: If there are no relevant pages for any first-level key, return an empty list `[]` for that field. Do not force a classification.
- **Never** discard any page which have a amount or premium value.

### Avoid Duplicate Pages:
- When for a field, its value appears on multiple different pages, Then return only one page for that field.
- Example: lets say value of field `effective_date` appears on page [1, 2, 5, 10...]. and all these pages have same value. Then, return only page 1

---

{second_level_instruction}

---

# PAGE MAPPING RULES:

- **Use Absolute Page Order**: You are required to use the policy document's **actual, sequential page number** for mapping. The first page of the PDF is page 1, the second is page 2, and so on.
- **Strictly Ignore All Internal Page Numbers**: Completely discard any page numbers written within the policy document's content. This includes:
  - Page numbers in headers or footers.
  - Page numbers found within a Table of Contents.
- **Rationale**: The provided PDF document was generated from filtering pages from a large policy document. So, there is very **high** chance that the PDF will be non-contiguous and may have **gaps** in the page numbers.
- **Example**: If the document's 25th physical page has "Page 63" printed in its footer, then the correct page number to use in output will be **25** and not **63**.
- **Final Output Rule**: All page numbers in your final mapping must correspond exclusively to the sequential order of pages from the beginning to the end of the policy document.

---

# OUTPUT FORMAT:
- Return your response as a JSON object, with mapping of page numbers for each **First-Level Key**.
- Mapping should be a list of page numbers that cover all fields for that first-level key.

---
"""

        user_prompt = f"""
## Schema Information

- All **leaf nodes (end nodes)** represent **direct value fields**.
- Direct value fields contain short, factual values such as:
- numbers
  - amounts
  - percentages
  - dates
  - booleans
  - short strings

## Page Mapping Rules

- Your task is **only to identify the page number(s)** where the field’s value appears.
- Locate the page where the **explicit value is present**, not where it is discussed generally.
- Ignore descriptive paragraphs unless they contain the actual value.


## Schema Field Types

### Object / Dictionary Fields
- The number of mapped pages should generally align with the number of nested keys.
- As a guideline, avoid mapping significantly more pages than the total nested fields.
- Example: If an object contains 5 keys, the mapped pages should typically not exceed ~5.
- Map only the pages where the actual field values appear — avoid broad or unnecessary page ranges.

### List Fields
- List fields may span multiple pages.
- Provide all relevant pages required to cover the items in the list.
- Do not add extra pages once the list content is fully covered.


## LOB Information:
- The provided document is created by filtering pages for "{parent_key}" LOB from a large policy document.
- So, the provided document will have pages highly related to "{parent_key}" LOB.

---

<name_of_lob>
"{parent_key}"
</name_of_lob>
   
<lob_schema>

{json.dumps(extraction_schema, indent=2)}

</lob_schema>

---
    """
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
            "seed": 67,
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
            file_path=new_pdf_path,
            file_processing="bytes",
            mime_type="application/pdf",
            text_contents=[user_prompt],
        )
        response = await get_gemini_result_with_retry(task_id)

        all_token_consumptions = []
        final_result = None

        if response and response.usage_metadata:
            usage_info = response.usage_metadata
            token_consumptions = {
                "model": model_name,
                "input_tokens": usage_info.prompt_token_count or 0,
                "output_tokens": usage_info.candidates_token_count or 0,
                "cached_tokens": usage_info.cached_content_token_count or 0,
                "thinking_tokens": usage_info.thoughts_token_count or 0,
            }
            all_token_consumptions.append(token_consumptions)

        if response and (response.parsed is not None or response.text is not None):
            if response.parsed:
                final_result = response.parsed
            else:
                print(f"final_result for '{parent_key}':", response.text)
                final_result = await clean_response(response.text)

        result["token_consumptions"] = all_token_consumptions

        if result["is_list"]:
            result["pages"] = final_result.get(parent_key)
            if result["pages"]:
                result["pages"] = await bind_new_pages_to_original(
                    orginal_pages=page_numbers, new_pages=result["pages"]
                )
            else:
                result["pages"] = page_numbers

            return {parent_key: result}
        else:
            result["nested_fields_mapping"] = final_result

        nested_mapping = {}
        for key, val in result["nested_fields_mapping"].items():
            orginal_pages = await bind_new_pages_to_original(
                orginal_pages=page_numbers, new_pages=val
            )
            nested_mapping[key] = {
                "relative_pages": val,
                "original_pages": orginal_pages,
            }

            # Save Forms pages in common exclusion
            if common_exclusions and key == "forms_and_endorsements":
                nested_mapping["common_exclusions"] = {
                    "relative_pages": val,
                    "original_pages": orginal_pages,
                }

        result["nested_fields_mapping"] = nested_mapping

        return {parent_key: result}
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(
            f"Error in nested mapping for '{parent_key}' at line {lineno}: {str(e)}",
            exc_info=True,
        )
        return {parent_key: result}


async def map_coverage_details(state: InsuranceDocumentState):
    try:
        # Get Stored Response
        # final_result = await load_json(file_path="final_result.json")
        # state.coverage_detail_mapping_result = final_result.get("coverage_detail_mapping_result")
        # logger.info("Coverage detail mapping result loaded from file")

        # return state

        logger.info("Starting coverage detail mapping")
        t1 = time.perf_counter()

        full_schema = state.insurance_extraction_schema
        coverage_mapping = state.coverage_mapping_result

        # Handle case where coverage_mapping_result is None or empty
        if not coverage_mapping:
            logger.warning(
                "Coverage mapping result is empty or None, skipping coverage detail mapping"
            )
            state.coverage_detail_mapping_result = {}
            return state

        for key, val in coverage_mapping.items():
            val = sorted(set(val))

        state.coverage_mapping_result = coverage_mapping

        nested_calls = []
        idx = 0
        for key, val in coverage_mapping.items():
            nested_schema = full_schema.get(key)
            if not nested_schema:
                continue

            nested_calls.append(
                nested_level_mapping(
                    extraction_schema=nested_schema,
                    parent_key=key,
                    page_numbers=val,
                    second_level_instruction=state.second_level_instruction,
                    tmp_dir=state.tmp_dir,
                    main_pdf_path=state.pdf_path,
                    idx=idx,
                    original_pdf_length=state.pdf_length,
                )
            )
            idx += 1

        # logger.info(f"Running {len(nested_calls)} nested mapping tasks in parallel")
        nested_results = await asyncio.gather(*nested_calls, return_exceptions=True)

        final_result = {}
        for result in nested_results:
            if isinstance(result, Exception):
                continue
            for key, val in result.items():
                tokens = val.pop("token_consumptions")
                if tokens:
                    state.coverage_detail_mapping_cost.extend(tokens)

                final_result[key] = val

        state.coverage_detail_mapping_result = final_result
        logger.info("Coverage detail mapping completed")
        logger.info(
            f"Coverage detail mapping finished in {time.perf_counter() - t1:.2f}s"
        )

        # Save Response
        # await save_json(file_path="second_mapping.json", data=final_result)
        # logger.info("Coverage detail mapping result saved to file")

        return state
    except Exception as e:
        tb = sys.exc_info()[2]
        lineno = tb.tb_lineno
        logger.error(
            f"Error in Coverage Detail Mapping at (Line {lineno}): {e}", exc_info=True
        )
        raise e
