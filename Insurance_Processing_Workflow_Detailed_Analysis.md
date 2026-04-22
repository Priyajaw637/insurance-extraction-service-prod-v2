# Insurance Document Processing Workflow Analysis

## Overview
The workflow is a Directed Acyclic Graph (DAG) implemented using **LangGraph**. It orchestrates the extraction of structured data from insurance policy PDFs. The process involves identifying Lines of Business (LOBs), mapping relevant pages to them, refining those mappings to specific schema fields, and finally extracting the data using highly focused/filtered PDF contexts to minimize noise and token usage.

**Entry Point:** `process_insurance_document` in `supervisor.py`.
**Workflow Creation:** `create_insurance_workflow` defines the graph nodes and edges.

---

## Workflow Steps (Nodes)

### 1. `lob_identifier` (Line of Business Identification)
**Goal:** Determine which Lines of Business (e.g., Commercial General Liability, Property) are present in the uploaded PDF and identifying the form type (Index/Schedule vs Individual Pages).
**Handler:** `get_lob_name_and_form_type` (`lob_indentifier.py`)

**Logic & Conditions:**
*   **Input:** `State` containing entire PDF and full extraction schema.
*   **Condition:** If `state.list_lobs` is provided (user pre-specified LOBs):
    *   If `tool_name` is `"proposal_generation"`: **Skip** identification completely. Return state as-is.
    *   If `tool_name` is `"policy_checking"`: Run **Gemini Flash Lite** to *only* identify the form type (`Index_Schedule_Type_Forms` vs `Individual_Pages_Type_Forms`). Skipping full LOB check.
*   **Default Path:** Run **Gemini Flash** with the full PDF to identify:
    1.  Active LOBs (checking Declarations/Premium Summary).
    2.  Form types.
*   **Schema Filtering:** The global `insurance_extraction_schema` is filtered to retain *only* the keys corresponding to identified LOBs + `MANDATORY_FIELDS`. Unused LOB schemas are dropped to save processing time later.

### 2. `coverage_mapper` (Page-to-LOB Mapping)
**Goal:** Identifying which pages belong to which identified Line of Business.
**Handler:** `coverage_mapping_wrapper` -> `map_coverage_sections` (`coverage_mapper/mapper.py`)

**Logic & Conditions:**
*   **Input:** `State` with filtered schema and identified LOBs.
*   **Prompt Selection:** Selects custom extraction instructions (`CUSTOM_PROMPT_FIRST`, etc.) based on `country`, `line_of_business`, and `identified_forms_type`.
*   **Execution:** Calls **Gemini Flash** with the full PDF.
*   **Task:** Returns a map of `LOB_Key -> [List of Page Numbers]`.
    *   *Optimization:* Uses "semantic probability" to map pages, not just keyword matching.
    *   *Rule:* Ignores internal page numbers; uses PDF absolute index.

### 3. `coverage_detail_mapping` (Nested Field Mapping)
**Goal:** For each LOB, drill down to identify which pages cover specific *groups* of fields (first-level keys of that LOB's schema).
**Handler:** `coverage_detail_mapping_wrapper` -> `map_coverage_details` (`coverage_detail_mapper/mapper.py`)

**Logic & Conditions:**
*   **Input:** Result from `coverage_mapper`.
*   **Parallelization:** Iterates over each LOB found in the previous step. Validates if the LOB has a nested schema.
*   **Execution (Per LOB):**
    *   **PDF Filtering:** Creates a *temporary, smaller PDF* containing only the pages mapped to this LOB in Step 2.
    *   **Mapping:** Calls **Gemini Flash** on this `filtered_pdf` to map pages to the *first-level keys* of that LOB's schema (e.g., "Coverages", "Exclusions", "Locations").
    *   **Page Binding:** The model returns "relative" page numbers (1..N of the filtered PDF). The code converts these back to "absolute" page numbers (Original PDF index).
*   **Output:** A granular map: `LOB -> Nested_Key -> { relative_pages, original_pages }`.

### 4. `policy_data_extractor` (Data Extraction)
**Goal:** Extract the actual values (Leaf nodes) or Lists/Objects for the fields.
**Handler:** `policy_data_extraction_wrapper` -> `extract_policy_data` (`policy_extractor.py`)

**Logic & Conditions:**
*   **Parallelization:** Iterates over the `coverage_detail_mapping` results (Per LOB).
*   **Routing (Per Field Group):**
    *   **List Fields:** Calls `extract_policy_list_fields`.
    *   **Simple Fields:** Calls `extract_policy_direct_fields`.
    *   **Complex Objects:** Calls `extract_nested_coverage_object`.
*   **Bifurcation Strategy (`bifurcate_object`):**
    *   Inside a complex object, it checks complexity:
        *   If `list_counts > 1` OR (`list_counts == 1` AND `object_counts >= 1`) OR (`total_keys > 5` with a list):
            *   It splits the extraction. It extracts lists separately using `extract_list_fields_with_new_mapper` and the remaining object fields using `extract_policy_object_fields`.
*   **Execution:**
    *   **PDF Filtering:** For each extraction task, it creates *another* temporary PDF containing only the pages relevant to *that specific nested key*.
    *   **Model:** Uses **Gemini Flash Lite** (usually) for the final extraction of values to ensure speed and low cost, given the context is now highly specific.
    *   **Page Citation:** The model is required to return the page number where the value was found.
*   **Aggregation:** All parallel extraction results are merged into the final `extracted_policy_data` JSON structure.

### 5. `processing_cost_calculator` (Cost Analysis)
**Goal:** Aggregate token usage and calculate costs.
**Handler:** `processing_cost_calculator` (`metrics_calculator.py`)

**Logic:**
*   Aggregates input, output, cached, and thinking tokens from all previous steps (`coverage_mapping_cost`, `coverage_detail_mapping_cost`, `extraction_cost`).
*   Updates `state.token_consumption` with the total usage per model.
*   *(Note: The actual cost logging/printing logic appears to be unreachable in the current code due to an early return statement).*

---

## State Management (`InsuranceDocumentState`)

The state object flows through the graph, accumulating data:
*   `pdf_path`, `uploaded_pdf`: The raw document.
*   `insurance_extraction_schema`: The target JSON structure (filtered dynamically).
*   `identified_lobs`: Result of Step 1.
*   `coverage_mapping_result`: Page maps from Step 2.
*   `coverage_detail_mapping_result`: Granular page maps from Step 3.
*   `extracted_policy_data`: Final JSON output from Step 4.
*   `token_consumption`: Final usage stats.

## Final Output
After the graph finishes:
1.  **Redis Storage:** The results (`extracted_policy_data`) and token usage are saved to Redis under `policy_document_id`.
2.  **Notification:** A completion message is published to a Redis channel (`insurance_document_processing_complete:...`).
3.  **Cleanup:** Temporary PDF slices and the uploaded file are deleted.
