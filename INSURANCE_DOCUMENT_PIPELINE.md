# Insurance Document Processing Pipeline (End-to-End)

This document explains the complete insurance document processing pipeline starting from `process_insurance_document`, detailing each stage, key functions, important rules/cases handled, and the shape of the data produced/consumed along the way.

## High-Level Flow

- **Entry point**: `process_insurance_document`
- **Orchestrator**: LangGraph workflow created by `create_insurance_workflow` and routed by `insurance_document_router`
- **Major stages (nodes)**:
  - `coverage_mapper`
  - `coverage_detail_mapping`
  - `policy_data_extractor`
  - `processing_cost_calculator`
- **Output**: Final state with extracted policy data and token consumption, persisted and published via Redis

---

## Entry and Workflow Construction

### process_insurance_document

- **File**: `app/insurance_policy_processor/policy_orchestrator/agents/supervisor.py`
- **Purpose**: Initializes and runs the LangGraph workflow, manages temp directory and Redis writes, and performs cleanup
- **Inputs**:
  - `pdf_path`: Path to the input insurance document PDF
  - `insurance_extraction_schema`: Full JSON schema that defines expected structure
  - `tmp_dir`: Base directory where the per-run temp folder is created
  - `policy_document_id`: Identifier used for Redis persistence and Pub/Sub
  - `list_lobs`: Optional list of LOB filters to scope extraction
- **Flow**:
  1. Calls `create_insurance_workflow()` to construct and compile the graph
  2. Creates a unique temporary directory for the run
  3. Builds `InsuranceDocumentState` with all inputs
  4. Invokes the workflow asynchronously
  5. On success:
     - Writes `{extracted_policy_data, token_usage}` to Redis key `policy_document_id`
     - Publishes a completion message to channel `insurance_document_processing_complete:{policy_document_id}`
  6. On failure:
     - Logs error, writes `{}` to Redis
     - Publishes failure event with error details
  7. Finally:
     - Removes the temporary directory
     - Attempts to delete any uploaded PDFs from Gemini

### create_insurance_workflow and insurance_document_router

- Builds nodes and edges for the pipeline using `StateGraph(InsuranceDocumentState)`
- **Entry node**: `coverage_mapper`
- **Routing** based on `pdf_length` (currently both branches lead to `coverage_mapper`, but the hook allows future branching)
- **Edges**: `coverage_mapper` → `coverage_detail_mapping` → `policy_data_extractor` → `processing_cost_calculator` → END

---

## Stage 1: process_insurance_document

- **File**: `app/insurance_policy_processor/policy_document_handler/policy_metadata_extractor.py`
- Opens the PDF with PyMuPDF and sets `state.pdf_length` (used for routing)
- Returns the updated state

---

## Stage 2A: Coverage Mapping (small PDFs)

### coverage_mapping_wrapper → map_coverage_sections → get_coverage_mapping

- **Files**:
  - Wrapper: `app/insurance_policy_processor/policy_orchestrator/agents/processor_wrappers.py`
  - Core: `app/insurance_policy_processor/coverage_mapper/mapper.py`
- **What it does**:
  - Uploads the full PDF via `upload_policy_file`
  - Builds a response schema that lists only first-level keys of `insurance_extraction_schema`
  - Prompts Gemini to map each coverage section to a list of relevant page numbers
- **Critical rules enforced**:
  - Use absolute page order (first PDF page is 1). Ignore internal page numbers printed in the document
  - Treat multi-page tables/sections as a single logical unit
  - Avoid forced matches—return `[]` if a coverage field has no relevant pages
- **Output shape**:
  - `state.coverage_mapping_result = { "<coverage_section>": [page_numbers...] }`
- Token usage is recorded into `state.coverage_mapping_cost`

---

## Stage 2B: Structural Analysis + Combined Coverage Mapper (large PDFs)

### structural_analysis → structural_analysis_agent

- **File**: `app/insurance_policy_processor/coverage_mapper/new_mapper.py`
- Uploads PDF and instructs Gemini to produce a machine-parsable "Document Structure Summary" with grouped `page_numbers` and a short `summary` per block
- This is context for the next step, not the final mapping
- Stores text summary in `state.coverage_document_structure` and records token usage

### combine_coverage_mapper → combine_coverage_mapper_agent

- Uses the structural summary + full schema to classify pages against coverage sections
- Enforces the same rules (absolute page order, no forced matches, multi-page handling)
- **Output shape**:
  - `state.coverage_mapping_result = { "<coverage_section>": [page_numbers...] }`
- Token usage recorded in `state.coverage_mapping_cost`

---

## Stage 3: Coverage Detail Mapping

### coverage_detail_mapping_wrapper → map_coverage_details

- **File**: `app/insurance_policy_processor/coverage_detail_mapper/mapper.py`
- **Inputs**:
  - `state.insurance_extraction_schema` (full)
  - `state.coverage_mapping_result`
- For each coverage section (parent key):
  - Calls `nested_level_mapping(insurance_extraction_schema[parent_key], parent_key, mapped_pages, ...)`
  - Aggregates per-field results into `state.coverage_detail_mapping_result`

### nested_level_mapping

- Creates a filtered PDF for only the relevant `page_numbers`
- Determines schema type for the parent key:
  - **List** → `is_list = True`; returns pages (rebased if necessary) and schema
  - **Scalar** (non-dict) → `is_field = True`; returns early with pages
  - **Dict** (nested) → `is_nested = True`; creates `schema_context`, asks Gemini to classify child fields to pages within the filtered PDF
- For nested dicts, transforms `nested_fields_mapping` into:
  - `{ child_key: { "relative_pages": [...], "original_pages": [...] } }` (binds sub-pdf pages back to original)
- Returns per-parent-key bundles containing: `pages/pdf_path/tmp_dir/schema/is_list/is_field/is_nested/nested_fields_mapping/token_consumptions`
- Token usage is tracked and bubbled up

**Edge cases and rules**:
- Absolute page numbering; ignore page numbers printed in content
- Multi-page table/section classification
- Optional post-validation hooks exist (commented in code) for large sets

---

## Stage 4: Policy Data Extraction Agent

### policy_data_extraction_wrapper → extract_policy_data

- **File**: `app/insurance_policy_processor/policy_data_extractor/policy_extractor.py`
- **Inputs**: `state.coverage_detail_mapping_result`
- For each coverage section bundle, calls `run_extraction_for_individual_coverage_section(...)` in parallel
- Aggregates results into `state.extracted_policy_data` and collects token usage into `state.extraction_cost`

### run_extraction_for_individual_coverage_section

- Decides extraction path by bundle flags:
  - `is_list` → `extract_policy_list_fields` (with `is_first_level=True` when top-level)
  - `is_field` (scalar) → `extract_policy_direct_fields`
  - Else (nested object) → `extract_nested_coverage_object`

### extract_nested_coverage_object

- Iterates over `nested_fields_mapping`:
  - If child schema is a list → `extract_policy_list_fields`
  - If child is a scalar → `extract_policy_direct_fields`
  - If child is a dict → `bifurcate_object` to separate list-heavy objects and extract both object and list parts appropriately
- Merges sub-results and token usage

### bifurcate_object

- Heuristics to decide if object splitting is needed (counts of lists/objects/total keys)
- If necessary, separates list sub-schemas and runs:
  - `extract_list_fields_with_new_mapper` for lists (per-item targeted extraction)
  - `extract_policy_object_fields` for remaining object part
- Merges final results and token usage

**Extraction helpers used across flows**:
- `extract_policy_direct_fields`: Extracts scalars/leaves from mapped pages
- `extract_policy_object_fields`: Extracts objects (dict-like data)
- `extract_policy_list_fields` / `extract_list_fields_with_new_mapper`: Extracts lists; the latter leverages a mapper to localize per-item segments when helpful
- **Common utilities**:
  - PDF page filtering via `filter_policy_pdf_pages`
  - Schema context generation for concise prompts
  - Mapping relative pages back to original pages
  - Cleaning/normalizing responses, structured output enforcement

---

## Stage 5: Processing Cost Calculator

### processing_cost_calculator

- **File**: `app/insurance_policy_processor/policy_orchestrator/agents/processor_wrappers.py`
- Invokes `calculate_insurance_cost` (see `app/insurance_policy_processor/processing_cost_calculator/metrics_calculator.py`)
- Aggregates token consumption across agents into `state.token_consumption`

---

## Redis Persistence and Events

- **On success** in `process_insurance_document`:
  - Saves to Redis key `policy_document_id`:
    - `extracted_policy_data = state.extracted_policy_data`
    - `token_usage = state.token_consumption`
  - Publishes to `insurance_document_processing_complete:{policy_document_id}` with:
    - `status`, `policy_document_id`, `data`, `token_usage`, and a JSON-serializable dump of the entire state (dates converted to ISO strings)

- **On failure**:
  - Stores empty `{}` value at key
  - Publishes a task failure message with error details

---

## Data Shapes (Key Artifacts)

### Coverage Mapping Result
```json
{
    "<coverage_section>": [<page_numbers>]
}
```

### Coverage Detail Mapping Result (examples)

**List field root:**
```json
{
    "<coverage_list_section>": {
        "is_list": true,
        "pages":,​​
        "pdf_path": "...",
        "tmp_dir": "...",
        "schema": {/* list schema */},
        "token_consumptions": [ ... ]
    }
}
```


**Scalar field root:**
```json
{
    "<coverage_scalar_section>": {
        "is_field": true,
        "pages":,​
        ...
    }
}
```

**Nested object root:**
```json
{
    "<coverage_object_section>": {
        "is_nested": true,
        "nested_fields_mapping": {
            "child_key": {
                "relative_pages":,​​
                "original_pages":​
            },
            ...
        },
        "schema_context": { ... },
        ...
    }
}
```

### Final Extraction Result
```json
{
    "<coverage_section>": <extracted_policy_data>,
    ...
}
```

---

## Key Rules and Edge Cases

- **Absolute PDF page order only**; ignore page numbers printed in content
- **Multi-page tables/sections** are treated as one logical unit in page mapping
- **No forced matches**—empty results are allowed when no relevant pages exist
- **Nested schema handling** distinguishes arrays, scalars, and objects, and routes to appropriate extractors
- **Sub-PDF creation** and relative→original page binding are used to focus and track provenance
- **Token usage** is collected per stage and aggregated at the end

---

## Outputs Returned by process_insurance_document

Returns final `InsuranceDocumentState`, notably:
- `state.extracted_policy_data` (final extracted JSON)
- `state.token_consumption` (aggregate tokens)
- `state.coverage_mapping_result` (coverage section page mapping)
- `state.coverage_detail_mapping_result` (nested field page mapping)

Also persists to Redis and publishes completion/failure events as described above.

---
