# Commit Changes Summary (Dec 20, 2025 - Jan 12, 2026)

This document summarizes the changes made to the codebase between **December 20, 2025** and **January 12, 2026**.

## Overview
During this period, the focus was on implementing and refining the **US Personal Insurance** document extraction and comparison logic, transitioning to a new architecture (DI Arch), and enhancing the commercial comparison functionality.

---

## Key Changes & New Features

### 1. US Personal Insurance Implementation
- **New Celery Task**: `process_proposal_documents_us_personal` was implemented to handle processing for US personal insurance proposals.
- **Dedicated Helper Utilities**: Created `app/utils/personal_helper.py` to manage styling and PDF generation specific to US personal lines (Auto and Home).
- **PDF Generation**:
    - Implemented `pdf_generation_comparision_table_home` for home insurance comparisons.
    - Updated `pdf_generation_comparision_table` for auto insurance comparisons.

### 2. Enhanced Extraction Logic (DI Arch)
- **New Extraction Architecture**: Shifted the US personal line to a more modular architecture.
- **Externalized Instructions**: Extraction instructions for policy data were externalized to improve maintainability.
- **New Utilities**: Added robust safe type conversion functions:
    - `safe_float`
    - `safe_int`
    - `safe_bool`
    - `safe_string`

### 3. Policy Comparison Enhancements
- **Function Update**: `commercial_compare_jsons` underwent significant updates to support US personal policy comparisons and improve normalization/formatting.
- **Schema Improvements**: 
    - Major updates to the policy schema for US personal lines.
    - Clarified field descriptions and types (e.g., ensuring `is_applied` and `is_included` are treated as booleans).

### 4. Route & Endpoint Updates
- **`app/routes/insurance_commercial.py`**:
    - Updated `process_single_document` to integrate with the new extraction and comparison workflows.
    - Enhanced logic for handling different document types (Commercial vs. Personal).

---

## Commit History Detail

| Date | Commit Hash | Description |
| :--- | :--- | :--- |
| 2026-01-12 | `dfca939` | Added policy checking US comparison function updates in `commercial_comparison.py`. |
| 2026-01-03 | `56a1d8b` | Clarified boolean fields in schema descriptions. |
| 2025-12-30 | `c44f0b1` | Documentation improvement for US personal policy schema. |
| 2025-12-30 | `1d3d5e3` | Updated policy schema for US lines. |
| 2025-12-29 | `40adb95` | Fixed issues in US personal line for auto and home. |
| 2025-12-27 | `0926594` | Externalized extraction instructions and enhanced commercial comparison normalization. |
| 2025-12-23 | `732c091` | Implemented insurance document extraction and shifted US personal line to DI Arch. |
| 2025-12-22 | `605f49b` | Initial implementation of US personal insurance proposal processing and helper utilities. |

---

## Files Modified
- `app/modules/Celery/tasks.py`
- `app/routes/insurance_commercial.py`
- `app/utils/helper.py`
- `app/utils/personal_helper.py`
- `app/modules/InsuranceDocument/commercial_comparison.py`
- `app/modules/InsuranceDocument/schema/policy_schema.json`
- `testing.py`
- `app/core/enums.py`
- `app/core/models.py`
