from enum import Enum


class GeminiModels(Enum):
    FLASH = "models/gemini-2.5-flash"
    FLASH_LITE = "models/gemini-2.5-flash-lite"


MANDATORY_FIELDS = ["general", "insured_profile"]

commercial_first_level_instruction = """

## INFORMATION ABOUT THE DOCUMENT:

The provided document is a "Commercial line Package Policy" that may contain multiple types of business insurance policies (LOB- Line of Business) consolidated into a single policy document. The commercial policies typically contain **5-20 different policy types** such as:
(Commercial General Liability, Commercial Property, Commercial Auto, Cyber Liability, Workers' Compensation, Professional Liability, Crime and Fidelity, Umbrella/Excess Liability, Business Interruption, etc.)

**LOB-Line of Business**: This means different types of polices in this policy document. Eg. General Liability, Commercial Property, etc.

### Critical Document Pages

**Essential Pages (High Priority):**
- **Common Declaration Page**: This page is usually present in start of the PDF-Document. This contains the (LOB/Type of policy) names with the coverage amounts. The (LOB) names may be present with Aliases. This is strong indicator which shows which (LOB) are present in this policy doc. If any (LOB) have ("nil", "0", "not-included") as coverage values means you should for sure discard that (LOB) at time of mapping. 
- **Declaration Pages**: These pages are present for each (LOB) separately having core policy details including policyholder info, policy numbers, effective dates, coverage limits, and premiums. Usually all declarations pages are titled "Declarations" or "Policy Declarations", "Supplementary Declarations" or either they have table with "Declarations" as title.
- **Summary of Coverages/Coverage Summary/Premium Summary**: These pages contain complete overview of all included coverages and their limits and sub-limits.
- **Forms and Endorsements Directory**: This type pages usually contain table of "Forms and Endorsements" having "form-number" against their "form-names". May span in multiple pages but mostly contains table or key-value like structure.
"""


commercial_second_level_instruction = """

## INFORMATION ABOUT DOCUMENT:
- The provided document was created from an large "Commercial line Package Policy Document". Usually a package policy may contain information about 15-20 different policy types. For example: (Commercial General Liability, Commercial Property, Commercial Auto, Cyber Liability, Workers' Compensation, Professional Liability, Crime and Fidelity, Umbrella/Excess Liability, Business Interruption, ... etc.)
- Each policy type may have its own set of coverages, limits, deductibles, exclusions, forms, and endorsements.
- So, the provided document was created by filtering out the pages from the original (500+ pages) package policy document that are relevant to only this policy type (schema that you will receive).
- The provided document may still contain some pages that are not relevant to this specific policy type. Your task is to identify and select only those pages that contain information pertinent to the fields in the schema and this policy type.
---

## SPECIFIC INSTRUCTIONS BASED ON DOCUMENT AND SCHEMA

### 1. High-Priority Pages (Always Classify as Important)
- **Summary Pages**: Any page titled "Summary of Coverages", "Premium Summary", or lists containing coverage amounts.
- **Declaration Pages**: All pages titled "Declarations", "Common Declarations", "Supplementary Declarations", or "Broad Form Declarations".
- **Limits Pages**: Any page with tables for "Limit of Insurance" or values for "limits".

### 2. Field-Specific Page Selection Rules

#### For 'common_exclusions':
- **Goal**: Find one page for each unique exclusion. The next agent will extract only the exclusion names.
- **Priority 1**: Find pages that contain table having list the exclusions names for this policy type.
- **Priority 2**: If first not found, then find pages which contains exclusion names as page headings or section headings.
- **Output**: Return only one representative page per unique exclusion.

#### For 'forms_and_endorsements':
- **Goal**: Find all pages that contains table like structure having list all forms and endorsements. Eg: Pages with a directory/index of form names and numbers, Pages having table for form-numbers and names.
- **Output**: Return all pages where this table spans.
- **CRITICAL**: DO NOT return pages containing the full text or detailed content of the forms. Only "forms-endorsement" table is required.

#### For all other fields:
('policy_identification', 'coverage_information', 'limits_and_sublimits', 'deductibles', 'additional_coverages_extensions')
- **Rule**: These fields contain direct values. Prioritize pages where the field name (or a synonym) appears, especially in tables, lists, or bullet points.
- **Avoid**: Pages with only general descriptions, terms, conditions, or examples.

"""


commercial_extraction_instructions = """

## SPECIFIC VALUE TYPE RULES:
- **Date-Values**: When extracting date values, always strictly return them in ISO 8601 format: YYYY-MM-DD (for example, 2025-09-13). Return only the date string without any extra text, explanation, or punctuation.
- **Amount-Values**: When extracting monetary amounts from text:
    - **Return only** the amount in the format **`$<number>`** (example: `$5000`, `$1000`).
    - **No spaces** after the `$` symbol.
    - **No commas** in the number.
    - **correct format**: `$5000`, `$1000`
    - **incorrect format**: `$ 5,000`
- **Percentage-Values**: When extracting percentage values, strictly return only the numerical value followed by the "%" symbol (e.g., "5%", "12.5%"). Do not include any additional text or explanations.
- **Location-Addresses**: When extracting location or address values, return the **full** address as a single string without any extra text or explanation (only the address is needed).
- For all other value types, always provide the exact value as it appears in the document without any modifications or additional text.

## RULES FOR SPECIFIC FIELDS:
- **'common_exclusions'**: 
    - You only have to find the exclusion-names and return the value, no need to provide any other explainations around it. Exclusion names are always short phrases of (max 3-5 words).
    - Exclusion names are either found in an "exclusions-table" or in headings of the page/section.
    - Example (correct format): ["War", "Nuclear Hazard", "Pollution"]
    - Example (incorrect format): ["This policy is excluded under War .....", etc.]
- **'forms_and_endorsements'**: 
    - You only have to find and return the form/endorsement name and its number against it.
    - Do not provide any other explainations, information or details around it. Just return the name and number.
    - Both "forms_or_endorsements_number" and "forms_or_endorsements_name" are direct values, so return them exactly as they appear in the document.
---

"""

# us_personal_first_level_instruction = """
# ## INFORMATION ABOUT THE DOCUMENT/SCHEMA:

# - The provided document is a "US Personal line Policy" that will only contain a single lob (line of business) either Personal auto or personal home.
# - The schema will contain -
#     - lead_repo: The personal detail of the insured person (like name, address, phone number, etc.). 
#     - questionnaire_repo: This will contain the policy details of the insured person (like policy details,mortgage details,claim details,etc.).
# """

us_personal_second_level_instruction = """
## Field-Specific Rules
- When the document lists multiple person names under the `insured` section:
  - Treat the first name as the primary `insured`.
  - Treat all subsequent names other than primary insured as `co_insured`, regardless of how they are labeled in the document.
- In case of Mailing Address, if the document has a mailing address section, use that address as the mailing address. If not, use the insured's address as the mailing address.
"""

us_personal_extraction_instructions = """
## SPECIFIC VALUE TYPE RULES:
- **Date-Values**: When extracting date values, always strictly return them in ISO 8601 format: YYYY-MM-DD (for example, 2025-09-13). Return only the date string without any extra text, explanation, or punctuation.
- **Amount-Values**: When extracting monetary amounts from text:
    - **Return only** the amount in the format **`$<number>`** (example: `$5000`, `$1000`).
    - **No spaces** after the `$` symbol.
    - **No commas** in the number.
    - **correct format**: `$5000`, `$1000`
    - **incorrect format**: `$ 5,000`
- **Percentage-Values**: When extracting percentage values, strictly return only the numerical value followed by the "%" symbol (e.g., "5%", "12.5%"). Do not include any additional text or explanations.
- **Location-Addresses**: When extracting location or address values, return the **full** address as a single string without any extra text or explanation (only the address is needed).
- For all other value types, always provide the exact value as it appears in the document without any modifications or additional text.
- When the document lists multiple person names under the `insured` section:
  - Treat the first name as the primary `insured`.
  - Treat all subsequent names other than primary insured as `co_insured`, regardless of how they are labeled in the document.
- In case of Mailing Address, if the document has a mailing address section, use that address as the mailing address. If not, use the insured's address as the mailing address.
- when any field description contain type boolean,then return only true or false.
---

"""
