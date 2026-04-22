from enum import Enum


class GeminiModels(Enum):
    FLASH = "models/gemini-2.5-flash"
    FLASH_LITE = "models/gemini-2.5-flash-lite"


MANDATORY_FIELDS = ["general", "insured_profile"]

# US, Canada,
# Commercial, Personal
# "Index_Schedule_Type_Forms", "Individual_Pages_Type_Forms", "Unknown"


CUSTOM_PROMPT_FIRST = {
    "US_Commercial_Index_Schedule_Type_Forms": """
# CRITICAL PAGES IN INSURANCE DOCUMENT:

### Common Declaration Page / Premium Summary Page: 
- This page is present at very start of the PDF-Document. 
- It contains the high-level summary of all the coverages and their premiums.
- It may be titled differently for example: "Common Policy Declaration", "Commercial Policy Common Declaration", "Schedule of Coverages", "Coverages Summary", etc.

### Individual Declaration Pages: 
- These are individual declarations pages which are present for each (LOB) separately having core policy details including policyholder info, policy numbers, effective dates, coverage limits, and premiums. 
- Usually these declarations pages are titled "Declarations" or "Policy Declarations", "Supplementary Declarations", "Coverage Declarations", etc.
- They may span in multiple pages.

### Forms and Endorsements:
- These are pages that primarily list all policy forms, endorsements and exclusions.
- Possible page headers: "Forms and Endorsements Schedule", "Forms and Endorsements List", "Forms list", "Endorsements Schedule", "Forms Applicable", etc.
- Usual identifiers for this field are: tabular/listed set of form numbers with form names/description.
- These pages usually span MAX of 3 pages.
- If this is present, then map these to every identified LOB.

### Pages containing values for: 
- Declarations
- Limits
- Sub-limits
- Deductibles.
- Any page that have **amount** mentioned (eg: $1,000) makes it a critical page.

---

# RULE FOR "GENERAL" and "INSURED PROFILE" FIELD:
- The provided schema contains either `general` or `insured_profile` field at first level.
- Both of these are not LOB (Lines of Business) rather they only contain general information about policyholder or insured.
- As they only contain very basic fields, you are only allowed to return a **maximum** of 10 pages and not more.
""",
    "US_Commercial_Individual_Pages_Type_Forms": """
# CRITICAL PAGES IN INSURANCE DOCUMENT:

### Common Declaration Page / Premium Summary Page: 
- This page is present at very start of the PDF-Document. 
- It contains the high-level summary of all the coverages and their premiums.
- It may be titled differently for example: "Common Policy Declaration", "Commercial Policy Common Declaration", "Schedule of Coverages", "Coverages Summary", etc.

### Individual Declaration Pages: 
- These are individual declarations pages which are present for each (LOB) separately having core policy details including policyholder info, policy numbers, effective dates, coverage limits, and premiums. 
- Usually these declarations pages are titled "Declarations" or "Policy Declarations", "Supplementary Declarations", "Coverage Declarations", etc.
- They may span in multiple pages.

### Forms and Endorsements:
- For each LOB there are multiple forms and endorsements and each of forms and endorsements are present with there own set of individual pages.
- Based on **LOB**, Map all the relevant forms and endorsements pages
- These Forms and endorsements pages are always present in adjacent/contiguous pages based on **LOB**.

### Pages containing values for: 
- Declarations
- Limits
- Sub-limits
- Deductibles.
- Any page that have **amount** mentioned (eg: $1,000) makes it a critical page.

---

# RULE FOR "GENERAL" and "INSURED PROFILE" FIELD:
- The provided schema contains either `general` or `insured_profile` field at first level.
- Both of these are not LOB (Lines of Business) rather they only contain general information about policyholder or insured.
- As they only contain very basic fields, you are only allowed to return a **maximum** of 10 pages and not more.
""",
    "US_Personal": """
# RULE FOR "GENERAL" FIELD:
- The provided schema contains `general` field at first level.
- This `general` field is not LOB (Lines of Business) rather it only contain general information about policyholder or insured.
- This field is the mandatory field and is always return its mapped pages.
- As it only contain very basic fields, you are only allowed to return a **maximum** of 10 pages and not more.
""",
    "Canada_Commercial_Index_Schedule_Type_Forms": """
# CRITICAL PAGES IN INSURANCE DOCUMENT:

### Common Declaration Page / Premium Summary Page: 
- This page is present at very start of the PDF-Document. 
- It contains the high-level summary of all the coverages and their premiums.
- It may be titled differently for example: "Common Policy Declaration", "Commercial Policy Common Declaration", "Schedule of Coverages", "Coverages Summary", etc.

### Individual Declaration Pages: 
- These are individual declarations pages which are present for each (LOB) separately having core policy details including policyholder info, policy numbers, effective dates, coverage limits, and premiums. 
- Usually these declarations pages are titled "Declarations" or "Policy Declarations", "Supplementary Declarations", "Coverage Declarations", etc.
- They may span in multiple pages.

### Forms and Endorsements:
- These are pages that primarily list all policy forms, endorsements and exclusions.
- Possible page headers: "Forms and Endorsements Schedule", "Forms and Endorsements List", "Forms list", "Endorsements Schedule", "Forms Applicable", etc.
- Usual identifiers for this field are: tabular/listed set of form numbers with form names/description.
- These pages usually span MAX of 3 pages.
- If this is present, then map these to every identified LOB.
- These do not include Forms, Endorsements, Exclusions that are mapped/present under "Others" Section.

### Pages containing values for: 
- Declarations
- Limits
- Sub-limits
- Deductibles.
- Any page that have **amount** mentioned (eg: $1,000) makes it a critical page.

---

# RULE FOR "GENERAL" and "INSURED PROFILE" FIELD:
- The provided schema contains either `general` or `insured_profile` field at first level.
- Both of these are not LOB (Lines of Business) rather they only contain general information about policyholder or insured.
- As they only contain very basic fields, you are only allowed to return a **maximum** of 10 pages and not more.
""",
    "Canada_Commercial_Individual_Pages_Type_Forms": """
# CRITICAL PAGES IN INSURANCE DOCUMENT:

### Common Declaration Page / Premium Summary Page: 
- This page is present at very start of the PDF-Document. 
- It contains the high-level summary of all the coverages and their premiums.
- It may be titled differently for example: "Common Policy Declaration", "Commercial Policy Common Declaration", "Schedule of Coverages", "Coverages Summary", etc.

### Individual Declaration Pages: 
- These are individual declarations pages which are present for each (LOB) separately having core policy details including policyholder info, policy numbers, effective dates, coverage limits, and premiums. 
- Usually these declarations pages are titled "Declarations" or "Policy Declarations", "Supplementary Declarations", "Coverage Declarations", etc.
- They may span in multiple pages.

### Forms and Endorsements:
- For each LOB there are multiple forms and endorsements and each of forms and endorsements are present with there own set of individual pages.
- Based on **LOB**, Map all the relevant forms and endorsements pages
- These Forms and endorsements pages are always present in adjacent/contiguous pages based on **LOB**.

### Pages containing values for: 
- Declarations
- Limits
- Sub-limits
- Deductibles.
- Any page that have **amount** mentioned (eg: $1,000) makes it a critical page.

---

# RULE FOR "GENERAL" and "INSURED PROFILE" FIELD:
- The provided schema contains either `general` or `insured_profile` field at first level.
- Both of these are not LOB (Lines of Business) rather they only contain general information about policyholder or insured.
- As they only contain very basic fields, you are only allowed to return a **maximum** of 10 pages and not more.
""",
}

CUSTOM_PROMPT_SECOND = {
    "US_Commercial_Index_Schedule_Type_Forms": """
# FIELD SPECIFIC RULES

### If the LOB contains a field 'forms_and_endorsements':
- Strictly map it with Pages that primarily list policy forms, endorsements, exclusions, endorsement schedules, forms list, forms applicable, etc (Always present with form numbers and form titles)
- Usual identifiers for this field are: tabular/listed set of form numbers with form names/description.
- For this field, a maximum of 1-3 page is required.
- For this field, DO NOT map pages containing detailed information, theory, terms and conditions, or examples for each form or endorsement.

### For all other fields in the schema:
- **Rule**: 
  - Focus on pages having the actual value of the field especially in tables, lists, or bullet points.
  - 90% of the fields are direct value fields requiring either amount or percentage or boolean values.
  - No need to map pages with just theory, terms and conditions, or examples without any specific field value.
- **Important Pages**:
  - All Coverage declaration pages
  - All Limits, Sub-limits, Deductibles pages
  - Any page having amount or percentage related fields

### Non-Important Pages which should be skipped/discarded:
- Pages with just aggrements, explanations, theory or examples of different conditions.
- terms and conditions pages that do not have any specific field value.
- In-Short: Any page that do not contain actual value of any field from the schema that can be extracted.
""",
    "US_Commercial_Individual_Pages_Type_Forms": """
# FIELD SPECIFIC RULES

### If the LOB contains a field 'forms_and_endorsements':
- There will individual pages for each Forms and Endorsements present in the policy document.
- For this field, **only** map the first page that contains the form/endorsement name and number as the title. 
- Even if a form spans multiple pages, only map the first page of that form containing the title (FORM NAME AND FORM NUMBER).
- Only map more then one page per form if that page contains any amount or percentage related fields.

### For all other fields in the schema:
- **Rule**: 
  - Focus on pages having the actual value of the field especially in tables, lists, or bullet points.
  - 90% of the fields are direct value fields requiring either amount or percentage or boolean values.
  - No need to map pages with just theory, terms and conditions, or examples without any specific field value.
- **Important Pages**:
  - All Coverage declaration pages
  - All Limits, Sub-limits, Deductibles pages
  - Any page having amount or percentage related fields

### Non-Important Pages which should be skipped/discarded:
- Pages with just aggrements, explanations, theory or examples of different conditions.
- terms and conditions pages that do not have any specific field value.
- In-Short: Any page that do not contain actual value of any field from the schema that can be extracted.
""",
    "US_Personal": """
# ADDITIONAL RULES
- Always map all the pages containing person names under the `insured` section
- In case of Mailing Address is not present, map it to address as the mailing address. If not, use the insured's address as the mailing address.
""",
    "Canada_Commercial_Index_Schedule_Type_Forms": """
# FIELD SPECIFIC RULES

### If the LOB contains a field 'forms_and_endorsements':
- Strictly map it with Pages that primarily list policy forms, endorsements, exclusions, endorsement schedules, forms list, forms applicable, etc (Always present with form numbers and form titles)
- Usual identifiers for this field are: tabular/listed set of form numbers with form names/description.
- For this field, a maximum of 1-3 page is required.
- For this field, DO NOT map pages containing detailed information, theory, terms and conditions, or examples for each form or endorsement.
- These do not include Forms, Endorsements, Exclusions that are mapped/present under "Others" Section.

### For all other fields in the schema:
- **Rule**: 
  - Focus on pages having the actual value of the field especially in tables, lists, or bullet points.
  - 90% of the fields are direct value fields requiring either amount or percentage or boolean values.
  - No need to map pages with just theory, terms and conditions, or examples without any specific field value.
- **Important Pages**:
  - All Coverage declaration pages
  - All Limits, Sub-limits, Deductibles pages
  - Any page having amount or percentage related fields

### Non-Important Pages which should be skipped/discarded:
- Pages with just aggrements, explanations, theory or examples of different conditions.
- terms and conditions pages that do not have any specific field value.
- In-Short: Any page that do not contain actual value of any field from the schema that can be extracted.
""",
    "Canada_Commercial_Individual_Pages_Type_Forms": """
# FIELD SPECIFIC RULES

### If the LOB contains a field 'forms_and_endorsements':
- There will individual pages for each Forms and Endorsements present in the policy document.
- For this field, **only** map the first page that contains the form/endorsement name and number as the title. 
- Even if a form spans multiple pages, only map the first page of that form containing the title (FORM NAME AND FORM NUMBER).
- Only map more then one page per form if that page contains any amount or percentage related fields.

### For all other fields in the schema:
- **Rule**: 
  - Focus on pages having the actual value of the field especially in tables, lists, or bullet points.
  - 90% of the fields are direct value fields requiring either amount or percentage or boolean values.
  - No need to map pages with just theory, terms and conditions, or examples without any specific field value.
- **Important Pages**:
  - All Coverage declaration pages
  - All Limits, Sub-limits, Deductibles pages
  - Any page having amount or percentage related fields

### Non-Important Pages which should be skipped/discarded:
- Pages with just aggrements, explanations, theory or examples of different conditions.
- terms and conditions pages that do not have any specific field value.
- In-Short: Any page that do not contain actual value of any field from the schema that can be extracted.
""",
}

CUSTOM_PROMPT_EXTRACTION = {
    "US_Commercial_Index_Schedule_Type_Forms": """
## FIELD SPECIFIC RULES:
- **Date-Type-Values**: 
  - When extracting date values, always strictly return them in ISO 8601 format: YYYY-MM-DD (for example, 2025-09-13). 
  - Return only the date string without any extra text, explanation, or punctuation.
- **Numeric-Type-Values (e.g., Amount, etc.):**
  - **Return only** the amount in the format **`$<number>`** (example: `$5000`, `$1000`).
  - **No spaces** after the `$` symbol.
  - **No commas** in the number.
  - **correct format**: `$5000`, `$1000`
  - **incorrect format**: `$ 5,000`
- **Percentage-Type-Values**: When extracting percentage values, strictly return only the numerical value followed by the "%" symbol (e.g., "5%", "12.5%"). Do not include any additional text or explanations.
- **Location-Addresses**: When extracting location or address values, return the **full** address as a single string without any extra text or explanation (only the address is needed).
- **Boolean-Type-Values**: 
  - return only the boolean string 'True' or 'False' value without any additional text or explanation.
  - If the condition fullfills the boolean condition, return 'True'. If not, return 'False'.
  - If no value is found, return empty string "" as default value.
- For all other value types, always provide the exact value as it appears in the document without any modifications or additional text.
""",
    "US_Commercial_Individual_Pages_Type_Forms": """
## FIELD SPECIFIC RULES:
- **Date-Type-Values**: 
  - When extracting date values, always strictly return them in ISO 8601 format: YYYY-MM-DD (for example, 2025-09-13). 
  - Return only the date string without any extra text, explanation, or punctuation.
- **Numeric-Type-Values (e.g., Amount, etc.):**
  - **Return only** the amount in the format **`$<number>`** (example: `$5000`, `$1000`).
  - **No spaces** after the `$` symbol.
  - **No commas** in the number.
  - **correct format**: `$5000`, `$1000`
  - **incorrect format**: `$ 5,000`
- **Percentage-Type-Values**: When extracting percentage values, strictly return only the numerical value followed by the "%" symbol (e.g., "5%", "12.5%"). Do not include any additional text or explanations.
- **Location-Addresses**: When extracting location or address values, return the **full** address as a single string without any extra text or explanation (only the address is needed).
- **Boolean-Type-Values**: 
  - return only the boolean string 'True' or 'False' value without any additional text or explanation.
  - If the condition fullfills the boolean condition, return 'True'. If not, return 'False'.
  - If no value is found, return empty string "" as default value.
- For all other value types, always provide the exact value as it appears in the document without any modifications or additional text.
""",
    "US_Personal": """
## FIELD SPECIFIC RULES:
- **Date-Type-Values**: 
  - When extracting date values, always strictly return them in ISO 8601 format: YYYY-MM-DD (for example, 2025-09-13). 
  - Return only the date string without any extra text, explanation, or punctuation.
- **Numeric-Type-Values (e.g., Amount, etc.):**
  - **Return only** the amount in the format **`$<number>`** (example: `$5000`, `$1000`).
  - **No spaces** after the `$` symbol.
  - **No commas** in the number.
  - **correct format**: `$5000`, `$1000`
  - **incorrect format**: `$ 5,000`
- **Percentage-Type-Values**: When extracting percentage values, strictly return only the numerical value followed by the "%" symbol (e.g., "5%", "12.5%"). Do not include any additional text or explanations.
- **Location-Addresses**: When extracting location or address values, return the **full** address as a single string without any extra text or explanation (only the address is needed).
- **Boolean-Type-Values**: 
  - return only the boolean string 'True' or 'False' value without any additional text or explanation.
  - If the condition fullfills the boolean condition, return 'True'. If not, return 'False'.
  - If no value is found, return empty string "" as default value.
- For all other value types, always provide the exact value as it appears in the document without any modifications or additional text.
- In case of Mailing Address, if the document has a mailing/postal address section, use that address as the mailing address. If not, use the insured's address as the mailing address.
""",
    "Canada_Commercial_Index_Schedule_Type_Forms": """
## FIELD SPECIFIC RULES:
- **Date-Type-Values**: 
  - When extracting date values, always strictly return them in ISO 8601 format: YYYY-MM-DD (for example, 2025-09-13). 
  - Return only the date string without any extra text, explanation, or punctuation.
- **Numeric-Type-Values (e.g., Amount, etc.):**
  - **Return only** the amount in the format **`$<number>`** (example: `$5000`, `$1000`).
  - **No spaces** after the `$` symbol.
  - **No commas** in the number.
  - **correct format**: `$5000`, `$1000`
  - **incorrect format**: `$ 5,000`
- **Percentage-Type-Values**: When extracting percentage values, strictly return only the numerical value followed by the "%" symbol (e.g., "5%", "12.5%"). Do not include any additional text or explanations.
- **Location-Addresses**: When extracting location or address values, return the **full** address as a single string without any extra text or explanation (only the address is needed).
- **Boolean-Type-Values**: 
  - return only the boolean string 'True' or 'False' value without any additional text or explanation.
  - If the condition fullfills the boolean condition, return 'True'. If not, return 'False'.
  - If no value is found, return empty string "" as default value.
- For all other value types, always provide the exact value as it appears in the document without any modifications or additional text.
""",
    "Canada_Commercial_Individual_Pages_Type_Forms": """
## FIELD SPECIFIC RULES:
- **Date-Type-Values**: 
  - When extracting date values, always strictly return them in ISO 8601 format: YYYY-MM-DD (for example, 2025-09-13). 
  - Return only the date string without any extra text, explanation, or punctuation.
- **Numeric-Type-Values (e.g., Amount, etc.):**
  - **Return only** the amount in the format **`$<number>`** (example: `$5000`, `$1000`).
  - **No spaces** after the `$` symbol.
  - **No commas** in the number.
  - **correct format**: `$5000`, `$1000`
  - **incorrect format**: `$ 5,000`
- **Percentage-Type-Values**: When extracting percentage values, strictly return only the numerical value followed by the "%" symbol (e.g., "5%", "12.5%"). Do not include any additional text or explanations.
- **Location-Addresses**: When extracting location or address values, return the **full** address as a single string without any extra text or explanation (only the address is needed).
- **Boolean-Type-Values**: 
  - return only the boolean string 'True' or 'False' value without any additional text or explanation.
  - If the condition fullfills the boolean condition, return 'True'. If not, return 'False'.
  - If no value is found, return empty string "" as default value.
- For all other value types, always provide the exact value as it appears in the document without any modifications or additional text.
""",
}

CUSTOM_FIELD_PROMPT_EXTRACTION = {
    "forms_and_endorsements": """
## FIELD SPECIFIC RULES:

**For field 'forms_and_endorsements'**: 
- You only have to find and return the name and number of the form/endorsement present for the mentioned LOB.
- Return them **exactly as they appear**.
- Do not return "edition date", "effective date", etc, only return the name and number of the form/endorsement.
- There are two possible format:
  - CASE 1: When you are provided with a schedule of forms and endorsements that will look like a list/table of forms and endorsements. Then extract the name and number of the same.
  - CASE 2: When you are provided with individual pages for each form/endorsement. Then you have to extract the name and number from the title of the page.
- (PARENT FIELD CONTEXT) will provide you the context of which LOB you need to look for the forms and endorsements.  
- Strictly do not include Forms, Endorsements, Exclusions that are mapped/present under "Others" Section. 
""",
    "common_exclusions": """
## FIELD SPECIFIC RULES:

**For field 'common_exclusions'**: 
- Go through the schedule of forms and endorsements and find the exclusion names mentioned in form names.
- Only focus on forms and endorsements for the LOB mentioned in User Input ( PARENT FIELD CONTEXT ).
- You only have to find the exclusion-names and return the value, Do not provide any other explainations around it.
- Exclusion names are always short phrases of (max 3-5 words) written as name of the form.
- DO not return all forms and endorsements names, only return names which are exclusions of this policy (LOB).
- Return them **exactly as they appear** in the form/endorsement Name.
""",
    "exclusions": """
## FIELD SPECIFIC RULES:

**For field 'exclusions'**: 
- Go through the provided pages and find the exclusion names mentioned in them.
- You only have to return boolean True or False for each exclusion from the Output schema based on whether the exclusion is present in the provided pages or not.
""",
    "basic_information": """
## FIELD SPECIFIC RULES:
- **Date-Values**: When extracting date values, always strictly return them in ISO 8601 format: YYYY-MM-DD (for example, 2025-09-13). Return only the date string without any extra text, explanation, or punctuation.
- **Percentage-Values**: When extracting percentage values, strictly return only the numerical value followed by the "%" symbol (e.g., "5%", "12.5%"). Do not include any additional text or explanations.
- **Location-Addresses**: When extracting location or address values, return the **full** address as a single string without any extra text or explanation (only the address is needed).
- In case of Mailing Address, if the document has a mailing address section, use that address as the mailing address. If not, use the insured's address as the mailing address.
- When the document lists multiple person names under the `insured` section:
  - Treat the first name as the primary `insured`.
  - Treat all subsequent names other than primary insured as `co_insured`, regardless of how they are labeled in the document.
""",
    "co_insured": """
## FIELD SPECIFIC RULES:
- **Date-Values**: When extracting date values, always strictly return them in ISO 8601 format: YYYY-MM-DD (for example, 2025-09-13). Return only the date string without any extra text, explanation, or punctuation.
- **Percentage-Values**: When extracting percentage values, strictly return only the numerical value followed by the "%" symbol (e.g., "5%", "12.5%"). Do not include any additional text or explanations.
- **Location-Addresses**: When extracting location or address values, return the **full** address as a single string without any extra text or explanation (only the address is needed).
- In case of Mailing Address, if the document has a mailing address section, use that address as the mailing address. If not, use the insured's address as the mailing address.
- When the document lists multiple person names under the `insured` section:
  - Treat the first name as the primary `insured`.
  - Treat all subsequent names other than primary insured as `co_insured`, regardless of how they are labeled in the document.
""",
}
