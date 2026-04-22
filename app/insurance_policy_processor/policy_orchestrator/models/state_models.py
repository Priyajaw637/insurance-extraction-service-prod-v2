from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class LobIdentifierResponse(BaseModel):
    identified_lobs: List[str]
    lob_extraction_source: Literal["Single Summary Page", "Individual Declarations"]
    lob_source_reasoning: str = Field(
        default="",
        description="Reasoning for why this source was choosen and other source was not in MAX 2 line",
    )
    identified_forms_type: Literal[
        "Index_Schedule_Type_Forms", "Individual_Pages_Type_Forms"
    ] = Field(default="Index_Schedule_Type_Forms")
    form_type_reasoning: str = Field(
        default="",
        description="Reasoning for why this form type was choosen in MAX 2 line.",
    )


class FormTypeIdentifierResponse(BaseModel):
    identified_forms_type: Literal[
        "Index_Schedule_Type_Forms", "Individual_Pages_Type_Forms"
    ] = Field(default="Index_Schedule_Type_Forms")
    form_type_reasoning: str = Field(
        default="",
        description="Reasoning for why this form type was choosen in MAX 2 line.",
    )


class InsuranceDocumentState(BaseModel):
    tmp_dir: str
    policy_document_id: str = Field(default="")
    insurance_extraction_schema: Dict[str, Any]

    # PDF Info
    pdf_path: str
    pdf_length: Optional[int] = Field(default=None)
    uploaded_pdf: Optional[Any] = Field(default=None)

    # Schema Info
    schema_summary: Optional[Dict[str, str]] = Field(default=None)
    coverage_document_structure: Optional[str] = Field(default=None)
    initial_schema_mapping: Optional[Dict[str, Any]] = Field(default=None)

    # Identified LOB
    identified_lobs: Optional[List[str]] = Field(default_factory=list)
    identified_forms_type: Literal[
        "Index_Schedule_Type_Forms",
        "Individual_Pages_Type_Forms",
    ] = Field(default="Index_Schedule_Type_Forms")

    # Coverage Data
    coverage_mapping_result: Optional[Dict[str, Any]] = Field(default=None)
    coverage_mapping_cost: List[Dict[str, Any]] = Field(default_factory=list)

    # Coverage Detail Data
    coverage_detail_mapping_result: Optional[Dict[str, Any]] = Field(default=None)
    coverage_detail_mapping_cost: List[Dict[str, Any]] = Field(default_factory=list)

    # Policy Extraction Data
    extracted_policy_data: Optional[dict] = Field(default=None)
    extraction_cost: List[Dict[str, Any]] = Field(default_factory=list)

    token_consumption: Dict[str, Any] = Field(default=None)
    list_lobs: Optional[List[str]] = Field(default=None)

    # Custom Instructions
    line_of_business: Optional[str] = Field(default=None)
    country: Optional[str] = Field(default=None)
    tool_name: Optional[str] = Field(default=None)

    first_level_instruction: Optional[str] = Field(default="")
    second_level_instruction: Optional[str] = Field(default="")
    extraction_instruction: Optional[str] = Field(default="")
