import fitz  # PyMuPDF

from app.insurance_policy_processor.policy_orchestrator.models.state_models import (
    InsuranceDocumentState,
)
from app.logging_config import get_logger


logger = get_logger(__name__)


async def extract_policy_document_metadata(state: InsuranceDocumentState):
    try:
        document = fitz.open(state.pdf_path)
        number_of_pages = document.page_count
        document.close()

        state.pdf_length = number_of_pages
        return state
    except Exception as e:
        logger.error(f"ERROR in pdf reader: {e}", exc_info=True)
        raise state
