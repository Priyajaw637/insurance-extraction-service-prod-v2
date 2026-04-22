from fastapi import (
    APIRouter,
    HTTPException,
    status,
    Request,
    Depends,
    UploadFile,
    File,
    Form,
)
from typing import Optional, Literal
import os
import shutil
from uuid import uuid4
import aiofiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.config import ENV_PROJECT
from app.insurance_policy_processor.policy_orchestrator.agents.supervisor import (
    process_insurance_document,
)
import json
import asyncio
from app.modules.async_redis import redis_client
from app.insurance_policy_processor.policy_orchestrator.enums import (
    commercial_extraction_instructions,
    commercial_first_level_instruction,
    commercial_second_level_instruction,
)
from app.logging_config import get_logger

test_router = APIRouter()
logger = get_logger("policy_extract")

bearer_scheme = HTTPBearer(auto_error=True)


# Token verification dependency
async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> str:
    expected_token = ENV_PROJECT.TOKEN
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid authentication scheme")
    if credentials.credentials != expected_token:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
    return credentials.credentials


@test_router.post("/extract")
async def extract_pdf_from_local_schema(
    request: Request,
    pdf_file: UploadFile = File(...),
    tmp_dir: Optional[str] = "/tmp",
    token: str = Depends(verify_token),
):
    safe_filename = str(uuid4())
    local_pdf_path = os.path.join(tmp_dir, f"{safe_filename}.pdf")
    try:
        logger.info(
            f"Policy PDF extraction request received - filename: {pdf_file.filename}"
        )

        # Content type check
        if pdf_file.content_type != "application/pdf":
            logger.warning(f"Invalid file type uploaded: {pdf_file.content_type}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only PDF files are allowed.",
            )

        async with aiofiles.open(local_pdf_path, "wb") as out:
            await out.write(await pdf_file.read())

        async with aiofiles.open("Policy_Canada.json", "r") as f:
            schema = json.loads(await f.read())

        final_schema = {}
        lob_possibilities = []

        for lob in schema.keys():
            if lob_possibilities and lob in lob_possibilities:
                final_schema[lob] = schema[lob]

        logger.info(f"Processing policy PDF with content_access_id: {safe_filename}")

        asyncio.create_task(
            process_insurance_document(
                pdf_path=local_pdf_path,
                insurance_extraction_schema=final_schema if final_schema else schema,
                tmp_dir=tmp_dir,
                policy_document_id=safe_filename,
                list_lobs=lob_possibilities,
                tool_name="policy_checking",
                line_of_business="Commercial",
                country="Canada",
            )
        )

        logger.info(
            f"Policy PDF processing task created successfully for content_access_id: {safe_filename}"
        )
        return {
            "success": True,
            "content_access_id": safe_filename,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error processing policy PDF extraction request: {str(e)}", exc_info=True
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error while processing PDF: {e}",
        )
