import asyncio
import os
from uuid import uuid4

from app.insurance_policy_processor.policy_orchestrator.modules.gemini import (
    gemini_client,
)
from app.logging_config import get_logger


logger = get_logger(__name__)


async def upload_policy_file(pdf_path: str, mime_type: str):
    if not os.path.exists(pdf_path):
        logger.error(f"File not found at '{pdf_path}'. Please update the path.")
        raise ("NO PDF FILE FOUND AT PROVIDED PATH")

    logger.info(f"Uploading file: {pdf_path} to Gemini with mime_type: {mime_type}")
    name = str(uuid4())
    uploaded_file = await gemini_client.aio.files.upload(
        file=pdf_path, config={"name": name, "mime_type": mime_type}
    )

    logger.info(f"Upload successful with name: '{name}'")

    return uploaded_file


async def get_policy_file_metadata(file_name: str):
    logger.debug(f"Getting file metadata for: {file_name}")
    myfile = await gemini_client.aio.files.get(name=file_name)
    return myfile


async def delete_policy_file_from_gemini(file_name: str):
    try:
        logger.info(f"Deleting file from Gemini: {file_name}")
        await gemini_client.aio.files.delete(name=file_name)
        logger.info(f"File successfully deleted from Gemini: {file_name}")
    except Exception as e:
        logger.error(f"Error deleting file from Gemini: {str(e)}", exc_info=True)


async def delete_policy_file_from_gemini_future(file_name: str):
    try:
        await asyncio.sleep(600)
        logger.info(f"Deleting file from Gemini: {file_name}")
        await gemini_client.aio.files.delete(name=file_name)
        logger.info(f"File successfully deleted from Gemini: {file_name}")
    except Exception as e:
        logger.error(f"Error deleting file from Gemini: {str(e)}", exc_info=True)
