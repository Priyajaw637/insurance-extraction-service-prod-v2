import base64
import json

import aiofiles

from app.logging_config import get_logger


logger = get_logger(__name__)


async def load_json(file_path: str):
    async with aiofiles.open(file_path, mode="r") as f:
        content = await f.read()
        data = json.loads(content)
    return data


async def save_json(data: dict, file_path):
    json_str = json.dumps(data, indent=2)
    async with aiofiles.open(file_path, mode="w") as f:
        await f.write(json_str)
    return True


async def pdf_to_base64(pdf_path: str) -> str:
    try:
        async with aiofiles.open(pdf_path, "rb") as f:
            pdf_bytes = await f.read()
            return base64.b64encode(pdf_bytes).decode("utf-8")
    except FileNotFoundError:
        logger.error(f"Error: The file at {pdf_path} was not found.")
        return ""
