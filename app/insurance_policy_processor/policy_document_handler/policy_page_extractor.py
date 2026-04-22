import asyncio
import concurrent.futures
from typing import List

import fitz

# MAX WORKERS
MAX_WORKERS = 15
executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)


async def read_policy_pdf(path: str) -> fitz.Document:
    """Open a PDF document using the limited thread pool."""
    loop = asyncio.get_running_loop()
    # run_in_executor(executor, function, *args)
    return await loop.run_in_executor(executor, fitz.open, path)


async def write_policy_pdf(doc: fitz.Document, output_path: str) -> str:
    """Save a PyMuPDF document using the limited thread pool."""
    loop = asyncio.get_running_loop()

    def write():
        doc.save(output_path)
        doc.close()

    await loop.run_in_executor(executor, write)
    return output_path


async def extract_policy_pages(
    reader: fitz.Document, page_numbers: List[int]
) -> fitz.Document:
    """Extract specific pages from a PyMuPDF document."""
    new_doc = fitz.open()
    total_pages = reader.page_count
    for page_num in page_numbers:
        if 0 <= page_num < total_pages:
            new_doc.insert_pdf(reader, from_page=page_num, to_page=page_num)
    return new_doc


async def filter_policy_pdf_pages(
    pdf_path: str, page_numbers: List[int], output_path: str
) -> str:
    try:
        page_numbers = sorted(set(page_numbers))
        reader = await read_policy_pdf(pdf_path)
        zero_based_pages = [p - 1 for p in page_numbers]
        writer = await extract_policy_pages(reader, zero_based_pages)
        reader.close()
        return await write_policy_pdf(writer, output_path)
    except Exception as e:
        return pdf_path
