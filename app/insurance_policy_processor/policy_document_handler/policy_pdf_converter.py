import base64
import multiprocessing as mp
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Literal

import aiofiles
import fitz
import psutil

# mp.set_start_method("fork")         # on Linux/macOS; no-op on Windows


def render_page_range(args):
    idx, total_procs, pdf_path, zoom, outdir = args
    doc = fitz.open(pdf_path)
    seg = doc.page_count // total_procs + 1
    start = idx * seg
    end = min((idx + 1) * seg, doc.page_count)
    mat = fitz.Matrix(zoom, zoom)
    for i in range(start, end):
        pix = doc[i].get_pixmap(matrix=mat, alpha=False)
        pix.save(os.path.join(outdir, f"page_{i:04d}.png"))
    doc.close()


def policy_pdf_to_image(
    pdf_path,
    outdir=None,
    zoom=2.0,
):
    if outdir is None:
        outdir = tempfile.gettempdir()
    os.makedirs(outdir, exist_ok=True)
    proc = psutil.Process()
    t0 = time.perf_counter()
    mem0 = psutil.virtual_memory().used
    cpu0 = sum(proc.cpu_times()[:2])
    mem0 = psutil.virtual_memory().used

    cpus = mp.cpu_count()
    args = [(i, cpus, pdf_path, zoom, outdir) for i in range(cpus)]
    with mp.Pool(processes=cpus) as pool:
        pool.map(render_page_range, args)

    # Gather all image paths, sort by page number
    image_files = [
        os.path.join(outdir, fname)
        for fname in os.listdir(outdir)
        if fname.startswith("page_") and fname.endswith(".png")
    ]
    image_files.sort()  # page_0000.png, page_0001.png, ...

    wall = time.perf_counter() - t0
    cpu = sum(proc.cpu_times()[:2]) - cpu0
    ram = (psutil.virtual_memory().used - mem0) / (1024**2)
    print(f"[PyMuPDF] wall={wall:.2f}s  CPU-sec={cpu:.2f}s  ΔRAM={ram:.1f} MB")

    paths = [str(Path(p)) for p in image_files]
    return paths


async def policy_image_to_base64_bytes(
    return_type: Literal["local_path", "base64", "image_bytes"] = "local_path",
    image_files=[],
):
    return_list = []
    if return_type == "local_path":
        return_list = image_files
    elif return_type == "base64":
        base64_list = []
        counter = 0
        for img_path in image_files:
            async with aiofiles.open(img_path, "rb") as f:
                encoded = base64.b64encode(await f.read()).decode("utf-8")
                base64_list.append({"page_number": counter, "image": encoded})
                counter += 1

        return_list = base64_list

    elif return_type == "image_bytes":
        image_bytes_list = []
        counter = 0

        for img_path in image_files:
            async with aiofiles.open(img_path, "rb") as f:
                image_bytes_list.append(
                    {"page_number": counter, "image": await f.read()}
                )
                counter += 1

        return_list = image_bytes_list
    else:
        raise ValueError("return_type must be 'local_path' or 'base64'")

    return return_list


async def remove_policy_images(outdir=None):
    try:
        shutil.rmtree(path=outdir, ignore_errors=True)
    except Exception as e:
        print(f"Failed to remove directory {outdir}. Reason: {e}")
