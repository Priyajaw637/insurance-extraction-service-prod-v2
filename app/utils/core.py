import json
from typing import Any, Dict

import aiofiles

from app.logging_config import get_logger


logger = get_logger(__name__)


async def clean_response(response):
    clean_text = response.strip()

    if clean_text.startswith("```json"):
        clean_text = clean_text[7:]
    if clean_text.startswith("```"):
        clean_text = clean_text[3:]
    if clean_text.endswith("```"):
        clean_text = clean_text[:-3]

    try:
        return json.loads(clean_text)
    except Exception as e:
        logger.error(f"Error while cleaning response: {e}")
        return None


async def pdf_to_bytes(pdf_path: str) -> str:
    try:
        async with aiofiles.open(pdf_path, "rb") as f:
            pdf_bytes = await f.read()
            return pdf_bytes
    except FileNotFoundError:
        logger.error(f"Error: The file at {pdf_path} was not found.")
        return ""


async def create_output_schema(
    full_schema: Dict[str, Any],
    description: str = "List of page numbers",
    item_type: str = "integer",
    schema_title: str = "OuputJSONSchema",
):
    properties = {}
    mandatory = []
    for key in full_schema.keys():
        title = key.replace("_", " ").title()

        properties[key] = {
            # "title": title,
            # "description": description,
            "default": [],
            "type": "array",
            "items": {"type": item_type},
        }

        mandatory.append(key)

    output_schema = {
        "title": schema_title,
        "type": "object",
        "properties": properties,
        "required": mandatory,
    }

    # print(output_schema)
    return output_schema


async def create_output_schema_with_reason(
    full_schema: Dict[str, Any],
    schema_title: str = "OutputJSONSchemaWithReason",
) -> Dict[str, Any]:
    properties = {}
    mandatory = []

    for key in full_schema.keys():
        properties[key] = {
            "type": "object",
            "properties": {
                "pages": {"type": "array", "items": {"type": "integer"}, "default": []},
                "reasoning": {"type": "string", "default": ""},
            },
            "required": ["pages", "reasoning"],
        }
        mandatory.append(key)

    return {
        "title": schema_title,
        "type": "object",
        "properties": properties,
        "required": mandatory,
    }


async def bind_new_pages_to_original(orginal_pages: list, new_pages: list):
    binded_pages = []
    try:
        for page in new_pages:
            binded_pages.append(orginal_pages[page - 1])

    except IndexError:
        logger.error(
            f"Error: One of the new pages is out of range. New pages: {new_pages} and original pages: {orginal_pages}"
        )
        return orginal_pages

    return binded_pages


async def generate_short_schema_for_context(extraction_schema: Dict[str, Any]):
    short_schema = {}
    for key, value in extraction_schema.items():
        if isinstance(value, dict):
            short_schema[key] = {}
        elif isinstance(value, list):
            short_schema[key] = []
        else:
            short_schema[key] = ""

    return short_schema


def _build_schema_recursively(descriptive_schema, is_root=True):
    # case 1: object/dictionary
    if isinstance(descriptive_schema, dict):
        properties = {}
        for key, value in descriptive_schema.items():
            if is_root:
                if isinstance(value, dict):
                    # pass is_root=True to nested dictionaries to ensure their children are wrapped.
                    properties[key] = _build_schema_recursively(value, is_root=True)
                elif isinstance(value, list):
                    # for lists, the list handler itself will correctly manage the is_root flag for its items.
                    properties[key] = _build_schema_recursively(value, is_root=True)
                elif isinstance(value, str):
                    # wrap direct string fields
                    properties[key] = {
                        "type": "object",
                        "properties": {
                            "value": {
                                "type": "string",
                                "description": value,
                                "default": None,
                            },
                            "pages": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "default": [],
                            },
                        },
                        "default": {"value": None, "pages": []},
                    }
            else:
                # nested level (is_root=False) - plain structure for items in an array (e.g., a single vehicle)
                if isinstance(value, str):
                    properties[key] = {
                        "type": "string",
                        "description": value,
                        "default": None,
                    }
                else:
                    properties[key] = _build_schema_recursively(value, is_root=False)

        return {
            "type": "object",
            "properties": properties,
            "default": {},
        }

    # case 2: list/array
    elif isinstance(descriptive_schema, list):
        # array of objects - wrap object with pages at array item level
        if descriptive_schema and isinstance(descriptive_schema[0], dict):
            inner_object = _build_schema_recursively(
                descriptive_schema[0], is_root=False
            )
            return {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "value": inner_object,
                        "pages": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "default": [],
                        },
                    },
                    "default": {"value": {}, "pages": []},
                },
                "default": [],
            }

        # array of strings - wrap entire array at parent level
        elif descriptive_schema and isinstance(descriptive_schema[0], str):
            return {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "description": descriptive_schema[0],
                            "default": "",
                        },
                        "default": [],
                    },
                    "pages": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "default": [],
                    },
                },
                "default": {"value": [], "pages": []},
            }

        return {"type": "array", "items": {}, "default": []}

    # case 3: leaf node string
    elif isinstance(descriptive_schema, str):
        if is_root:
            return {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": descriptive_schema,
                        "default": None,
                    },
                    "pages": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "default": [],
                    },
                },
                "default": {"value": None, "pages": []},
            }
        else:
            return {
                "type": "string",
                "description": descriptive_schema,
                "default": None,
            }

    else:
        return {}


def create_gemini_schema(descriptive_schema):
    if not isinstance(descriptive_schema, dict):
        raise TypeError("top-level schema must be a dictionary")

    schema = _build_schema_recursively(descriptive_schema)
    schema["required"] = list(descriptive_schema.keys())

    return schema


async def map_relative_pages_to_original(data, original_pages_map: list):
    if isinstance(data, dict):
        # check if this is a leaf node with the 'value'/'pages' structure
        if "value" in data and "pages" in data and isinstance(data.get("pages"), list):
            relative_pages = data["pages"]
            # use the binding function to map relative pages to the original ones
            original_pages = await bind_new_pages_to_original(
                orginal_pages=original_pages_map, new_pages=relative_pages
            )
            data["pages"] = original_pages

        for value in data.values():
            await map_relative_pages_to_original(value, original_pages_map)

    elif isinstance(data, list):
        # recurse into each item in the list
        for item in data:
            await map_relative_pages_to_original(item, original_pages_map)

    return data
