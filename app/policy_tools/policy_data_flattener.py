async def flatten_json_with_values(data):
    """
    Recursively flattens a JSON structure where fields may be dicts with 'value' and 'pages',
    converting each such dictionary to just its 'value', handling all nested lists and dicts.

    Args:
        data (dict or list or any): The original JSON-like structure.

    Returns:
        The flattened JSON-like structure with 'value' extracted everywhere possible.
    """
    if isinstance(data, dict):
        # Check if data itself has 'value' key, if so flatten that first
        if "value" in data and isinstance(data["value"], (dict, list)):
            return await flatten_json_with_values(data["value"])
        if "value" in data and not isinstance(data["value"], (dict, list)):
            # Return simple value directly
            return data["value"]
        # Otherwise, recurse into all keys
        return {k: await flatten_json_with_values(v) for k, v in data.items()}

    elif isinstance(data, list):
        # Recursively process each item in the list
        return [await flatten_json_with_values(item) for item in data]

    else:
        # Base case: simple value, return as is
        return data
