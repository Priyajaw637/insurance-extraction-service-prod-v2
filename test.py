import json


def reformat_json() -> None:
    """
    Reads a JSON file and rewrites it with 2-space indentation.
    """
    file_path = "canada_new.json"
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Reformatted JSON file with 2-space indentation: {file_path}")


def get_cost(data: dict = None) -> None:

    if not data:
        file_path = "without.json"
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

    coverage_mapping_cost = data.get("coverage_mapping_cost")
    coverage_detail_mapping_cost = data.get("coverage_detail_mapping_cost")
    extraction_cost = data.get("extraction_cost")
    token_consumption = data.get("token_consumption")

    print("total_token_usage: ", token_consumption)

    input_token = 0
    output_token = 0
    thinking_token = 0
    for cost in coverage_mapping_cost:
        input_token += cost.get("input_tokens") if cost.get("input_tokens") else 0
        output_token += cost.get("output_tokens") if cost.get("output_tokens") else 0
        thinking_token += (
            cost.get("thinking_tokens") if cost.get("thinking_tokens") else 0
        )

    print("===== Coverage Mapping Cost =====")
    print("input_token: ", input_token)
    print("output_token: ", output_token)
    print("thinking_token: ", thinking_token)
    print("=================================")

    input_token = 0
    output_token = 0
    thinking_token = 0
    for cost in coverage_detail_mapping_cost:
        input_token += cost.get("input_tokens") if cost.get("input_tokens") else 0
        output_token += cost.get("output_tokens") if cost.get("output_tokens") else 0
        thinking_token += (
            cost.get("thinking_tokens") if cost.get("thinking_tokens") else 0
        )
    print("===== Coverage Detail Cost =====")
    print("input_token: ", input_token)
    print("output_token: ", output_token)
    print("thinking_token: ", thinking_token)
    print("=================================")

    input_token = 0
    output_token = 0
    thinking_token = 0

    for cost in extraction_cost:
        input_token += cost.get("input_tokens") if cost.get("input_tokens") else 0
        output_token += cost.get("output_tokens") if cost.get("output_tokens") else 0
        thinking_token += (
            cost.get("thinking_tokens") if cost.get("thinking_tokens") else 0
        )

    print("===== Extraction Cost =====")
    print("input_token: ", input_token)
    print("output_token: ", output_token)
    print("thinking_token: ", thinking_token)
    print("=================================")


if __name__ == "__main__":
    get_cost()
