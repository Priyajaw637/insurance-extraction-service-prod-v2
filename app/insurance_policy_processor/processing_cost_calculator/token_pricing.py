cost_mapping = {
    "gpt-4.1-nano": {"prompt": 0.100, "completion": 0.400, "cached": 0.025},
    "gpt-4.1-mini": {"prompt": 0.400, "completion": 1.600, "cached": 0.100},
    "gpt-5.1-nano": {"prompt": 0.005, "completion": 0.400, "cached": 0.003},
    "gpt-5.1-mini": {"prompt": 0.250, "completion": 2.000, "cached": 0.025},
    "models/gemini-2.5-flash-lite": {
        "prompt": 0.100,
        "completion": 0.400,
        "cached": 0.025,
    },
    "models/gemini-2.5-flash": {
        "prompt": 0.300,
        "completion": 2.500,
        "cached": 0.075,
    },
}
