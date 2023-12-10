_OLDER_MODEL_CONFIG = {
    "gpt-4-0314": {
        "max_tokens": 8192,
        "prompt_cost_per_token": 0.00003,
        "completion_cost_per_token": 0.00006,
    },
    "gpt-4-0613": {
        "max_tokens": 8192,
        "prompt_cost_per_token": 0.00003,
        "completion_cost_per_token": 0.00006,
    },
    "gpt-4-32k-0314": {
        "max_tokens": 32768,
        "prompt_cost_per_token": 0.00006,
        "completion_cost_per_token": 0.00012,
    },
    "gpt-4-32k-0613": {
        "max_tokens": 32768,
        "prompt_cost_per_token": 0.00006,
        "completion_cost_per_token": 0.00012,
    },
    "gpt-3.5-turbo-0301": {
        "max_tokens": 4096,
        "prompt_cost_per_token": 0.0000015,
        "completion_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-0613": {
        "max_tokens": 4096,
        "prompt_cost_per_token": 0.0000015,
        "completion_cost_per_token": 0.000002,
    },
    "gpt-3.5-turbo-16k-0613": {
        "max_tokens": 16384,
        "prompt_cost_per_token": 0.000003,
        "completion_cost_per_token": 0.000004,
    },
    "gpt-3.5-turbo-instruct": {
        "max_tokens": 8192,
        "prompt_cost_per_token": 0.0000015,
        "completion_cost_per_token": 0.000002,
    },
}

_NEWER_MODEL_CONFIG = {
    "gpt-4-1106-preview": {
        "max_tokens": 128000,
        "prompt_cost_per_token": 0.00001,
        "completion_cost_per_token": 0.00003,
    },
    "gpt-3.5-turbo-1106": {
        "max_tokens": 4096,
        "prompt_cost_per_token": 0.000001,
        "completion_cost_per_token": 0.000002,
    },
}

MODEL_POINT = {
    "gpt-4": "gpt-4-0613",
    "gpt-4-32k": "gpt-4-32k-0613",
    "gpt-4-128k": "gpt-4-1106-preview",
    "gpt-3.5-turbo": "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-16k": "gpt-3.5-turbo-16k-0613",
}

EMBEDDING_CONFIG = {
    "text-ada-embeddings-002": {
        "max_tokens": 8192,
        "prompt_cost_per_token": 0.0000001,
    },
}


MODEL_CONFIG = {}
MODEL_CONFIG.update(_OLDER_MODEL_CONFIG)
MODEL_CONFIG.update(_NEWER_MODEL_CONFIG)

_MODEL_POINT_CONFIG = {
    "gpt-4": MODEL_CONFIG[MODEL_POINT["gpt-4"]],
    "gpt-4-32k": MODEL_CONFIG[MODEL_POINT["gpt-4-32k"]],
    "gpt-4-128k": MODEL_CONFIG[MODEL_POINT["gpt-4-128k"]],
    "gpt-3.5-turbo": MODEL_CONFIG[MODEL_POINT["gpt-3.5-turbo"]],
    "gpt-3.5-turbo-16k": MODEL_CONFIG[MODEL_POINT["gpt-3.5-turbo-16k"]],
}

MODEL_CONFIG.update(_MODEL_POINT_CONFIG)
