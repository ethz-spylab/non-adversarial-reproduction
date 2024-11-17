#!/usr/bin/env bash

BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

TEMPERATURES=("0.0" "0.7")
MODELS=(
    "openai/gpt-4o-mini-2024-07-18"
    "openai/gpt-4o-2024-05-13"
    "openai/gpt-4-turbo-2024-04-09"
    "anthropic/claude-3-haiku"
    "anthropic/claude-3.5-sonnet"
    "anthropic/claude-3-opus"
    "meta-llama/llama-3.1-8b-instruct"
    "meta-llama/llama-3.1-70b-instruct"
    "meta-llama/llama-3.1-405b-instruct"
    "openai/o1-mini-2024-09-12"
    "openai/o1-preview-2024-09-12"
    "google/gemini-1.5-flash-002"
    "google/gemini-1.5-pro-002"
)
MAX_OUTPUT_TOKENS=4500 # all models except o1 manage with 1250
MAX_PARALLEL_DEFAULT=20
MAX_PARALLEL_REDUCED=16
INPUT_FILE="${BASE_DIR}/data/prompts/writing_prompts.jsonl"
OUTPUT_DIR="${BASE_DIR}/data/results/"
BASE_SEED=0xC0FFEE
NUM_SEEDS=1
SEED_RANGE=$(seq $BASE_SEED $((BASE_SEED + NUM_SEEDS - 1)))

for model in ${MODELS[@]}; do
    # Use provider-specific API for openai and google models
    if [[ $model == openai/* ]]; then
        api_choice="openai"
        use_openai_arg="--use-openai"
        use_google_arg=""
    elif [[ $model == google/* ]]; then
        api_choice="google"
        use_openai_arg=""
        use_google_arg="--use-google"
    else
        api_choice="openrouter"
        use_openai_arg=""
        use_google_arg=""
    fi

    # If $model starts with meta-llama, then reduce parallelism
    if [[ $model == meta-llama/* ]] || [[ $model == google/* ]]; then
        max_parallel=${MAX_PARALLEL_REDUCED}
    else
        max_parallel=${MAX_PARALLEL_DEFAULT}
    fi

    echo "# Model: ${model} (API: ${api_choice})"
    for temperature in ${TEMPERATURES[@]}; do
        # Special case for o1 models, because they only support temperature=1
        if [[ $model == openai/o1* ]]; then
            if [[ $temperature != "0.7" ]]; then
                echo "Skipping temperature ${temperature} for ${model} because not supported"
                continue
            else
                echo "Using temperature 1 instead of ${temperature} for ${model}"
                temperature="1"
            fi
        fi

        python src/non_adversarial_reproduction/generate.py \
            $use_openai_arg \
            $use_google_arg \
            --model "${model}" \
            --data_path "${INPUT_FILE}" \
            --temperature "${temperature}" \
            --max_new_tokens "${MAX_OUTPUT_TOKENS}" \
            --max-parallel-requests "${max_parallel}" \
            --output_path "${OUTPUT_DIR}" \
            --seeds $SEED_RANGE
    done
    echo
done