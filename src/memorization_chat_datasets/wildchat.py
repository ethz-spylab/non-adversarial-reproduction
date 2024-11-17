import argparse
import pathlib
import typing
import datasets
import tqdm
import json


TARGET_LANGUAGE = "English"
MIN_CONVERSATION_LENGTH = 500
SEED = 0xC0FFEE
MAX_CONVERSATIONS_PER_MODEL = 10_000


def main():
    args = _parse_args()
    output_dir = args.results_dir.resolve() / "wildchat"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = datasets.load_dataset("allenai/WildChat-1M", token=True)
    all_models = set(dataset["train"]["model"])

    candidate_subset = dataset["train"].filter(_is_candidate, num_proc=20)
    print("Number of candidates:", len(candidate_subset))

    # If there are multiple instances of the same prompt for a single model, only use one of them at random
    candidate_subset = candidate_subset.shuffle(seed=SEED)
    prompts_per_model = {model: set() for model in all_models}
    completions_per_model = {model: list() for model in all_models}

    for sample in tqdm.tqdm(candidate_subset):
        model = sample["model"]
        if len(completions_per_model[model]) >= MAX_CONVERSATIONS_PER_MODEL:
            continue
        prompt = sample["conversation"][0]["content"]
        completion = sample["conversation"][1]["content"]
        if prompt not in prompts_per_model[model]:
            prompts_per_model[model].add(prompt)
            prompt_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "completion": completion,
                "text_type": "misc",
                "type": "wildchat",
                "baseline": "wild",
                "wildchat_conversation_hash": sample["conversation_hash"],
            }
            completions_per_model[model].append(prompt_data)

    print("Number of prompts per model:")
    for model, completions in completions_per_model.items():
        print(f"{model}: {len(completions)}")

    for model, completions in completions_per_model.items():
        if len(completions) == 0:
            continue
        output_file_name = f"{model}_tempX.X.jsonl"
        with open(output_dir / output_file_name, "w") as f:
            for prompt_data in completions:
                f.write(json.dumps(prompt_data) + "\n")


def _is_candidate(sample: typing.Dict[str, typing.Any]) -> bool:
    conversation = sample["conversation"]
    return (
        sample["language"] == TARGET_LANGUAGE
        # Use non-redacted; else memorization scores might be off
        and not sample["redacted"]
        # Only consider first turn of conversations of the form (user -> assistant)
        and conversation[0]["role"] == "user"
        and conversation[1]["role"] == "assistant"
        # then require min. conversation length
        and len(conversation[1]["content"]) >= MIN_CONVERSATION_LENGTH
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        required=True,
        help="Path to results/ directory where completions will be stored",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
