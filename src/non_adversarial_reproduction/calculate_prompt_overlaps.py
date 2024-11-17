import argparse
import json
import pathlib
import numpy as np
import tqdm

def main(args: argparse.Namespace) -> None:
    results_folder = args.results_folder
    all_files = tuple(results_folder.glob("*.jsonl"))
    for file in (pbar := tqdm.tqdm(all_files, desc="Processing files")):
        with open(file, "r") as f:
            data = tuple(json.loads(line) for line in f)
        for current_result in tqdm.tqdm(data, desc="Processing results", leave=False):
            if "error" in current_result:
                pbar.write(f"Skipping entry in {file} due to error: {current_result['error']}")
                continue

            completion = current_result["completion"]
            suffix_lengths = None
            assert len(current_result["messages"]) > 0
            for message in current_result["messages"]:
                assert message["role"] in ("system", "user")
                prompt = message["content"]
                current_suffix_lengths = calculate_overlaps(prompt, completion)
                if suffix_lengths is None:
                    suffix_lengths = current_suffix_lengths
                else:
                    suffix_lengths = np.maximum(suffix_lengths, current_suffix_lengths)

            # DEBUG: Check that calculations are correct
            # for idx in range(len(completion)):
            #     if suffix_lengths[idx] > 0:
            #         assert any(completion[idx:idx + suffix_lengths[idx]] in message["content"] for message in current_result["messages"])
            #     assert all(idx + suffix_lengths[idx] == len(completion) or completion[idx:idx + suffix_lengths[idx] + 1] not in message["content"] for message in current_result["messages"])

            current_result["prompt_suffix_lengths"] = suffix_lengths.tolist()

        # Write results back to file
        with open(file, "w") as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")


def calculate_overlaps(prompt: str, completion: str) -> np.ndarray:
    # Very inefficient implementation, but fast enough for our purposes
    result = np.zeros(len(completion), dtype=int)
    for idx in range(len(completion)):
        current_length = 0
        while idx + current_length < len(completion) and completion[idx:idx + current_length + 1] in prompt:
            current_length += 1
        result[idx] = current_length
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-folder",
        help="Path to the results folder containing .jsonl files",
        type=pathlib.Path,
        required=True,
    )
    args = parser.parse_args()
    main(args)
