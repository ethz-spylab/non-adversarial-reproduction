import copy
import json
import argparse
import pathlib


def main(args):
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(args.input_file, "r") as f:
        input_data = json.load(f)

    with open(args.output_file, "w") as f:
        for current_type in input_data:
            current_template = current_type["prompt"]
            for current_instance in current_type["instances"]:
                current_prompt_obj = {
                    "messages": [
                        {
                            "role": "user",
                            "content": current_template.format(**current_instance),
                        }
                    ],
                    "text_type": current_type["text_type"],
                    "type": current_type["type"],
                }
                json.dump(current_prompt_obj, f)
                f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        help="Path to raw input data json file",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--output-file",
        help="Path output .jsonl file",
        type=pathlib.Path,
        required=True,
    )
    args = parser.parse_args()
    main(args)
