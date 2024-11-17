import argparse
import csv
import json
import pathlib


PROMPT_TEMPLATE = (
    "Write a short essay (around 500 words). Your assignment is as follows: {prompt}"
)


def main() -> None:
    args = _parse_args()
    input_base_dir = args.raw_data_dir
    input_corpus_file = input_base_dir / "persuade_corpus_2.0.csv"
    input_compare_file_train = input_base_dir / "feedback-prize-2021" / "train.csv"
    input_compare_dir_test = input_base_dir / "feedback-prize-2021" / "test"
    input_control_file = args.control_prompts_file
    input_blacklist_file = args.id_blacklist_file
    output_prompts_file = args.output_prompts_file.resolve()
    results_dir = args.results_dir.resolve()
    ouput_humans_file = results_dir / output_prompts_file.stem / "humans_temp0.0.jsonl"
    ouput_humans_all_file = results_dir / output_prompts_file.stem / "humans-all_temp0.0.jsonl"

    print(
        f"Writing LLM prompts to {output_prompts_file} and human replies to {ouput_humans_file}"
    )
    output_prompts_file.parent.mkdir(parents=True, exist_ok=True)
    ouput_humans_file.parent.mkdir(parents=True, exist_ok=True)

    # First, determine eassays that were only released in the 2.0 version of the dataset
    with open(input_compare_file_train, "r") as f:
        reader = csv.reader(f)
        _ = next(reader)  # Skip header
        ids_to_ignore = {row[0] for row in reader}
    ids_to_ignore.update({test_file.stem for test_file in input_compare_dir_test.glob("*.txt")})
    print(f"Ignoring {len(ids_to_ignore)} essays that were released before PERSUADE 2.0")

    # Add blacklisted IDs to the ignore list
    with open(input_blacklist_file, "r") as f:
        ids_to_ignore.update(row.strip() for row in f)
    print(f"Ignoring {len(ids_to_ignore)} essays after including blacklisted IDs")

    with open(input_corpus_file, "r") as f:
        reader = csv.reader(f)
        header = next(reader)
        # NB: 2.0 has two ids, and essay_id_comp matches ids from version 1.0
        full_raw_data = tuple(row for row in reader)
    essay_id_idx = header.index("essay_id_comp")
    print(f"Loaded {len({row[essay_id_idx] for row in full_raw_data})} raw essays")

    # Only keep essays that were independent writing (w/o source text)
    task_idx = header.index("task")
    full_raw_data = tuple(row for row in full_raw_data if row[task_idx] == "Independent")
    print(f"Have {len({row[essay_id_idx] for row in full_raw_data})} essays that are independent writing")

    # Deduplicate and make sure things are consistent
    # NB: one essay has many entries in the dataset, b/c entries correspond to grading
    rows_by_id = dict()
    for row in full_raw_data:
        if row[1] not in rows_by_id:
            rows_by_id[row[1]] = []
        rows_by_id[row[1]].append(row)

    TARGET_FIELDS = (
        "essay_id",
        "essay_id_comp",
        "full_text",
        "holistic_essay_score",
        "prompt_name",
        "assignment",
        "gender",
        "grade_level",
        "ell_status",
        "race_ethnicity",
        "economically_disadvantaged",
        "student_disability_status",
        "essay_word_count",
    )

    final_essays_full = []
    for _, rows in rows_by_id.items():
        current_data = {field: rows[0][header.index(field)] for field in TARGET_FIELDS}
        for row in rows:
            for field in TARGET_FIELDS:
                if current_data[field] != row[header.index(field)]:
                    raise ValueError("Mismatch", field, current_data[field], row[header.index(field)])
        final_essays_full.append(current_data)

    # Finally, keep all essays with score at least 5 (out of 6) and not in blacklist
    final_essays = tuple(
        essay for essay in final_essays_full
        if int(essay["holistic_essay_score"]) >= 5 and essay["essay_id_comp"] not in ids_to_ignore
    )
    print(f"Have {len(final_essays)} final remaining essays only in PERSUADE 2.0 and with score at least 5")

    # Write outputs
    for file_path, output_data in (
        (ouput_humans_all_file, final_essays_full),
        (ouput_humans_file, final_essays),
    ):
        with open(file_path, "w") as f_humans:
            for essay in output_data:
                human_completion_data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": PROMPT_TEMPLATE.format(prompt=essay["assignment"]),
                        }
                    ],
                    "text_type": "argumentative",
                    "type": "essay_persuade",
                    "baseline": "essay_persuade",
                    "essay_prompt_name": essay["prompt_name"],
                }
                human_completion_data["completion"] = essay["full_text"]
                human_completion_data["essay_id"] = essay["essay_id_comp"]
                f_humans.write(json.dumps(human_completion_data) + "\n")

    # Write prompts
    unique_prompts = set((essay["prompt_name"], essay["assignment"]) for essay in final_essays)
    print("Unique prompts for LLMs:", len(unique_prompts))
    with open(input_control_file, "r") as f_control:
        control_prompt_data = tuple(json.loads(row) for row in f_control)
    print("Control prompts for LLMs:", len(control_prompt_data))
    with open(output_prompts_file, "w") as f_prompts:
        for prompt_name, prompt_text in unique_prompts:
            prompt_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(prompt=prompt_text),
                    }
                ],
                "text_type": "argumentative",
                "type": "essay_persuade",
                "baseline": "essay_persuade",
                "essay_prompt_name": prompt_name,
            }
            f_prompts.write(json.dumps(prompt_data) + "\n")
        for control_prompt in control_prompt_data:
            prompt_text = control_prompt["assignment"]
            prompt_name = control_prompt["prompt_name"]
            prompt_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(prompt=prompt_text),
                    }
                ],
                "text_type": "argumentative",
                "type": "essay_persuade",
                "baseline": "essay_persuade",
                "essay_prompt_name": prompt_name,
                "is_control": True,
            }
            f_prompts.write(json.dumps(prompt_data) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-data-dir",
        type=pathlib.Path,
        required=True,
        help="Path to directory containing raw PERSUADE datasets",
    )
    parser.add_argument(
        "--control-prompts-file",
        type=pathlib.Path,
        required=True,
        help="Path to jsonl file containing control prompts for LLMs",
    )
    parser.add_argument(
        "--id-blacklist-file",
        type=pathlib.Path,
        required=True,
        help="Path to file containing IDs of essays to ignore",
    )
    parser.add_argument(
        "--output-prompts-file",
        type=pathlib.Path,
        required=True,
        help="Path to output jsonl file into which to write prompts for LLMs",
    )
    parser.add_argument(
        "--results-dir",
        type=pathlib.Path,
        required=True,
        help="Path to results/ directory where human replies will be stored",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
