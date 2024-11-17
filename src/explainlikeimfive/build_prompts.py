import argparse
import json
import copy
import pathlib
import numpy as np
import tiktoken
import writing_prompts.models
import re


MIN_WORDS = 100
TARGET_NUM_WORDS = 500
TARGET_NUM_SUBMISSIONS = 1000
PROMPT_TEMPLATE = "Provide a layperson-friendly explanation of the following: {prompt}"
TITLE_PREFIX = "ELI5"
DROP_TERM_LIST = (
    "just happened",
    "news",
)


def main() -> None:
    args = _parse_args()
    input_submissions_path = args.input_submissions
    input_comments_path = args.input_comments
    output_prompts_file = args.output_prompts_file.resolve()
    results_dir = args.results_dir.resolve()
    ouput_humans_file = results_dir / output_prompts_file.stem / "humans_temp0.0.jsonl"

    print(
        f"Writing LLM prompts to {output_prompts_file} and human replies to {ouput_humans_file}"
    )
    output_prompts_file.parent.mkdir(parents=True, exist_ok=True)
    ouput_humans_file.parent.mkdir(parents=True, exist_ok=True)

    # For everything, store post id. For comments, also store comment id (and author)
    # This allows matching LLM and human responses during evaluation.
    #
    # 1. Load and filter submissions
    #  - Titles must start with "ELI5", else they were removed
    #  - Filter titles that are likely to be rejected/too vague
    # 2. Load, filter and match comments
    #  - Top-level replies to one of the filtered posts
    #  - Min. length, no mod comments, not removed
    # 3. Remove submissions with no comments
    # 4. Select a final subset of submissions+comments (closest to target number of words; one comment per submission)
    # 5. Write prompts for LLMs
    # 6. Write results file for humans in same format as LLM completions

    # Load and pre-filter submissions
    submissions = {}
    title_regex = re.compile(r"^[\s\:\.\,\-\"]*(.*?)[\s\:\.\,\-\"]*$")
    num_dropped_due_to_drop_terms = 0
    with open(input_submissions_path) as f:
        for line in f:
            submission = writing_prompts.models.PostRecord.model_validate_json(line)

            if submission.was_removed:
                continue

            submission_title = submission.title.strip()
            if not submission_title.upper().startswith(TITLE_PREFIX.upper()):
                continue

            # Drop submissions with certain terms in the title
            if any(term.lower() in submission_title.lower() for term in DROP_TERM_LIST):
                num_dropped_due_to_drop_terms += 1
                continue

            submission_prompt = title_regex.match(submission_title[len(TITLE_PREFIX):]).group(1).strip()
            submissions[submission.id] = {
                "submission": submission,
                "prompt": submission_prompt,
                "comments": [],
            }
    print(f"Loaded {len(submissions)} candidate submissions")
    print(f"Dropped {num_dropped_due_to_drop_terms} submissions due to drop terms")

    # Load, filter, and match comments
    num_too_short = 0
    with open(input_comments_path) as f:
        for line in f:
            comment = writing_prompts.models.CommentRecord.model_validate_json(line)
            comment_text = comment.text.strip()
            parent_post_id = comment.link_id[len("t3_") :]

            # NB: link_id = parent post, parent_id = direct parent (comment or submission)

            # Only comments with retained submissions
            if parent_post_id not in submissions:
                # NB: Some parent submissions are not available at all; e.g., from earlier months
                continue

            # Only top-level comments
            if not comment.parent_id.startswith("t3_"):
                continue

            # Only non-deleted comments
            if comment.was_removed:
                continue

            # No mod/admin comments
            if (
                comment.distinguished_as is not None
                and comment.distinguished_as.lower().strip() in ("moderator", "admin")
            ):
                continue

            # No too short comments
            if count_words(comment_text) < MIN_WORDS:
                num_too_short += 1
                continue

            submissions[parent_post_id]["comments"].append(comment)

    print(f"Dropped {num_too_short} comments for being too short")

    # Remove submissions with no comments
    submissions = {
        submission_id: submission_data
        for submission_id, submission_data in submissions.items()
        if len(submission_data["comments"]) > 0
    }
    print(f"Retained {len(submissions)} submissions with at least one writing reply")
    print(
        f"Have a total of {sum(len(submission_data['comments']) for submission_data in submissions.values())} comments before final selection"
    )

    # Select the final subset of submissions+comments
    enc = tiktoken.get_encoding("cl100k_base")
    final_candidates = []
    for submission in submissions.values():
        selected_comment = min(
            submission["comments"],
            key=lambda comment: abs(TARGET_NUM_WORDS - count_words(comment.text)),
        )
        final_candidates.append(
            {
                "submission": submission["submission"],
                "prompt": submission["prompt"],
                "comment": selected_comment,
                "num_words": count_words(selected_comment.text),
            }
        )
    selected_submissions = sorted(
        final_candidates,
        key=lambda candidate: abs(TARGET_NUM_WORDS - candidate["num_words"]),
    )[:TARGET_NUM_SUBMISSIONS]
    assert len(selected_submissions) == TARGET_NUM_SUBMISSIONS
    selected_num_tokens = tuple(
        len(enc.encode(candidate["comment"].text)) for candidate in selected_submissions
    )
    print(
        f"Selected {len(selected_submissions)} submissions with closest to {TARGET_NUM_WORDS} words"
    )
    word_counts = np.array([candidate["num_words"] for candidate in selected_submissions])
    print(
        f"Average words for selected comments: {np.mean(word_counts):.1f} (min {np.min(word_counts)}, max {np.max(word_counts)})"
    )
    print(
        f"Average tokens for selected comments: {np.mean(selected_num_tokens):.1f} (total {np.sum(selected_num_tokens)})"
    )

    # Write prompts for LLMs and responses from humans
    with (
        open(output_prompts_file, "w") as f_prompts,
        open(ouput_humans_file, "w") as f_humans,
    ):
        for submission in selected_submissions:
            prompt_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(prompt=submission["prompt"]),
                    }
                ],
                "text_type": "expository",
                "type": "explainlikeimfive",
                "baseline": "explainlikeimfive",
                "reddit_submission_id": submission["submission"].id,
            }
            f_prompts.write(json.dumps(prompt_data) + "\n")

            comment = submission["comment"]
            human_completion_data = copy.deepcopy(prompt_data)
            human_completion_data["completion"] = comment.text.strip()
            human_completion_data["reddit_comment_id"] = comment.id
            human_completion_data["reddit_author"] = comment.author
            f_humans.write(json.dumps(human_completion_data) + "\n")


def count_words(text: str) -> int:
    words = text.split()
    # Remove leading and trailing punctuation from each word
    words = [word.strip(".,!?") for word in words]
    # Remove empty words
    words = [word for word in words if word]
    return len(words)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-submissions",
        type=pathlib.Path,
        required=True,
        help="Path to input jsonl file for submissions",
    )
    parser.add_argument(
        "--input-comments",
        type=pathlib.Path,
        required=True,
        help="Path to input jsonl file for comments",
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
