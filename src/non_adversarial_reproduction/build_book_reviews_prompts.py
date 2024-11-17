import argparse
import json
import copy
import pathlib
import numpy as np
import tiktoken
import tqdm
import csv


CUTOFF_YEAR = 2021  # all models have knowledge cutoff later than September 2021
REVIEW_TYPES = {
    "neutral": "",
    "positive": "positive ",
    "negative": "negative ",
}
TARGET_NUM_BOOKS = 250
PROMPT_TEMPLATE = (
    "Write a {review_type}review for the book \"{title}\" by {author}. Only output the review text without a title or rating."
)


def main() -> None:
    args = _parse_args()
    input_file = args.input_file.resolve()
    output_file = args.output_file.resolve()

    print(
        f"Writing LLM prompts to {output_file}"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_file, "r", encoding="utf-8-sig") as f:  # NB: has BOM
        reader = csv.DictReader(f)
        all_books = tuple(reader)

    print(f"Loaded {len(all_books)} books in total")

    # Filter books that are too new and sort by ranking
    books = tuple(sorted(
        (book for book in all_books if int(book["Published Date"]) < CUTOFF_YEAR),
        key=lambda book: int(book["Position"]),
    ))[:TARGET_NUM_BOOKS]
    print(f"Have final list of {len(books)} books released before {CUTOFF_YEAR}")

    enc = tiktoken.get_encoding("cl100k_base")
    total_input_tokens = 0

    with open(output_file, "w") as f:
        for book in books:
            title = book["Title"]
            author = book["Authors"]

            for review_type, review_type_str in REVIEW_TYPES.items():
                prompt_data = {
                    "messages": [
                        {
                            "role": "user",
                            "content": PROMPT_TEMPLATE.format(review_type=review_type_str, title=title, author=author),
                        }
                    ],
                    "text_type": "argumentative",
                    "type": "book_reviews",
                    "book_title": title,
                    "book_author": author,
                    "review_type": review_type,
                }

                f.write(json.dumps(prompt_data) + "\n")

                total_input_tokens += len(enc.encode(prompt_data["messages"][0]["content"]))

    print(f"Total number of input tokens ({enc.name}): {total_input_tokens}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file",
        type=pathlib.Path,
        required=True,
        help="Path to CSV containing books rankings",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        required=True,
        help="Path to output jsonl file into which to write prompts for LLMs",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
