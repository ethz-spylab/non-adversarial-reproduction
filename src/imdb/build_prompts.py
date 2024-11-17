import argparse
import json
import copy
import pathlib
import numpy as np
import tiktoken
import tqdm


CUTOFF_YEAR = 2021  # all models have knowledge cutoff later than September 2021
REVIEW_TYPES = {
    "neutral": "",
    "positive": "positive ",
    "negative": "negative ",
}
PROMPT_TEMPLATE = (
    "Write a {review_type}review for the {year} movie \"{title}\". Only output the review text without a title or rating."
)


def main() -> None:
    args = _parse_args()
    input_dir = args.input_dir
    input_movies_file = input_dir / "movies.json"
    input_reviews_dir = input_dir / "reviews"
    output_prompts_file = args.output_prompts_file.resolve()
    results_dir = args.results_dir.resolve()
    ouput_humans_file = results_dir / output_prompts_file.stem / "humans_temp0.0.jsonl"

    print(
        f"Writing LLM prompts to {output_prompts_file} and human reviews to {ouput_humans_file}"
    )
    output_prompts_file.parent.mkdir(parents=True, exist_ok=True)
    ouput_humans_file.parent.mkdir(parents=True, exist_ok=True)

    with open(input_movies_file) as f:
        all_movies = json.load(f)

    print(f"Loaded {len(all_movies)} movies")

    # Filter movies that are too new (keep index to map reviews)
    movies = {idx: movie for idx, movie in enumerate(all_movies) if movie["year"] < CUTOFF_YEAR}
    print(f"Filtered out {len(all_movies) - len(movies)} movies released in {CUTOFF_YEAR} or later")

    # Use ALL human reviews; potential filtering happens during evaluation
    # For LLMs, generate a set of prompts for each movie
    num_human_reviews = 0

    enc = tiktoken.get_encoding("cl100k_base")
    total_input_tokens = 0

    with open(ouput_humans_file, "w") as f_humans, open(output_prompts_file, "w") as f_prompts:
        for idx, movie in (pbar := tqdm.tqdm(movies.items(), desc="Building prompts and human completions")):
            review_file = input_reviews_dir / f"reviews_{idx}.json"
            movie_title = movie["title"]
            movie_year = movie["year"]

            # Update pbar postfix with movie title and number of reviews so far
            pbar.set_postfix(movie=movie_title, num_human_reviews=num_human_reviews)

            # Add human reviews
            with open(review_file) as f_reviews:
                current_reviews = json.load(f_reviews)
            prompt_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPT_TEMPLATE.format(review_type=REVIEW_TYPES["neutral"], year=movie_year, title=movie_title),
                    }
                ],
                "text_type": "argumentative",
                "type": "imdb_reviews",
                "baseline": "imdb_reviews",
                "movie_title": movie_title,
                "movie_year": movie_year,
            }
            for review in current_reviews:
                human_completion_data = copy.deepcopy(prompt_data)
                current_rating = None
                if review["rating_value"] is not None:
                    assert review["rating_scale"] == 10
                    current_rating = review["rating_value"]
                review_text = review["text"].strip()

                human_completion_data.update({
                    "completion": review_text,
                    "review_id": review["id"],
                    "review_date": review["date"],
                    "review_rating": current_rating,
                })
                f_humans.write(json.dumps(human_completion_data) + "\n")

                num_human_reviews += 1

            # Add LLM prompts
            for review_type, review_type_str in REVIEW_TYPES.items():
                llm_prompt = copy.deepcopy(prompt_data)
                llm_prompt["review_type"] = review_type
                llm_prompt["messages"][0]["content"] = PROMPT_TEMPLATE.format(review_type=review_type_str, year=movie_year, title=movie_title)

                f_prompts.write(json.dumps(llm_prompt) + "\n")

                total_input_tokens += len(enc.encode(llm_prompt["messages"][0]["content"]))

    print(f"Generated a total of {num_human_reviews} human reviews")
    print(f"Total number of input tokens ({enc.name}): {total_input_tokens}")


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
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Path to directory containing movies.json and review/ subdirectory",
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
