import argparse
import datetime
import json
import pathlib
import typing
import torch
from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast
import tqdm
import re
import numpy as np


MODEL_NAME = "EleutherAI/pythia-70m-deduped"
SNIPPET_LENGTH = 50
RE_WORD_START = re.compile(r"\b\w|^\w")
IMDB_REVIEW_DATE_MIN = datetime.date(2024, 5, 1)


def main(args) -> None:

    rng = np.random.default_rng(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Storing results in", output_dir)

    # Extract memorized and non-memorized snippets
    # Snippets are:
    # - 50 characters long
    # - Start at the beginning of a word
    # - Either fully contained in a reproduced substring or not overlapping with it
    # - For every completion, select one reproduced and one non-reproduced snippet (if possible)
    # - If there are multiple candidates, select one at random

    # Extract snippets from raw data
    # Make sure everything is in a deterministic order for reproducibility
    input_base_dir = args.input_base_dir
    memorized_snippets = []
    non_memorized_snippets = []
    for setting in tqdm.tqdm(
        sorted(args.settings), desc="Extracting snippets", unit="setting"
    ):
        rng_setting, rng = rng.spawn(2)
        setting_dir = input_base_dir / setting
        for model_temp in sorted(args.models):
            rng_file, rng_setting = rng_setting.spawn(2)
            input_file = setting_dir / f"{model_temp}.jsonl"
            with open(input_file, "r") as f:
                for row in f.readlines():
                    raw_data = json.loads(row)
                    if "error" in raw_data:
                        continue

                    # Special case: skip IMDB reviews that are too old
                    if (
                        setting == "imdb_reviews"
                        and model_temp == "humans_temp0.0"
                        and datetime.date.fromisoformat(raw_data["review_date"])
                        < IMDB_REVIEW_DATE_MIN
                    ):
                        continue

                    completion = raw_data["completion"]
                    memorized_chars = np.array(raw_data["memorized_chars"])
                    current_memorized, current_non_memorized = extract_snippets(
                        completion, memorized_chars, rng_file
                    )
                    if current_memorized is not None:
                        memorized_snippets.append(current_memorized)
                    if current_non_memorized is not None:
                        non_memorized_snippets.append(current_non_memorized)
            del rng_file
        del rng_setting

    # Save snippets for reproducibility
    with open(output_dir / "memorized_snippets.jsonl", "w") as f:
        for snippet in memorized_snippets:
            f.write(json.dumps(snippet) + "\n")
    print(
        "Saved",
        len(memorized_snippets),
        "reproduced snippets to",
        output_dir / "memorized_snippets.jsonl",
    )
    with open(output_dir / "non_memorized_snippets.jsonl", "w") as f:
        for snippet in non_memorized_snippets:
            f.write(json.dumps(snippet) + "\n")
    print(
        "Saved",
        len(non_memorized_snippets),
        "non-reproduced snippets to",
        output_dir / "non_memorized_snippets.jsonl",
    )

    # Calculate perplexity for both memorized and non-memorized snippets

    # Load the model and tokenizer
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME)
    model.eval()

    # Ensure we use GPU if available
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Calculate perplexities
    perplexities_reproduced = calculate_perplexity(
        memorized_snippets, model, tokenizer, args.batch_size, device
    )
    perplexities_non_reproduced = calculate_perplexity(
        non_memorized_snippets, model, tokenizer, args.batch_size, device
    )

    # Save results (numpy)
    np.save(output_dir / "perplexities_reproduced.npy", perplexities_reproduced)
    np.save(output_dir / "perplexities_non_reproduced.npy", perplexities_non_reproduced)

    print("Average perplexity for reproduced: ", np.mean(perplexities_reproduced))
    print(
        "Average perplexity for non-reproduced: ", np.mean(perplexities_non_reproduced)
    )


def extract_snippets(
    completion: str, memorized_chars: np.ndarray, rng: np.random.Generator
) -> typing.Tuple[str, str]:
    # Inefficient implementation that is quadratic in the length of the completion.
    # However, since completions are relatively short, this should be fine.
    # This could be optimized but at the cost of readability.

    memorized_mask = memorized_chars >= SNIPPET_LENGTH

    # For every word start, check if it is completely contained in a memorized or non-memorized segment
    candidates_memorized = []
    candidates_non_memorized = []
    for word_start_match in RE_WORD_START.finditer(completion):
        start_idx = word_start_match.start()
        if start_idx + SNIPPET_LENGTH > len(completion):
            continue

        # Check if the snippet is fully contained in a memorized or non-memorized segment
        if np.all(
            memorized_mask[start_idx]
            == memorized_mask[start_idx : start_idx + SNIPPET_LENGTH]
        ):
            if memorized_mask[start_idx]:
                candidates_memorized.append(start_idx)
            else:
                candidates_non_memorized.append(start_idx)

    # for start_idx in candidates_memorized:
    #     assert start_idx + SNIPPET_LENGTH <= len(completion) and np.all(memorized_mask[start_idx : start_idx + SNIPPET_LENGTH])
    # for start_idx in candidates_non_memorized:
    #     assert start_idx + SNIPPET_LENGTH <= len(completion) and np.all(~memorized_mask[start_idx : start_idx + SNIPPET_LENGTH])

    # Select random snippet from candidates
    if len(candidates_memorized) == 0:
        memorized_snippet = None
    else:
        start_idx = rng.choice(candidates_memorized)
        memorized_snippet = completion[start_idx : start_idx + SNIPPET_LENGTH]
        assert len(memorized_snippet) == SNIPPET_LENGTH
    if len(candidates_non_memorized) == 0:
        non_memorized_snippet = None
    else:
        start_idx = rng.choice(candidates_non_memorized)
        non_memorized_snippet = completion[start_idx : start_idx + SNIPPET_LENGTH]
        assert len(non_memorized_snippet) == SNIPPET_LENGTH

    return memorized_snippet, non_memorized_snippet


@torch.no_grad()
def calculate_perplexity(
    snippets: typing.Collection[str],
    model: GPTNeoXForCausalLM,
    tokenizer: GPTNeoXTokenizerFast,
    batch_size: int,
    device: torch.device,
):
    perplexities = []
    for batch_idx in tqdm.trange(
        0, len(snippets), batch_size, desc="Calculating perplexity", unit="batch"
    ):
        batch_texts = [
            "Copy this text: " + snippet
            for snippet in snippets[batch_idx : batch_idx + batch_size]
        ]

        # Tokenize batch
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        prefix_offset = 4  # length of constant prefix

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # Get the logits and calculate per-token loss (cross-entropy)
        logits = outputs.logits
        shift_logits = logits[..., prefix_offset - 1 : -1, :].contiguous()
        shift_labels = input_ids[..., prefix_offset:].contiguous()
        shift_attention_mask = attention_mask[..., prefix_offset:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        per_token_loss = per_token_loss.view(
            shift_labels.size()
        )  # Reshape to batch_size x seq_len

        # Mask padding tokens in loss calculation
        per_token_loss = per_token_loss * shift_attention_mask

        # Sum the losses over each token in each sample, then divide by number of tokens (to get mean loss per string)
        per_sample_loss = per_token_loss.sum(dim=1) / shift_attention_mask.sum(dim=1)

        # Calculate perplexity for each string in the batch
        batch_perplexities = torch.exp(per_sample_loss).cpu().tolist()
        perplexities.extend(batch_perplexities)

    return perplexities


def _parse_args() -> argparse.Namespace:
    # Parse a python argument for modelname
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-base-dir",
        help="Path to base input dir",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--settings",
        help="Setting names to consider",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--models",
        help="Models to consider",
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--output-dir",
        help="Path to output directory",
        type=pathlib.Path,
    )
    parser.add_argument(
        "--device",
        help="Torch device to use",
        type=str,
        default="cuda:7",
    )
    parser.add_argument(
        "--batch-size",
        help="Batch size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--seed",
        help="Random seed",
        type=int,
        default=0xCAFED00D,
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
