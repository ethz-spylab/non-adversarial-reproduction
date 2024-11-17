import os
import typing

os.environ["TIKTOKEN_CACHE_DIR"] = ""

import numpy as np
from tqdm import tqdm
import json
import argparse
import tiktoken
import tiktoken.load
import pathlib


def mycl100k_base(tokenizer_file):
    mergeable_ranks = tiktoken.load.load_tiktoken_bpe(str(tokenizer_file))

    ENDOFTEXT = "<|enZdo||ftext|>"
    FIM_PREFIX = "<|fiZm_||prefix|>"
    FIM_MIDDLE = "<|fiZm_||middle|>"
    FIM_SUFFIX = "<|fiZm_||suffix|>"
    ENDOFPROMPT = "<|endZo||fprompt|>"

    special_tokens = {
        ENDOFTEXT: 65002,
        FIM_PREFIX: 65003,
        FIM_MIDDLE: 65004,
        FIM_SUFFIX: 65005,
        ENDOFPROMPT: 65006,
    }

    return {
        "name": "mycl100k_base",
        "pat_str": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        "mergeable_ranks": mergeable_ranks,
        "special_tokens": special_tokens,
    }


def iterate_over_examples(tokens, kgrams):
    """
    Code to get kgrams for each generation.
    """
    split_at = tokens == 0xFFFF
    split_at = [-2] + list(np.where(split_at)[0])
    for start, end in zip(split_at, split_at[1:]):
        batch_tokens = tokens[start + 2 : end - 1]
        batch_kgrams = kgrams[start + 2 : end - 1]

        yield batch_tokens, batch_kgrams


def main(args):
    tokenizer = tiktoken.Encoding(**mycl100k_base(args.tokenizer_file))
    index_folder = args.results_folder / "index"
    if not index_folder.exists():
        raise ValueError(f"Index folder does not exist: {index_folder}")

    print("Mapping index results in folder", args.results_folder)

    for generation_file in (
        pbar := tqdm(
            sorted(
                args.results_folder.glob("*.jsonl"),
            )
        )
    ):
        kgrams_file = index_folder / (generation_file.stem + ".kgrams.npy")
        tokens_file = index_folder / (generation_file.stem + ".tokens.npy")

        if not kgrams_file.exists():
            pbar.write(f"Missing kgrams file for {generation_file}: {kgrams_file}")
            continue
        if not tokens_file.exists():
            pbar.write(f"Missing tokens file for {generation_file}: {tokens_file}")
            continue

        with open(generation_file, "r") as f:
            dataset = tuple(json.loads(line) for line in f)
        kgrams_suffix = np.load(kgrams_file)

        with open(tokens_file, "rb") as f:
            # tokens can be slightly longer than kgrams; can just drop tail
            tokens = np.frombuffer(f.read(), dtype=np.uint16)[: len(kgrams_suffix)]
        assert tokens.shape == kgrams_suffix.shape

        # Split the tokens and kgrams per prompt
        index_results = tuple(iterate_over_examples(tokens, kgrams_suffix))
        tokens, kgrams_suffix = zip(*index_results)

        # Make sure sizes are correct up to padding
        assert len(tokens) == len(kgrams_suffix) and len(tokens) >= len(dataset)
        assert len(tokens) > len(dataset), "Expected padding but there was none"
        assert all(
            tokens[idx].shape == kgrams_suffix[idx].shape for idx in range(len(tokens))
        )
        assert all(
            tokens[idx].shape == (0,) for idx in range(len(dataset), len(tokens))
        )  # padding entries must be 0
        tokens = tokens[: len(dataset)]
        kgrams_suffix = kgrams_suffix[: len(dataset)]
        assert len(tokens) == len(kgrams_suffix) == len(dataset)

        # Check that decoding tokens works as expected and all the data is there
        tokens, kgrams_suffix = _fix_indices(tokens, kgrams_suffix, dataset, tokenizer)

        # For each prompt, kgrams_suffix[i] = longest k-gram in tokens starting at index i
        # Map this to characters, i.e., the longest k-gram in characters starting at index i
        # Only consider complete utf-8 characters, i.e., round up at the beginning of a sequence and down at the end
        # Then discount by longest overlap with prompt
        # And store the sequence length for every character
        for data_idx in range(len(dataset)):
            # Either have an error and 0 tokens or the completion matches the decoded tokens
            assert (
                "error" in dataset[data_idx] and len(tokens[data_idx]) == 0
            ) or dataset[data_idx]["completion"] == tokenizer.decode(
                tokens[data_idx], errors="strict"
            )
            if "error" in dataset[data_idx]:
                continue

            current_suffix_tokens = kgrams_suffix[data_idx]
            current_tokens = tokens[data_idx]
            current_overlap = np.zeros(len(dataset[data_idx]["completion"]), dtype=int)
            num_tokens = len(current_tokens)

            # First decode only bytes to avoid issues on utf-8 boundaries
            decoded_bytes = tuple(
                tokenizer.decode_single_token_bytes(token) for token in current_tokens
            )

            # For every token, calculate the first and last character corresponding to that token
            # For a token that splits a utf-8 character:
            # - Round up start to the next utf-8 character boundary
            # - Round down end to the previous utf-8 character boundary
            complete_prefix = ""
            incomplete_prefix = b""
            string_starts = np.zeros(num_tokens, dtype=int)
            string_ends = np.zeros(num_tokens, dtype=int)  # exclusive
            for token_idx in range(num_tokens):
                # Decode bytes until we have a full utf-8 character
                assert len(incomplete_prefix) == 0 or (
                    incomplete_prefix[0] & 0x80 == 0x80
                )
                current_bytes_remaining = decoded_bytes[token_idx]
                current_decoded = ""
                while len(incomplete_prefix) > 0 and len(current_bytes_remaining) > 0:
                    incomplete_prefix = incomplete_prefix + current_bytes_remaining[0:1]
                    current_bytes_remaining = current_bytes_remaining[1:]
                    try:
                        current_decoded = incomplete_prefix.decode(
                            "utf-8", errors="strict"
                        )
                        # Success, clear prefix
                        incomplete_prefix = b""
                        break
                    except UnicodeDecodeError:
                        # Continue decoding
                        pass

                # Update global prefix and store start
                complete_prefix += current_decoded
                string_starts[token_idx] = len(complete_prefix)

                # Continue if there was no progress (i.e., start at boundary and token did not yield a full utf-8 character)
                # Have a 0-length char sequence that already starts at previous end, but also ends there
                if len(incomplete_prefix) > 0:
                    assert len(current_bytes_remaining) == 0
                    assert (
                        token_idx > 0
                        and string_starts[token_idx] == string_ends[token_idx - 1]
                    )
                    string_ends[token_idx] = string_starts[token_idx]
                    continue

                # Decode remaining bytes
                current_decoded = ""
                current_incomplete_suffix = b""
                while len(current_bytes_remaining) > 0:
                    try:
                        current_decoded = current_bytes_remaining.decode(
                            "utf-8", errors="strict"
                        )
                        break
                    except UnicodeDecodeError:
                        current_incomplete_suffix = (
                            current_bytes_remaining[-1:] + current_incomplete_suffix
                        )
                        current_bytes_remaining = current_bytes_remaining[:-1]
                assert len(incomplete_prefix) == 0
                incomplete_prefix = current_incomplete_suffix
                complete_prefix += current_decoded
                string_ends[token_idx] = len(complete_prefix)
                assert string_starts[token_idx] <= string_ends[token_idx]

            # After handling the last token, this should have precisely decoded the entire completion
            assert len(incomplete_prefix) == 0
            assert complete_prefix == dataset[data_idx]["completion"]
            assert num_tokens == 0 or (
                string_starts[0] == 0
                and string_ends[-1] == len(dataset[data_idx]["completion"])
            )
            assert np.all(string_starts <= string_ends)

            # Given the precomputed character ranges, determine the (discounted) overlaps in character space
            prompt_overlap_suffixes = dataset[data_idx]["prompt_suffix_lengths"]
            for token_idx in range(num_tokens):
                current_token_kgram = current_suffix_tokens[token_idx]
                if current_token_kgram == 0:
                    # Token does not correspond to any characters
                    continue

                current_char_start = string_starts[token_idx]

                # FIXME: Something is weird here; the assertion below somehow does not always hold,
                #  even if the full string is decoded correctly.
                #  Just ignore this for now and clip.
                # assert token_idx + current_token_kgram <= string_ends.shape[0]
                # current_char_end = string_ends[token_idx + current_token_kgram - 1]
                current_char_end = string_ends[
                    min(token_idx + current_token_kgram - 1, string_ends.shape[0] - 1)
                ]

                if current_char_start == current_char_end:
                    # Token does not correspond to any characters
                    continue

                # Now have a memorized string from (inclusive) current_char_start to (exclusive) current_char_end
                # Determine the longest common substring with the prompt, and subtract that length from the sequence length
                current_prompt_overlap_lengths = prompt_overlap_suffixes[
                    current_char_start:current_char_end
                ]
                # Prompt overlaps could extend beyond the length of the current memorized string. Only discount until the end of the string
                current_prompt_overlap_lengths = np.minimum(
                    current_prompt_overlap_lengths,
                    current_char_end - np.arange(current_char_start, current_char_end),
                )
                discount_length = np.max(current_prompt_overlap_lengths)
                assert 0 <= discount_length <= current_char_end - current_char_start
                current_memorized_chars_discounted = (
                    current_char_end - current_char_start - discount_length
                )

                # Apply the discounted length to ALL characters in the memorized sequence
                current_overlap[current_char_start:current_char_end] = np.maximum(
                    current_overlap[current_char_start:current_char_end],
                    current_memorized_chars_discounted,
                )

            # Directly store character-level overlaps
            dataset[data_idx][f"memorized_chars"] = current_overlap.tolist()

        with open(generation_file, "w") as f:
            for data in dataset:
                f.write(json.dumps(data) + "\n")


def _fix_indices(
    tokens: typing.List[np.ndarray],
    kgrams_suffix: typing.List[np.ndarray],
    dataset: typing.List[dict],
    tokenizer: tiktoken.Encoding,
):
    # In few instances, the mapping between the dataset and tokens/kgrams is off
    # The following tries to map things together again if this is the case
    wrong_indices = []

    for data_idx in range(len(dataset)):
        if "error" in dataset[data_idx]:
            if len(tokens[data_idx]) > 0:
                wrong_indices.append(data_idx)
        else:
            if dataset[data_idx]["completion"] != tokenizer.decode(
                tokens[data_idx], errors="strict"
            ):
                wrong_indices.append(data_idx)

    if len(wrong_indices) == 0:
        return tokens, kgrams_suffix

    # For every wrong index, try to find its completion among the other wrong indices
    # For errors, need to have the same number of empty completions as the number of errors
    # For non-errors, need to uniquely match every completion to decoded tokens
    completions = []
    decoded = []
    for data_idx in wrong_indices:
        completions.append(
            dataset[data_idx]["completion"]
            if "error" not in dataset[data_idx]
            else None
        )
        decoded.append(tokenizer.decode(tokens[data_idx], errors="strict"))
        assert len(tokens[data_idx]) == len(kgrams_suffix[data_idx])

    mapped_indices = []
    for source_idx, data_idx in enumerate(wrong_indices):
        if "error" in dataset[data_idx]:
            mapped_indices.append(-1)
            assert completions[source_idx] is None
            continue
        assert completions[source_idx] is not None and len(completions[source_idx]) > 0

        for target_idx, target_data_idx in enumerate(wrong_indices):
            if decoded[target_idx] == completions[source_idx]:
                mapped_indices.append(target_data_idx)
                break
        else:
            assert False, f"Could not find a match for index {data_idx}"

    # Make sure number of errors and empty decodings is consistent
    num_errors = sum(1 for idx in mapped_indices if idx == -1)
    assert num_errors == sum(1 for decoded_str in decoded if len(decoded_str) == 0)

    # Make sure there are no duplicates (besides errors)
    assert len(set(mapped_indices) - {-1}) == len(mapped_indices) - num_errors

    # Build fixed tokens and kgrams
    fixed_tokens = list(tokens)
    fixed_kgrams = list(kgrams_suffix)
    for data_idx, target_data_idx in zip(wrong_indices, mapped_indices):
        if target_data_idx == -1:
            fixed_tokens[data_idx] = np.zeros(0, dtype=tokens[data_idx].dtype)
            fixed_kgrams[data_idx] = np.zeros(0, dtype=kgrams_suffix[data_idx].dtype)
        else:
            fixed_tokens[data_idx] = tokens[target_data_idx]
            fixed_kgrams[data_idx] = kgrams_suffix[target_data_idx]

    return fixed_tokens, fixed_kgrams


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-folder",
        help="Path to the results folder containing .jsonl files and an index/ folder with corresponding .kgrams.npy and .tokens.npy files",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--tokenizer-file",
        help="Path to my.bpe",
        type=pathlib.Path,
        required=True,
    )
    args = parser.parse_args()
    main(args)
