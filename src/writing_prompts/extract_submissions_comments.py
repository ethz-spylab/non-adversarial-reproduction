import argparse
import json
import multiprocessing
import os
import pathlib
import typing
import warnings
import tqdm
import zstandard
import writing_prompts.models


SUBREDDIT_NAME = "WritingPrompts"


_REFRESH_EVERY = 10_000


def main() -> None:
    args = _parse_args()
    input_root = args.input_dir
    output_submissions_path = args.output_submissions
    output_comments_path = args.output_comments
    months = args.months
    print(
        f"Writing submissions to {output_submissions_path} and comments to {output_comments_path}"
    )
    output_submissions_path.parent.mkdir(parents=True, exist_ok=True)
    output_comments_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        open(output_submissions_path, "w") as f_submissions,
        open(output_comments_path, "w") as f_comments,
    ):
        for month in (
            pbar := tqdm.tqdm(months, desc="Processing months", unit="month")
        ):
            pbar.set_postfix({"month": month})

            submissions_file = input_root / "submissions" / f"RS_{month}.zst"
            comments_file = input_root / "comments" / f"RC_{month}.zst"

            num_lines, num_bad, num_collected = process_file(
                submissions_file, f_submissions, is_comments=False
            )
            pbar.write(
                f"Processed {num_lines:,} submission; {num_bad:,} bad lines and {num_collected:,} collected"
            )
            num_lines, num_bad, num_collected = process_file(
                comments_file, f_comments, is_comments=True
            )
            pbar.write(
                f"Processed {num_lines:,} comments; {num_bad:,} bad lines and {num_collected:,} collected"
            )


def process_file(
    input_path: pathlib.Path, f_out: typing.TextIO, is_comments: bool
) -> typing.Tuple[int, int, int]:
    file_size = os.stat(input_path).st_size
    file_lines = 0
    file_bytes_processed = 0
    bad_lines = 0
    collected_lines = 0
    for line, file_bytes_processed in (
        pbar := tqdm.tqdm(
            read_lines_zst(input_path),
            miniters=_REFRESH_EVERY,
            leave=False,
            desc=f"Processing {'comments' if is_comments else 'submissions'}",
            unit="line",
        )
    ):
        try:
            obj = json.loads(line)
        except (KeyError, json.JSONDecodeError):
            bad_lines += 1
        file_lines += 1

        # NB: no subreddit for ads
        if (
            obj["author"].lower() != "[deleted]"
            and "subreddit" in obj
            and obj["subreddit"].lower() == SUBREDDIT_NAME.lower()
        ):
            # Store all comments and only self-posts (no prompts with external links)
            current_record = None
            if "_meta" in obj:
                was_removed = "removal_type" in obj["_meta"]
            else:
                was_removed = False
            if not is_comments:
                if obj["is_self"]:
                    assert "selftext" in obj
                    current_record = writing_prompts.models.PostRecord(
                        id=obj["id"],
                        author=obj["author"],
                        was_removed=was_removed,
                        created_utc=obj["created_utc"],
                        title=obj["title"],
                        text=obj["selftext"].strip(),
                    )
            else:
                current_record = writing_prompts.models.CommentRecord(
                    id=obj["id"],
                    author=obj["author"],
                    was_removed=was_removed,
                    distinguished_as=obj["distinguished"],
                    created_utc=obj["created_utc"],
                    text=obj["body"].strip(),
                    parent_id=obj["parent_id"],
                    link_id=obj["link_id"],
                )

            if current_record is not None:
                collected_lines += 1
                f_out.write(current_record.model_dump_json(indent=None) + "\n")

        if file_lines % _REFRESH_EVERY == 0:
            pbar.set_postfix(
                {
                    "lines": f"{file_lines:,}",
                    "bad_lines": f"{bad_lines:,}",
                    "collected_lines": f"{collected_lines:,}",
                    "bytes_percent": f"{(file_bytes_processed / file_size) * 100:.1f}%",
                }
            )

    return file_lines, bad_lines, collected_lines


def read_lines_zst(path: pathlib.Path):
    with open(path, "rb") as f:
        buffer = ""
        reader = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(f)
        while True:
            chunk = _read_and_decode(reader, 2**25, 2**30)

            if not chunk:
                break
            lines = (buffer + chunk).split("\n")

            for line in lines[:-1]:
                yield line, f.tell()

            buffer = lines[-1]

        if buffer.strip():
            yield buffer.strip("\n"), f.tell()

        reader.close()


def _read_and_decode(
    reader, chunk_size, max_window_size, previous_chunk=None, bytes_read=0
):
    chunk = reader.read(chunk_size)
    bytes_read += chunk_size
    if previous_chunk is not None:
        chunk = previous_chunk + chunk
    try:
        return chunk.decode()
    except UnicodeDecodeError:
        if bytes_read > max_window_size:
            raise UnicodeError(
                f"Unable to decode frame after reading {bytes_read:,} bytes"
            )
        return _read_and_decode(reader, chunk_size, max_window_size, chunk, bytes_read)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=pathlib.Path,
        required=True,
        help="Path to root directory containing 'comments/' and 'submissions/'",
    )
    parser.add_argument(
        "--output-submissions",
        type=pathlib.Path,
        required=True,
        help="Path to output jsonl file for submissions",
    )
    parser.add_argument(
        "--output-comments",
        type=pathlib.Path,
        required=True,
        help="Path to output jsonl file for comments",
    )
    parser.add_argument(
        "--months",
        type=str,
        nargs="+",
        help="List of dates to process in YYYY-MM format",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
