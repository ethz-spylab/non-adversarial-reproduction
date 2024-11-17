# Measuring Non-Adversarial Reproduction of Training Data in Large Language Models

*[Michael Aerni*](https://www.michaelaerni.com/), [Javier Rando*](https://javirando.com/), [Edoardo Debenedetti](https://edoardo.science/), [Nicholas Carlini](https://nicholas.carlini.com/), [Daphne Ippolito](https://www.daphnei.com/), [Florian Tramèr](https://floriantramer.com/)*

Official repository for the paper [**Measuring Non-Adversarial Reproduction of Training Data in Large Language Models**](https://arxiv.org/abs/2411.10242).
See also our [**blog post**](https://spylab.ai/blog/non-adversarial-reproduction/) and [Twitter/X thread](https://x.com/AerniMichael/status/1858556259706040570).

Our **raw results** required to reproduce our analysis are currently available via [Hugging Face](https://huggingface.co/datasets/nonadvreproduction/data/blob/main/data.zip).


## Setup and general information
### Setup
To run the code, [install Rye](https://rye.astral.sh/guide/installation/)
and run
```bash
rye sync
```
in the root of this repository.

To reproduce our analysis,
also download our [dataset](https://huggingface.co/datasets/nonadvreproduction/data/blob/main/data.zip),
and copy the `perplexities/` and `results/` directories
into the `data/` directory in this repository.

### File formats
Each prompt, LLM generation and human-written text in our study is a JSON object,
where the collection of those objects is stored in jsonl files
(json objects separated by newlines).
In general, we start with just prompts,
and add more information (e.g., LLM generations) in-place.

Each prompt/generation object has a `messages` attribute
containing the prompt (in OpenAI API format),
a `text_type` attribute
(either `creative`, `expository`, or `argumentative`)
and `type` attribute that specifies the subtype.
All mitigation strategies have an additional `ablation` attribute that indicates
the type of mitigation (`assistant` or `simple`).
LLM completions or human-written text is stored in a `completion` attribute
(which can be missing).
Certain types of prompts may contain additional attributes.
For all LLM generations, we also store the seed in an a `seed` attribute
for reproducibility.
Finally, after mapping the LLM generations to AuxDataset,
`memorized_chars` is an array with the same length as completion;
for each character, this array contains the lenght of the longest substring
overlapping that character and found in AuxDataset.


## Building prompts and baselines

### General prompts
Most prompts are generated from templates and instantiations
in `data/prompts/raw_data/general_prompts.json`.
To build the prompts, run
```
python src/non_adversarial_reproduction/build_general_prompts.py --input-file data/prompts/raw_data/general_prompts.json --output-file data/prompts/general_prompts.jsonl
```

### Book reviews
We use book titles from https://thegreatestbooks.org/.
To build prompts,
download the Fall 2024 version (rc38) from https://thegreatestbooks.org/rc/38.csv,
and run the following (replace `PATH_TO_RAW_DATA` with the path to the downloaded file):
```bash
python src/non_adversarial_reproduction/build_book_reviews_prompts.py --input-file PATH_TO_RAW_DATA --output-file data/prompts/book_reviews.jsonl
```

### Essays (PERSUADE 2.0)
We use the [PERSUADE 2.0 corpus](https://www.sciencedirect.com/science/article/pii/S1075293524000588#fn1)
of essay prompts and responses.
We only use prompts in our paper and omit the human responses due to a high degree
of overlap with AuxDataset.
Nevertheless, the following builds prompts and a human baseline.
Since the set of usable prompts in the corpus is relatively small,
we manually invented additional prompts,
which can be found in `data/prompts/raw_data/essay_persuade_additional_prompts.jsonl`.

To build the prompts,
download the [PERSUADE 2.0 corpus](https://github.com/scrosseye/persuade_corpus_2.0),
the [PERSUADE 1.0 corpus](https://www.kaggle.com/c/feedback-prize-2021/data)
(used for filtering human responses),
and store them in some `PERSUADE_DIR` directory to have files
`PERSUADE_DIR/persuade_corpus_2.0.csv`,
`PERSUADE_DIR/feedback-prize-2021/train.csv`,
and `PERSUADE_DIR/feedback-prize-2021/test/`.
Then, run
```bash
python src/essays/build_persuade_prompts.py --raw-data-dir PERSUADE_DIR --output-prompts-file data/prompts/essay_persuade.jsonl --results-dir data/results/ --control-prompts-file data/prompts/raw_data/essay_persuade_additional_prompts.jsonl --id-blacklist-file data/baselines/essay_persuade/blacklist_ids.txt
```

### Reddit data (explainlikeimfive and WritingPrompts)
First, download Reddit submissions and comments for May to July 2014
(`2024-05`, `2024-06`, and `2024-07`)
from AcademicTorrents,
and store them in some `REDDIT_RAW_DATA_DIR` directory
(`REDDIT_RAW_DATA_DIR` should contain `reddit/comments/RC-*.zst` and `reddit/submissions/RS-*.zst` files).

Then, extract all submissions and comments for the two baselines
with the following two commands (this may take a few hours);

```bash
# explainlikeimfive
python src/explainlikeimfive/extract_submissions_comments.py --input-dir REDDIT_RAW_DATA_DIR/reddit --output-submissions data/baselines/explainlikeimfive/submissions.jsonl --output-comments data/baselines/explainlikeimfive/comments.jsonl --months 2024-05 2024-06 2024-07

# WritingPrompts
python src/writing_prompts/extract_submissions_comments.py --input-dir REDDIT_RAW_DATA_DIR/reddit --output-submissions data/baselines/writing_prompts/submissions.jsonl --output-comments data/baselines/writing_prompts/comments.jsonl --months 2024-05 2024-06 2024-07
```

Finally, filter and sample submissions and comments to serve as prompts and human baselines:

After running those commands,
`data/prompts/explainlikeimfive.jsonl`
and
`data/prompts/writing_prompts.jsonl`
will contain prompts in the format to be used with `generate.py`.
Furthermore, `data/results/explainlikeimfive/`
and `data/results/writing_prompts/`
will each contain a file `humans_temp0.0.jsonl` containing human responses
(as produced by `generate.py`)
ready to be mapped against AuxDataset.

All prompt objects will have a `reddit_submission_id` field
to identify the prompt's submission;
human results additionally have `reddit_comment_id` and `reddit_author` fields
to identify the Reddit comment.

### IMDb reviews
For IMDb reviews,
we first collect a top movies list from IMDb,
then collect the reviews for each movie,
and finally build prompts and human baselines.

We use the top 250 movies list found at https://www.imdb.com/chart/top/,
and store the results in `data/baselines/imdb/movies.json`.
This file contains a JSON array of movie objects,
each with a `title`, `year`, `duration` (in minutes), `rating` (string) and `url` (to the movie's page) attribute.
For example, `movies.json` might look like
```json
[
    {"title": "The Shawshank Redemption", "year": 1994, "duration": 142, "rating": "R", "url": "https://www.imdb.com/title/tt0111161/"},
    ...
]
```

We then collect each movie's reviews from IMDb.
For the movie in `movies.json` with index `i` (0-based),
we store the reviews in ``data/baselines/imdb/reviews/reviews_i.json``.
This file contains a JSON array of review objects,
each
For example, a reviews file might look like
```json
[
    {"id": "rw9955642", "title": "A film that is truly memorable, [...].", "date": "2024-08-15", "text": "Its a film where you just feel pure raw emotion as if you're the character [...].", "rating_value": 10, "rating_scale": 10},
    ...
]
```

Having collected all the movies and reviews,
the following script builds prompts and human baselines:
```bash
python src/imdb/build_prompts.py --input-dir data/baselines/imdb/ --output-prompts-file data/prompts/imdb_reviews.jsonl --results-dir data/results/
```
This also creates a human baseline file in `data/results/imdb_reviews/humans_temp0.0.jsonl`.
Note that this baseline file contains all reviews,
including ones that are too old to be considered.
We account for this (i.e., filter) in the analysis.

### In-the-wild baselines (lmsys and WildChat)
We provide scripts to automatically download the source datasets
and extract prompts and completions.
The results will be in the same format as returned by `generate.py`.
Make sure to once run `huggingface-cli login` as the dataset requires a one-time agreement.
Then run
```bash
python src/memorization_chat_datasets/lmsys_chat.py --results-dir data/results/
python src/memorization_chat_datasets/wildchat.py --results-dir data/results/
```

### Ablation prompts
We generate ablation prompts from all original prompts automatically.
**Hence, make sure to generate all the necessary original prompt files first!**
The script generates one large input file per ablation,
combining all the original prompts:
```bash
python src/non_adversarial_reproduction/build_prompt_ablations.py --output-dir data/prompts/ --prompt-files data/prompts/general_prompts.jsonl data/prompts/essay_persuade.jsonl
```


## Running experiments
### Inference (LLM generations)
To simply reproduce a set of experiments in our paper,
run one of the `generate_*.sh` scripts in the root of this repository.
Make sure to inspect the script first and modify things where necessary,
and be aware that running those scripts can incur significant costs.

All LLM generations are created with the `claude_memorization/generate.py` script.
We use OpenRouter by default, but support using the OpenAI or Google Gemini APIs directly.

To use OpenRouter (default),
put the API key into the `OPENROUTER_API_KEY` environment variable and run `generate.py`.
For example, to get generations for the general prompts with Claude 3 Haiku and temperature 0.7 on five seeds, run
```bash
python src/non_adversarial_reproduction/generate.py \
    --model "anthropic/claude-3-haiku" \
    --data_path "data/prompts/general_prompts.jsonl" \
    --temperature "0.7" \
    --max_new_tokens "8000" \
    --max-parallel-requests "8" \
    --output_path "data/results/" \
    --seeds $(seq 0xC0FFEE $((0xC0FFEE + 5 - 1)))
```
The script will determine the output directory and file name from the prompt file name.
In the example above, the results will be stored in
`data/results/general_prompts/claude-3-haiku_temp0.7.jsonl`.

Run `python src/non_adversarial_reproduction/generate.py --help` for more details.

To force using the OpenAI API,
add the `--use-openai` flag and set the `OPENAI_API_KEY` and `OPENAI_ORG` environment variables.
Make sure to still include the `openai/` prefix in the `--model` argument
even if the OpenAI API does not require it.

To force using the Google Gemini API,
add the `--use-google` flag and set the `GOOGLE_API_KEY` environment variable.
Make sure to still include the `google/` prefix in the `--model` argument
even if the OpenAI API does not require it.


### Matching snippets with AuxDataset
For a description of AuxDataset, see [Nasr et al. (2023)](https://arxiv.org/abs/2311.17035).
We do provide the raw matching data from AuxDataset for all results in our paper
(except tokens for copyrighted human-written texts).

For every results directory (e.g., `data/results/general_prompts/`),
the raw AuxDataset data must be stored in an `index/` subdirectory
(e.g., `data/results/general_prompts/index/`).
For every `*.jsonl` file in the results directory,
there must be a corresponding `*.tokens.npy` and `*.kgrams.npy` file in the `index/` subdirectory.
The `*.kgrams.npy` file contains, for every completion and token in the completion,
the length of the longest k-gram in AuxDataset that starts at that token.
Tokenization is with respect to `data/my.bpe`.
The `*.tokens.npy` file just contains the tokenized completion,
which is used to map overlaps in token space to character space.

To process matches in AuxDataset for our analysis,
first calculate overlaps between the completion and prompt (for discounting) via
```bash
PROMPT_FILE_NAMES=("ablation_assistant" "ablation_simple" "book_reviews" "essay_persuade" "explainlikeimfive" "general_prompts" "imdb_reviews" "writing_prompts" "lmsys_chat" "wildchat")
for prompt_file_name in "${PROMPT_FILE_NAMES[@]}"; do
    python src/non_adversarial_reproduction/calculate_prompt_overlaps.py --results-folder "data/results/${prompt_file_name}" ;
done
```
and then run the mapping script
```bash
PROMPT_FILE_NAMES=("ablation_assistant" "ablation_simple" "book_reviews" "essay_persuade" "explainlikeimfive" "general_prompts" "imdb_reviews" "writing_prompts" "lmsys_chat" "wildchat")
for prompt_file_name in "${PROMPT_FILE_NAMES[@]}"; do
    python src/non_adversarial_reproduction/map_index_results.py --tokenizer-file data/my.bpe --results-folder "data/results/${prompt_file_name}" ;
done
```


### Calculate the perplexity of 50-character snippets
Calculating the perplexities requires two script invocations,
once for all LLMs, and once for the human counterparts.
Each script invocation yields two json files containing
the sampled memorized and non-memorized snippets,
and NumPy files containing the corresponding perplexities.

You can optionally specify a PyTorch device via `--device DEVICE`.
If no GPUs are available, the device will always be CPU.

**For models**:
```bash
models=(
    "gpt-4o-mini-2024-07-18_temp0.7"
    "gpt-4o-2024-05-13_temp0.7"
    "gpt-4-turbo-2024-04-09_temp0.7"
    "claude-3-haiku_temp0.7"
    "claude-3.5-sonnet_temp0.7"
    "claude-3-opus_temp0.7"
    "llama-3.1-8b-instruct_temp0.7"
    "llama-3.1-70b-instruct_temp0.7"
    "llama-3.1-405b-instruct_temp0.7"
    "gemini-1.5-flash-002_temp0.7"
    "gemini-1.5-pro-002_temp0.7"
)
python src/non_adversarial_reproduction/calculate_perplexities.py --input-base-dir "data/results/" --output-dir "data/perplexities/models/" \
    --models ${models[@]} \
    --settings "book_reviews" "essay_persuade" "explainlikeimfive"  "general_prompts" "imdb_reviews" "writing_prompts" \
    --device "cuda:0"
```

**For humans**:
```bash
python src/non_adversarial_reproduction/calculate_perplexities.py --input-base-dir "data/results/" --output-dir "data/perplexities/humans/" \
    --models "humans_temp0.0" \
    --settings "explainlikeimfive" "writing_prompts" "imdb_reviews" \
    --device "cuda:0"
```


## Analyzing results
All our plots can be reproduced by simply running the `plots.ipynb` notebook.
This notebook assumes that all results are stored in the `data/results/` directory
and are mapped against AuxDataset.
