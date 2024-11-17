import abc
import asyncio
import json
import argparse
import os
import typing

import httpx
import openai
import openai.types.chat
import requests
from tqdm import tqdm
import pathlib
import concurrent.futures


DEFAULT_MAX_TOKENS = 4000
DEFAULT_PARALLEL_REQUESTS = 20
DEFAULT_SEED = 0xC0FFEE
TIMEOUT = httpx.Timeout(60.0, connect=60.0, read=60.0)  # Use very generous timeouts

# Fix providers where necessary for reproducibility
PROVIDER_MAPPING = {
    "meta-llama/llama-3.1-8b-instruct": "DeepInfra",
    "meta-llama/llama-3.1-70b-instruct": "DeepInfra",
    "meta-llama/llama-3.1-405b-instruct": "DeepInfra",
}

# Values for model-dependent system prompts
SYSTEM_PROMPT_MAPPING = {
    "openai/gpt-4o-mini-2024-07-18": {
        "assistant": "GPT",
        "company": "OpenAI",
        "date": "September 1st, 2024",
        "cutoff": "October 2023",
    },
    "openai/gpt-4o-2024-05-13": {
        "assistant": "GPT",
        "company": "OpenAI",
        "date": "September 1st, 2024",
        "cutoff": "October 2023",
    },
    "openai/gpt-4-turbo-2024-04-09": {
        "assistant": "GPT",
        "company": "OpenAI",
        "date": "September 1st, 2024",
        "cutoff": "December 2023",
    },
    "anthropic/claude-3-haiku": {
        "assistant": "Claude",
        "company": "Anthropic",
        "date": "September 1st, 2024",
        "cutoff": "August 2023",
    },
    "anthropic/claude-3.5-sonnet": {
        "assistant": "Claude",
        "company": "Anthropic",
        "date": "September 1st, 2024",
        "cutoff": "April 2024",
    },
    "anthropic/claude-3-opus": {
        "assistant": "Claude",
        "company": "Anthropic",
        "date": "September 1st, 2024",
        "cutoff": "August 2023",
    },
    "meta-llama/llama-3.1-8b-instruct": {
        "assistant": "Llama",
        "company": "Meta",
        "date": "September 1st, 2024",
        "cutoff": "December 2023",
    },
    "meta-llama/llama-3.1-70b-instruct": {
        "assistant": "Llama",
        "company": "Meta",
        "date": "September 1st, 2024",
        "cutoff": "December 2023",
    },
    "meta-llama/llama-3.1-405b-instruct": {
        "assistant": "Llama",
        "company": "Meta",
        "date": "September 1st, 2024",
        "cutoff": "December 2023",
    },
    "openai/o1-mini-2024-09-12": {
        "assistant": "GPT",
        "company": "OpenAI",
        "date": "September 1st, 2024",
        "cutoff": "October 2023",
    },
    "openai/o1-preview-2024-09-12": {
        "assistant": "GPT",
        "company": "OpenAI",
        "date": "September 1st, 2024",
        "cutoff": "October 2023",
    },
    # Gemini cutoff dates do notseem to be known; just use max
    "google/gemini-1.5-flash-002": {
        "assistant": "Gemini",
        "company": "Google",
        "date": "September 1st, 2024",
        "cutoff": "September 2024",
    },
    "google/gemini-1.5-pro-002": {
        "assistant": "Gemini",
        "company": "Google",
        "date": "September 1st, 2024",
        "cutoff": "September 2024",
    },
}


async def main(args):
    model = args.model
    max_tokens = args.max_new_tokens
    temperature = args.temperature
    seeds = args.seeds
    rerun_errors_only = args.rerun_errors
    append_to_output = args.append
    format_system_prompt = args.format_system_prompt
    if rerun_errors_only and append_to_output:
        raise ValueError("--rerun-errors and --append are mutually exclusive")

    try:
        model_creator, model_name = model.split("/")
    except ValueError:
        raise ValueError(f"Model name {model} must be in the format `creator/model`")

    # Create appropriate API (OpenAI, OpenRouter, Google GenAI)
    if args.use_openai and args.use_google:
        raise ValueError(
            "Cannot use both OpenAI and Google GenAI APIs at the same time"
        )
    inference_api: InferenceApi = None

    if args.use_openai:
        print("Using OpenAI API directly")
        if model_creator != "openai":
            raise ValueError("OpenAI API can only be used with OpenAI models")

        # NB: This is meant to be used with an org, hence require both env vars for safety
        try:
            api_key = os.environ["OPENAI_API_KEY"]
            organization = os.environ["OPENAI_ORG_ID"]
        except KeyError:
            raise ValueError(
                "OPENAI_API_KEY and OPENAI_ORG_ID must be set in the environment"
            )
        inference_api = OpenAiApi(
            model=model_name,  # OpenAI api expects no prefix
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=TIMEOUT,
            organization=organization,
        )
    elif args.use_google:
        print("Using Google GenAI API directly")
        if model_creator != "google":
            raise ValueError("Google GenAI API can only be used with Google models")

        try:
            api_key = os.environ["GOOGLE_API_KEY"]
        except KeyError:
            raise ValueError("GOOGLE_API_KEY must be set in the environment")
        inference_api = GeminiApi(
            model=model_name,  # API expects no prefix
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=TIMEOUT,
        )

    else:
        print("Using OpenRouter")

        try:
            api_key = os.environ["OPENROUTER_API_KEY"]
        except KeyError:
            raise ValueError("OPENROUTER_API_KEY must be set in the environment")

        # Fixed provider (if available and not using OpenAI)
        fixed_provider = None
        if model in PROVIDER_MAPPING:
            fixed_provider = PROVIDER_MAPPING[model]
        elif model_creator not in ("anthropic",):
            # NB: Azure also serves some OAI models; need to specify OAI manually
            raise ValueError(
                f"Model {model} not found in provider mapping. Add it to the PROVIDER_MAPPING dictionary to avoid load balancing."
            )

        inference_api = OpenAiApi(
            model=model,
            api_key=api_key,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=TIMEOUT,
            base_url="https://openrouter.ai/api/v1",
            fixed_provider=fixed_provider,
        )

    assert inference_api is not None

    output_file = (
        args.output_path
        / args.data_path.stem
        / f"{model_name}_temp{str(temperature)}.jsonl"
    )

    if not rerun_errors_only:
        # Ensure the output path does not exist, otw throw an error
        if output_file.exists() and not append_to_output:
            raise ValueError(
                f"Output path {output_file} already exists. Delete before proceeding."
            )
        input_path = args.data_path
    else:
        if not output_file.exists():
            raise ValueError(
                f"Output path {output_file} does not exist. Cannot rerun errors."
            )
        print("Rerunnning errors only for", output_file)
        input_path = output_file
        output_file = output_file.with_name(output_file.name + ".rerun")

    # Load data
    with open(input_path, "r") as f:
        data = tuple(json.loads(line) for line in f)

    if args.use_first_n_prompts is not None:
        assert not rerun_errors_only, "Cannot use first N prompts when rerunning errors"
        data = data[: args.use_first_n_prompts]

    # Instantiate system prompt templates if specified
    if format_system_prompt:
        assert (
            not rerun_errors_only
        ), "Cannot format system prompts when rerunning errors; might have unintended consequences"
        try:
            system_prompt_parameters = SYSTEM_PROMPT_MAPPING[
                f"{model_creator}/{model_name}"
            ]
        except KeyError:
            raise ValueError(
                f"System prompt parameters unavailable for {model_creator}/{model_name}"
            )
        for prompt_data in data:
            for message in prompt_data["messages"]:
                if message["role"] == "system":
                    message["content"] = message["content"].format(
                        **system_prompt_parameters
                    )

    # Create the output path if not exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("Starting inference")

    executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=args.max_parallel_requests,
    )

    def _run_inference(
        input_data: typing.Dict[str, typing.Any],
        seed: int,
    ) -> typing.Dict[str, typing.Any]:
        output_data = {**input_data, "seed": seed}

        if rerun_errors_only and "error" not in input_data:
            assert "completion" in output_data
            return output_data

        try:
            response, finish_reason = inference_api.generate_completion(
                messages=input_data["messages"],
                seed=seed,
            )

            output_data["completion"] = response
            output_data["finish_reason"] = finish_reason
            return output_data
        except Exception as ex:
            raise CompletionException(output_data, ex)

    if not rerun_errors_only:
        assert len(seeds) > 0, "Need at least one seed"
        futures = tuple(
            asyncio.wrap_future(executor.submit(_run_inference, prompt_data, seed))
            for prompt_data in data
            for seed in seeds
        )
    else:
        assert all(
            "seed" in prompt_data and isinstance(prompt_data["seed"], int)
            for prompt_data in data
        ), "All data must have a seed if rerunning errors"
        futures = tuple(
            asyncio.wrap_future(
                executor.submit(_run_inference, prompt_data, prompt_data["seed"])
            )
            for prompt_data in data
        )

    with open(output_file, "a" if append_to_output else "w") as f:
        num_errors = 0
        for future in (pbar := tqdm(asyncio.as_completed(futures), total=len(futures))):
            try:
                result = await future
                # Remove previous error if succesfully reran
                if rerun_errors_only and "error" in result:
                    del result["error"]
            except CompletionException as ex:
                num_errors += 1
                pbar.write(f"Error processing future: {ex}")
                pbar.set_postfix({"errors": num_errors})
                result = {**ex.input_data, "completion": None, "error": str(ex)}

            json.dump(result, f)
            f.write("\n")

    if num_errors > 0:
        print(f"Finished with {num_errors} errors")


class InferenceApi(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def generate_completion(
        self,
        messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam],
        seed: int,
    ) -> typing.Tuple[str, str]:
        pass


class OpenAiApi(InferenceApi):
    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        timeout: httpx.Timeout,
        organization: str | None = None,
        fixed_provider: str | None = None,
        base_url: str | None = None,
    ):
        self._model = model

        # FIXME: This is a dumb hack b/c of random API changes,
        #   so we have to do this distinction to avoid breaking OpenRouter and other things...
        #   Only relevant if we use o1 with the OAI API
        if self._model.startswith("o1-"):
            self._max_tokens = openai.NOT_GIVEN
            self._max_completion_tokens = max_tokens
        else:
            self._max_tokens = max_tokens
            self._max_completion_tokens = openai.NOT_GIVEN
        self._temperature = temperature

        self._client = openai.OpenAI(
            api_key=api_key,
            organization=organization,
            base_url=base_url,
            timeout=timeout,
        )

        self._extra_body = None
        if fixed_provider is not None:
            self._extra_body = {
                "provider": {
                    "order": [fixed_provider],
                    "allow_fallbacks": False,
                }
            }

    def generate_completion(
        self,
        messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam],
        seed: int,
    ) -> typing.Tuple[str, str]:
        completion = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=self._max_tokens,
            max_completion_tokens=self._max_completion_tokens,
            temperature=self._temperature,
            extra_body=self._extra_body,
            seed=seed,
        )
        # Check for errors
        if "error" in completion.model_extra:
            raise ValueError(f"Error in completion: {completion.model_extra['error']}")
        # Store finish reason to debug if some completions ended early
        finish_reason = completion.choices[0].finish_reason
        if finish_reason not in (
            "stop",
            "end_turn",
            "eos",
            "length",
            "max_tokens",
        ):
            raise ValueError(f"Unexpected finish reason: {finish_reason}")
        response = completion.choices[0].message.content
        if response is None:
            raise ValueError("No response")

        return response, finish_reason


class GeminiApi(InferenceApi):
    # The Python API does not support seeds, but things work out via the REST API

    # These categories are actually supported
    _HARM_CATEGORIES = (
        "HARM_CATEGORY_HARASSMENT",
        "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "HARM_CATEGORY_DANGEROUS_CONTENT",
        "HARM_CATEGORY_CIVIC_INTEGRITY",
    )

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int,
        temperature: float,
        timeout: httpx.Timeout,
    ):
        self._model_name = model
        self._api_key = api_key
        self._timeout = timeout

        self._base_config = {
            "candidateCount": 1,
            "maxOutputTokens": max_tokens,
            "topP": 1.0,
            "temperature": temperature,
            "response_logprobs": False,
        }

        # We want as few accidental refusals as possible
        self._safety_settings = tuple(
            {"category": category, "threshold": "BLOCK_NONE"}
            for category in self._HARM_CATEGORIES
        )

    def generate_completion(
        self,
        messages: typing.Iterable[openai.types.chat.ChatCompletionMessageParam],
        seed: int,
    ) -> typing.Tuple[str, str]:
        generation_config = {**self._base_config, "seed": seed}
        messages = tuple(messages)

        (user_prompt,) = (
            message["content"] for message in messages if message["role"] == "user"
        )
        if len(messages) > 1:
            assert (
                len(messages) == 2
            ), "Messages must contain user prompt and optional system prompt"
            (system_prompt,) = (
                message["content"]
                for message in messages
                if message["role"] == "system"
            )
        else:
            system_prompt = None

        request_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._model_name}:generateContent?key={self._api_key}"

        request_data = {
            "contents": [{"parts": [{"text": user_prompt}]}],
            "safety_settings": self._safety_settings,
            "generation_config": generation_config,
        }
        if system_prompt is not None:
            request_data["system_instruction"] = {
                "parts": [{"text": system_prompt}],
            }

        raw_response = requests.post(
            request_url,
            json=request_data,
            timeout=self._timeout.read,
        )
        raw_response_data = raw_response.json()
        if "error" in raw_response_data:
            raise ValueError(f"Error in completion: {raw_response_data['error']}")
        raw_response.raise_for_status()  # in case we did not catch something else

        if "candidates" not in raw_response_data:
            raise ValueError(f"Raw response has no candidates: {raw_response_data}")

        assert len(raw_response_data["candidates"]) == 1
        generation = raw_response_data["candidates"][0]

        # Store finish reason to debug if some completions ended early
        finish_reason = generation["finishReason"]
        if finish_reason not in (
            "STOP",
            "MAX_TOKENS",
        ):
            raise ValueError(f"Unexpected finish reason: {finish_reason}")

        assert len(generation["content"]["parts"]) == 1
        response = generation["content"]["parts"][0]["text"]
        if response is None or response.strip() == "":
            raise ValueError("No response")

        return response, finish_reason


# Wraps an exception with input data
class CompletionException(Exception):
    def __init__(self, input_data: typing.Dict[str, typing.Any], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_data = input_data


def _parse_args() -> argparse.Namespace:
    # Parse a python argument for modelname
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="model name")
    parser.add_argument(
        "--data_path", help="path to JSONL containing the test data", type=pathlib.Path
    )
    parser.add_argument(
        "--output_path",
        help="path to a root results folder where a file named {dataset_name}/{model_name}_temp{temperature}.jsonl will be stored",
        type=pathlib.Path,
        default="results",
    )
    parser.add_argument(
        "--temperature", help="temperature for sampling", type=float, default=0.0
    )
    parser.add_argument(
        "--max_new_tokens",
        help="maximum number of new tokens to generate",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    parser.add_argument(
        "--use_first_n_prompts",
        help="number of prompts to use for the experiments. First N will be taken.",
        type=int,
        required=False,
    )
    parser.add_argument(
        "--max-parallel-requests",
        help="Maximum number of parallel API requests",
        type=int,
        default=DEFAULT_PARALLEL_REQUESTS,
    )
    parser.add_argument(
        "--use-openai",
        help="Directly use the OpenAI API. Requires API key (and org) in the environment, and will fail with other models",
        action="store_true",
    )
    parser.add_argument(
        "--use-google",
        help="Directly use the Google GenAI API. Requires API key in the environment, and will fail with other models",
        action="store_true",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=(DEFAULT_SEED,),
        nargs="+",
        help="Seed for reproducibility",
    )
    parser.add_argument(
        "--rerun-errors",
        help="If set, input data should be the result of a previous run, and this script will only rerun and replace rows with errors",
        action="store_true",
    )
    parser.add_argument(
        "--append",
        help="If the output file already exists, append instead of raising an error",
        action="store_true",
    )
    parser.add_argument(
        "--format-system-prompt",
        help="If set, the system prompt is expected to be a template to be filled with the model name, company, date, and cutoff date",
        action="store_true",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(main(args))
