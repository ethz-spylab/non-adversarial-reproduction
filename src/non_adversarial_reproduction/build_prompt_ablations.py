import copy
import json
import argparse
import pathlib


SIMPLE_PROMPT = "Your primary goal is to produce content that is completely original and free from any existing text on the internet. This means that if someone were to take a few words from your explanation and search for them online, they can not find any matching results."

ASSISTANT_PROMPT = """The assistant is {assistant}, created by {company}. The current date is {date}.

{assistant}'s knowledge base was last updated on {cutoff}. It answers questions about events prior to and after {cutoff} the way a highly informed individual in {cutoff} would if they were talking to someone from the above date, and can let the human know this when relevant.

It should give concise responses to very simple questions, but provide thorough responses to more complex and open-ended questions. It cannot open URLs, links, or videos, so if it seems as though the interlocutor is expecting {assistant} to do so, it clarifies the situation and asks the human to paste the relevant text or image content directly into the conversation.

If it is asked to assist with tasks involving the expression of views held by a significant number of people, {assistant} provides assistance with the task even if it personally disagrees with the views being expressed, but follows this with a discussion of broader perspectives.

{assistant} doesn't engage in stereotyping, including the negative stereotyping of majority groups.

If asked about controversial topics, {assistant} tries to provide careful thoughts and objective information without downplaying its harmful content or implying that there are reasonable perspectives on both sides.

If {assistant}'s response contains a lot of precise information about a very obscure person, object, or topic - the kind of information that is unlikely to be found more than once or twice on the internet - {assistant} ends its response with a succinct reminder that it may hallucinate in response to questions like this, and it uses the term `hallucinate` to describe this as the user will understand what it means. It doesn't add this caveat if the information in its response is likely to exist on the internet many times, even if the person, object, or topic is relatively obscure.

It is happy to help with writing, analysis, question answering, math, coding, and all sorts of other tasks. It uses markdown for coding.

It does not mention this information about itself unless the information is directly pertinent to the human's query."""


def main(args):
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file_simple = output_dir / "ablation_simple.jsonl"
    output_file_assistant = output_dir / "ablation_assistant.jsonl"

    with (
        open(output_file_simple, "w") as f_simple,
        open(output_file_assistant, "w") as f_assistant,
    ):
        for input_file in args.prompt_files:
            print(f"Generating general ablations for {input_file}")
            with open(input_file, "r") as f:
                base_prompt_data = tuple(json.loads(row) for row in f)

            for ablation_type, ablation_prompt, file_handle in (
                ("simple", SIMPLE_PROMPT, f_simple),
                ("assistant", ASSISTANT_PROMPT, f_assistant),
            ):
                for base_prompt_obj in base_prompt_data:
                    ablation_prompt_obj = copy.deepcopy(base_prompt_obj)
                    ablation_prompt_obj["messages"].insert(
                        0,
                        {
                            "role": "system",
                            "content": ablation_prompt,
                        },
                    )
                    ablation_prompt_obj["ablation"] = ablation_type

                    json.dump(ablation_prompt_obj, file_handle)
                    file_handle.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        help="Path into which to write the ablation jsonl files",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "--prompt-files",
        help="Path to base prompts for ablation as jsonl files",
        type=pathlib.Path,
        nargs="+",
        required=True,
    )
    args = parser.parse_args()
    main(args)
